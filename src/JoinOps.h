#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "Block.h"
#include "BlockStore.h"
#include "Hash.h"
#include "JoinOutput.h"
#include "TwoLevelHashTable.h"
#include "Types.h"


namespace phj
{

namespace detail
{

/// Dispatch to the locked or unlocked HT insert based on table type.
/// CHJ uses `TwoLevelJoinHashTable` (per-bucket locks); PHJ uses
/// `JoinHashTable` (single-writer, no lock).
template <class HT>
[[gnu::always_inline]] inline RowRefCell htInsertDispatch(HT & ht, uint64_t h, uint64_t k, RowRefCell r)
{
    if constexpr (std::is_same_v<HT, TwoLevelJoinHashTable>)
        return ht.insertLocked(h, k, r);
    else
        return ht.insert(h, k, r);
}


/// Dispatch to the locked or unlocked BlockStore append based on HT
/// type — the synchronisation requirements are coupled (a shared
/// hashtable implies a shared block store).
template <class HT>
[[gnu::always_inline]] inline BlockNo storeAppendDispatch(BlockStore & store, Block && block)
{
    if constexpr (std::is_same_v<HT, TwoLevelJoinHashTable>)
        return store.appendLocked(std::move(block));
    else
        return store.append(std::move(block));
}

}


/// Probe-side per-worker materialiser. Maintains one in-progress
/// `OutputBlock` of capacity `PIPELINE_BLOCK_ROWS`; emits to
/// `worker.blocks` whenever the in-progress block fills.
///
/// `gatherFromProbeBlock` performs the column-major projection: one
/// pass per output column over the matched-row index list, gathering
/// build-side values via `(block_no, row_no)` against `build_store`
/// and probe-side values via the probe block's row index. The matched
/// arrays describe `m` output rows total; the projection chunks them
/// across output blocks as needed.
class ProbeMaterialiser
{
public:
    void init(const PayloadSchema & left_schema, const PayloadSchema & right_schema, OutputWorker & worker, size_t out_block_rows)
    {
        left_schema_ = &left_schema;
        right_schema_ = &right_schema;
        worker_ = &worker;
        out_block_rows_ = out_block_rows;
        left_sizes_.assign(left_schema.types.size(), 0);
        right_sizes_.assign(right_schema.types.size(), 0);
        for (size_t c = 0; c < left_schema.types.size(); ++c)
            left_sizes_[c] = payloadTypeSize(left_schema.types[c]);
        for (size_t c = 0; c < right_schema.types.size(); ++c)
            right_sizes_[c] = payloadTypeSize(right_schema.types[c]);
        ensureInProgress();
    }

    /// Project `m` matched rows into output blocks. `probe_idx[k]` is
    /// the probe row index in `probe` and `build_ref[k]` is the
    /// build-side cell, both for the k-th matched output row.
    ///
    /// Column-major loop nest: outer over columns, inner over the
    /// matched-row chunk that fits in the current in-progress
    /// output block.
    void
    gatherFromProbeBlock(BlockView probe, const BlockStore & build_store, const size_t * probe_idx, const RowRefCell * build_ref, size_t m)
    {
        size_t processed = 0;
        while (processed < m)
        {
            ensureInProgress();
            OutputBlock & cur = in_progress_;
            const size_t remaining_cap = out_block_rows_ - cur.rows;
            const size_t take = std::min(m - processed, remaining_cap);

            /// Join key column.
            for (size_t k = 0; k < take; ++k)
                cur.keys[cur.rows + k] = probe.keys[probe_idx[processed + k]];

            /// Left (build-side) payload columns: column-major gather
            /// across blocks via `(block_no, row_no)`.
            for (size_t c = 0; c < left_schema_->types.size(); ++c)
            {
                const size_t sz = left_sizes_[c];
                std::byte * dst_base = cur.left[c].raw() + cur.rows * sz;
                for (size_t k = 0; k < take; ++k)
                {
                    const RowRefCell ref = build_ref[processed + k];
                    const std::byte * src = build_store.block(ref.block_no).payloads[c].raw() + static_cast<size_t>(ref.row_no) * sz;
                    copySized(dst_base + k * sz, src, sz);
                }
            }

            /// Right (probe-side) payload columns: column-major gather
            /// within the probe block.
            for (size_t c = 0; c < right_schema_->types.size(); ++c)
            {
                const size_t sz = right_sizes_[c];
                std::byte * dst_base = cur.right[c].raw() + cur.rows * sz;
                for (size_t k = 0; k < take; ++k)
                {
                    const std::byte * src = probe.payloads[c].data.data() + probe_idx[processed + k] * sz;
                    copySized(dst_base + k * sz, src, sz);
                }
            }

            cur.rows += take;
            processed += take;

            if (cur.rows == out_block_rows_)
                flushInProgress();
        }
    }

    /// End-of-stream: emit a (possibly partial) final output block.
    void finish()
    {
        if (in_progress_.rows > 0)
            flushInProgress();
    }

private:
    const PayloadSchema * left_schema_ = nullptr;
    const PayloadSchema * right_schema_ = nullptr;
    OutputWorker * worker_ = nullptr;
    size_t out_block_rows_ = PIPELINE_BLOCK_ROWS;
    std::vector<size_t> left_sizes_;
    std::vector<size_t> right_sizes_;

    OutputBlock in_progress_;
    bool ip_allocated_ = false;

    void ensureInProgress()
    {
        if (!ip_allocated_)
        {
            in_progress_.init(*left_schema_, *right_schema_, out_block_rows_);
            ip_allocated_ = true;
        }
    }

    void flushInProgress()
    {
        if (!ip_allocated_)
            return;
        /// Truncate ownership of the in-progress block down to its
        /// actual filled row count and emit. We move it into the
        /// worker's vector and re-allocate a fresh in-progress block
        /// next time it's needed.
        in_progress_.keys.resize(in_progress_.rows);
        for (auto & col : in_progress_.left)
            col.data.resize(in_progress_.rows * payloadTypeSize(col.type));
        for (auto & col : in_progress_.right)
            col.data.resize(in_progress_.rows * payloadTypeSize(col.type));
        worker_->blocks.push_back(std::move(in_progress_));
        in_progress_ = OutputBlock{};
        ip_allocated_ = false;
    }

    [[gnu::always_inline]] static inline void copySized(std::byte * dst, const std::byte * src, size_t sz) noexcept
    {
        switch (sz)
        {
            case 1:
                *dst = *src;
                return;
            case 2:
                std::memcpy(dst, src, 2);
                return;
            case 4:
                std::memcpy(dst, src, 4);
                return;
            case 8:
                std::memcpy(dst, src, 8);
                return;
            case 16:
                std::memcpy(dst, src, 16);
                return;
            default:
                std::memcpy(dst, src, sz);
                return;
        }
    }
};


/// Vectorised build for one input block: append to `store`, SIMD-hash
/// the keys, batch-insert into `ht`, and thread each row's previous
/// chain ref into the store's next-chain. Templated on the HT type:
/// `TwoLevelJoinHashTable` selects the locked-append + locked-insert
/// path (CHJ), `JoinHashTable` selects the single-writer path (PHJ).
/// `hashes` is a worker-owned scratch buffer reused across calls.
template <class HT>
void buildOneBlock(Block && block, BlockStore & store, HT & ht, std::vector<uint64_t> & hashes)
{
    const size_t rows = block.rows;
    if (rows == 0)
        return;

    hashes.resize(rows);
    intHash64Batch(block.keyData(), rows, hashes.data());

    /// std::vector's move ctor preserves the underlying buffer pointer,
    /// so the pointer captured here remains valid after the block is
    /// moved into the store. The store's `blocks_` vector is pre-
    /// reserved by the caller so concurrent appends never reallocate.
    const uint64_t * keys = block.keyData();
    const BlockNo block_no = detail::storeAppendDispatch<HT>(store, std::move(block));

    for (size_t i = 0; i < rows; ++i)
    {
        const RowRefCell ref{block_no, static_cast<RowNo>(i)};
        const RowRefCell prev = detail::htInsertDispatch(ht, hashes[i], keys[i], ref);
        store.setPrev(ref, prev);
    }
}


/// Vectorised probe for one input block view: SIMD-hash the keys,
/// batch-find against `ht`, walk multi-match chains in `build_store`
/// to produce flat `(probe_idx, build_ref)` arrays, then column-major
/// project them into output blocks via `out`. Caller owns the
/// scratch buffers (reused across calls within a worker).
template <class HT>
void probeOneBlock(
    BlockView block,
    const BlockStore & build_store,
    const HT & ht,
    ProbeMaterialiser & out,
    std::vector<uint64_t> & hashes,
    std::vector<RowRefCell> & heads,
    std::vector<size_t> & probe_idx,
    std::vector<RowRefCell> & build_ref)
{
    const size_t rows = block.rows;
    if (rows == 0)
        return;

    hashes.resize(rows);
    heads.resize(rows);
    intHash64Batch(block.keys, rows, hashes.data());
    ht.batchFind(hashes.data(), block.keys, heads.data(), rows);

    probe_idx.clear();
    build_ref.clear();
    probe_idx.reserve(rows);
    build_ref.reserve(rows);
    for (size_t i = 0; i < rows; ++i)
    {
        RowRefCell ref = heads[i];
        while (ref.valid())
        {
            probe_idx.push_back(i);
            build_ref.push_back(ref);
            ref = build_store.getPrev(ref);
        }
    }

    out.gatherFromProbeBlock(block, build_store, probe_idx.data(), build_ref.data(), probe_idx.size());
}

}
