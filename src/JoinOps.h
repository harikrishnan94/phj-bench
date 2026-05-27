#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
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

/// Single-writer HT insert (PHJ partitions; CHJ slot-under-mutex).
template <class HT>
[[gnu::always_inline]] inline RowRefCell htInsertDispatch(HT & ht, uint64_t h, uint64_t k, RowRefCell r)
{
    return ht.insert(h, k, r);
}

/// Single-writer BlockStore append (PHJ partitions; CHJ slot-under-mutex).
template <class HT>
[[gnu::always_inline]] inline BlockNo storeAppendDispatch(BlockStore & store, Block && block)
{
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
/// the keys, insert into `ht`, and thread each row's previous chain ref
/// into the store's next-chain. `HT` must be `JoinHashTable`; the caller
/// is responsible for serialising concurrent access (per-slot mutex in CHJ,
/// single-writer partition ownership in PHJ). `hashes` is a worker-owned
/// scratch buffer reused across calls.
template <class HT>
void buildOneBlock(Block && block, BlockStore & store, HT & ht, std::pmr::vector<uint64_t> & hashes)
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
    std::pmr::vector<uint64_t> & hashes,
    std::pmr::vector<RowRefCell> & heads,
    std::pmr::vector<size_t> & probe_idx,
    std::pmr::vector<RowRefCell> & build_ref)
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


/// CH-style scattered build for one input block into a slotted table.
///
/// Mirrors ClickHouse's ConcurrentHashJoin::addBlockToJoin exactly:
///  1. SIMD-hash all keys.
///  2. Bin row indices by `slot = hash & (n_slots - 1)` (→ dispatchBlock).
///  3. For each non-empty bin, scatter the selected rows into a sub-Block
///     allocated via `mr` so that memory is tracked.
///  4. Insert the sub-blocks with std::try_to_lock: if a slot's mutex is
///     already held by another thread, skip it and try the next slot.
///     Only yield when an entire pass produced no progress (all remaining
///     slots were contested). This is the same spin-on-try-lock loop CH
///     uses to avoid blocking while other slots remain free.
///
/// `hashes` and `slot_rows` are worker-local scratch reused across calls;
/// `hashes` is overwritten by each `buildOneBlock` sub-call.
inline void buildOneBlockCHJ(
    const Block & src,
    ChjSlottedTable & table,
    std::pmr::vector<uint64_t> & hashes,
    std::vector<std::vector<size_t>> & slot_rows,
    std::pmr::memory_resource * mr)
{
    const size_t rows = src.rows;
    if (rows == 0)
        return;

    const size_t n_slots = table.numSlots();

    hashes.resize(rows);
    intHash64Batch(src.keyData(), rows, hashes.data());

    for (size_t s = 0; s < n_slots; ++s)
        slot_rows[s].clear();
    for (size_t i = 0; i < rows; ++i)
        slot_rows[table.slotOf(hashes[i])].push_back(i);

    // Build all per-slot sub-blocks upfront (= CH's dispatchBlock).
    // The pmr vector propagates mr to each Block element so their keys
    // and payload data are allocated through the MemTracker.
    std::pmr::vector<Block> sub_blocks(n_slots, std::pmr::polymorphic_allocator<Block>(mr));
    size_t blocks_left = 0;
    for (size_t s = 0; s < n_slots; ++s)
    {
        const auto & ridx = slot_rows[s];
        if (ridx.empty())
            continue;

        Block & sub = sub_blocks[s];
        sub.rows = ridx.size();
        sub.keys.resize(sub.rows);
        sub.payloads.resize(src.payloads.size());

        for (size_t k = 0; k < sub.rows; ++k)
            sub.keys[k] = src.keys[ridx[k]];

        for (size_t c = 0; c < src.payloads.size(); ++c)
        {
            sub.payloads[c].type = src.payloads[c].type;
            const size_t esz = src.payloads[c].elementSize();
            sub.payloads[c].data.resize(sub.rows * esz);
            for (size_t k = 0; k < sub.rows; ++k)
                std::memcpy(sub.payloads[c].raw() + k * esz, src.payloads[c].raw() + ridx[k] * esz, esz);
        }
        ++blocks_left;
    }

    // Spin-on-try-lock insert loop (= CH's addBlockToJoin while loop).
    // Try every pending slot; skip contested ones and only yield when no
    // slot was acquirable in the entire pass.
    while (blocks_left > 0)
    {
        bool made_progress = false;
        for (size_t s = 0; s < n_slots; ++s)
        {
            Block & sub = sub_blocks[s];
            if (sub.rows == 0)
                continue;

            auto & sl = table.slot(s);
            std::unique_lock lock(sl.mu, std::try_to_lock);
            if (!lock.owns_lock())
                continue;

            made_progress = true;
            buildOneBlock(std::move(sub), sl.store, sl.ht, hashes);
            sub.rows = 0; // mark as inserted (rows is not reset by move)
            --blocks_left;
        }
        if (!made_progress)
            std::this_thread::yield();
    }
}


/// CH-style scattered probe for one input block view against a slotted table.
///
/// Hashes all probe rows, bins them by `slot = hash & (n_slots - 1)`, then
/// for each slot probes its independent JoinHashTable and gathers matching
/// (probe_row_idx, build_ref) pairs into `out` via `gatherFromProbeBlock`.
///
/// No sub-Block copy is needed on the probe side: `gatherFromProbeBlock`
/// accepts the full `probe` view and indexes specific rows via `probe_idx`,
/// which holds the original row indices within `probe`. This mirrors how
/// ClickHouse routes probe rows to each slot's HashJoin.
///
/// Entirely lock-free after the build barrier (each slot's table is read-only).
inline void probeOneBlockCHJ(
    BlockView probe,
    const ChjSlottedTable & table,
    ProbeMaterialiser & out,
    std::pmr::vector<uint64_t> & hashes,
    std::pmr::vector<size_t> & probe_idx,
    std::pmr::vector<RowRefCell> & build_ref,
    std::vector<std::vector<size_t>> & slot_rows)
{
    const size_t rows = probe.rows;
    if (rows == 0)
        return;

    const size_t n_slots = table.numSlots();

    hashes.resize(rows);
    intHash64Batch(probe.keys, rows, hashes.data());

    for (size_t s = 0; s < n_slots; ++s)
        slot_rows[s].clear();
    for (size_t i = 0; i < rows; ++i)
        slot_rows[table.slotOf(hashes[i])].push_back(i);

    for (size_t s = 0; s < n_slots; ++s)
    {
        const auto & ridx = slot_rows[s];
        if (ridx.empty())
            continue;

        const auto & sl = table.slot(s);

        probe_idx.clear();
        build_ref.clear();

        for (const size_t ri : ridx)
        {
            RowRefCell ref = sl.ht.find(hashes[ri], probe.keys[ri]);
            while (ref.valid())
            {
                probe_idx.push_back(ri);
                build_ref.push_back(ref);
                ref = sl.store.getPrev(ref);
            }
        }

        if (!probe_idx.empty())
            out.gatherFromProbeBlock(probe, sl.store, probe_idx.data(), build_ref.data(), probe_idx.size());
    }
}

}
