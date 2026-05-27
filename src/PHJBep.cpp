#include "PHJBep.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

#include "BlockStore.h"
#include "Hash.h"
#include "HashTable.h"
#include "JoinOps.h"
#include "MemTracker.h"
#include "Threading.h"
#include "Timer.h"


namespace phj
{

namespace
{

/// Release the trailing unused capacity of an `OutBlock` by re-
/// allocating its key and payload buffers at exactly `rows` slots.
/// Used after the build shuffle.
void compactOutBlock(OutBlock & blk)
{
    if (blk.rows >= blk.capacity)
        return;
    auto * mr = blk.keys.get_allocator().resource();
    const auto rows_diff = static_cast<std::ptrdiff_t>(blk.rows);
    std::pmr::vector<uint64_t> new_keys(mr);
    new_keys.assign(blk.keys.begin(), blk.keys.begin() + rows_diff);
    blk.keys = std::move(new_keys);
    for (auto & col : blk.payloads)
    {
        const size_t sz = payloadTypeSize(col.type);
        const auto bytes_diff = static_cast<std::ptrdiff_t>(blk.rows * sz);
        std::pmr::vector<std::byte> new_data(mr);
        new_data.assign(col.data.begin(), col.data.begin() + bytes_diff);
        col.data = std::move(new_data);
    }
    blk.capacity = blk.rows;
}


/// Per-buffered-block partition index. After `partitionOne`:
///   `sorted_idx` holds all row indices of the block sorted by leaf
///   partition id (counting-sort layout). `p_offsets[L]` is the start
///   offset in `sorted_idx` for leaf L; `p_offsets[total_leaves]` is
///   the row count (== buffered_block.rows).
///
/// All buffers are scratch — they live for the duration of one flush.
/// `std::vector` (not `pmr`) so the outer container's allocator-aware
/// `emplace_back()` doesn't need this type to forward through pmr's
/// uses-allocator construction machinery.
struct BlockPartitioning
{
    std::vector<uint32_t> sorted_idx;
    std::vector<uint32_t> p_offsets;
};


/// Probe a subset of rows from one buffered probe block against one
/// leaf's pre-built HT. `sel_idx` are the row indices to probe, all
/// pointing into `probe_block`; `precomputed_hashes` is the worker's
/// per-block hash array (so we don't re-hash). The materialiser's
/// existing `gatherFromProbeBlock` is reused unchanged — it accepts
/// a `probe_idx` array that holds original block-row indices, which
/// is exactly what `sel_idx` provides.
///
/// This is the kernel of Idea 2: no physical scatter, only a selection
/// vector. We pay one (key, hash) gather per row at probe time and one
/// payload gather per output row at materialise time; both reads
/// originate from the original buffered block (which sits in the
/// thread-local working set).
template <class HT>
[[gnu::always_inline]] inline void probeSubsetOfBuffered(
    BlockView probe_block,
    const uint32_t * sel_idx,
    size_t n_sel,
    const uint64_t * precomputed_hashes,
    const HT & ht,
    const BlockStore & store,
    ProbeMaterialiser & mat,
    std::pmr::vector<uint64_t> & sub_hashes,
    std::pmr::vector<uint64_t> & sub_keys,
    std::pmr::vector<RowRefCell> & heads,
    std::pmr::vector<size_t> & probe_idx,
    std::pmr::vector<RowRefCell> & build_ref)
{
    if (n_sel == 0)
        return;

    sub_hashes.resize(n_sel);
    sub_keys.resize(n_sel);
    for (size_t k = 0; k < n_sel; ++k)
    {
        const uint32_t i = sel_idx[k];
        sub_keys[k] = probe_block.keys[i];
        sub_hashes[k] = precomputed_hashes[i];
    }

    heads.resize(n_sel);
    ht.batchFind(sub_hashes.data(), sub_keys.data(), heads.data(), n_sel);

    probe_idx.clear();
    build_ref.clear();
    probe_idx.reserve(n_sel);
    build_ref.reserve(n_sel);
    for (size_t k = 0; k < n_sel; ++k)
    {
        RowRefCell ref = heads[k];
        while (ref.valid())
        {
            probe_idx.push_back(sel_idx[k]);
            build_ref.push_back(ref);
            ref = store.getPrev(ref);
        }
    }

    if (probe_idx.empty())
        return;
    mat.gatherFromProbeBlock(probe_block, store, probe_idx.data(), build_ref.data(), probe_idx.size());
}

}


/// PHJ-BEP — IDEA 2: selection-index buffering, no physical scatter.
///
/// Departure from baseline BEP AND from Idea 1: there is no probe-side
/// radix scatter at all (no per-thread / per-leaf OutBlock chains, no
/// refinement). Each worker accumulates DEEP COPIES of input probe
/// blocks into a worker-local buffer pool up to its per-worker share
/// of `bep_budget_mib`. When the buffered row-bytes cross the per-
/// worker budget, the worker:
///
///   1. SIMD-hashes the freshly-added block(s) (incrementally, so a
///      block's hashes are computed once and reused).
///   2. Builds a per-block leaf-partitioning via counting sort over
///      the precomputed hashes. Output is `(sorted_idx[], p_offsets[])`
///      per block — a selection vector indexed by leaf id; the block
///      payloads stay in place.
///   3. Iterates leaves L = 0..total_leaves; for each leaf, walks
///      every buffered block and probes the subset `sorted_idx[
///      p_offsets[L]..p_offsets[L+1])` against the pre-built leaf HT.
///      No payload movement: keys are gathered from the original
///      buffered block; the existing `gatherFromProbeBlock` uses the
///      selection vector to gather payload columns directly from the
///      same buffered block at materialise time.
///   4. Drops the entire buffer pool. Resume streaming.
///
/// All leaf HTs are constructed up-front (same as Idea 1) so the probe
/// phase sees them as read-only and needs no synchronisation.
///
/// Hypothesis: the baseline scatter writes ~`bytes_per_row` per probe
/// row at scatter time (every payload column, every row). Idea 2
/// replaces those writes with a 4-byte selection-index write per row
/// and a single 8-byte hash buffer write. For 68 B/row × 25M rows /
/// worker × 16 workers = 27 GB of saved write traffic.
PhjBepResult runPhjBep(const BlockStream & build, const BlockStream & probe, const RadixConfig & cfg, size_t threads, size_t bep_budget_mib)
{
    if (threads == 0)
        threads = 1;

    MemTracker tracker;

    PhjBepResult result;
    result.output.left_schema = build.schema;
    result.output.right_schema = probe.schema;
    result.output.workers.resize(threads);
    result.bep_budget_mib = bep_budget_mib;

    const TimePoint t_e2e0 = now();

    /// -------- BUILD SHUFFLE (full leaf depth, identical to PHJ-PURE) --------
    const TimePoint t_bs0 = now();
    PartitionedShuffleOutput build_part = radixShuffle(build, cfg, threads, &tracker);

    parallelRun(
        threads,
        [&](size_t tid)
        {
            const size_t n = build_part.chains.size();
            for (size_t p = tid; p < n; p += threads)
                for (auto & blk : build_part.chains[p].blocks)
                    compactOutBlock(blk);
        });
    const TimePoint t_bs1 = now();
    const uint64_t build_shuffle_ns = toNanos(t_bs1 - t_bs0);

    const size_t total_leaves = build_part.partitions;
    const uint8_t total_bits = cfg.totalBits();
    const uint32_t leaf_shift = static_cast<uint32_t>(64u - total_bits);
    const uint64_t leaf_mask = total_leaves - 1;
    const size_t bytes_per_row = sizeof(uint64_t) + probe.schema.rowByteSize();

    /// Worker-local view of the global budget. The budget is split
    /// strictly per-worker, no sharing.
    const size_t budget_bytes = bep_budget_mib * size_t{1024} * size_t{1024};
    const size_t per_worker_budget = budget_bytes / threads;

    /// -------- LEAF HT STORE + EAGER BUILD --------
    std::pmr::vector<JoinHashTable> leaf_hts(&tracker);
    leaf_hts.resize(total_leaves);
    std::vector<std::unique_ptr<BlockStore>> leaf_stores(total_leaves);
    for (auto & p : leaf_stores)
        p = std::make_unique<BlockStore>(&tracker);

    std::vector<uint64_t> ns_build(threads, 0);
    std::atomic<size_t> next_build_leaf{0};

    parallelRun(
        threads,
        [&](size_t tid)
        {
            std::pmr::vector<uint64_t> build_hashes(&tracker);
            uint64_t my_ns = 0;
            while (true)
            {
                const size_t L = next_build_leaf.fetch_add(1, std::memory_order_relaxed);
                if (L >= total_leaves)
                    break;
                if (build_part.partition_rows[L] == 0)
                {
                    build_part.chains[L].blocks.clear();
                    continue;
                }
                const TimePoint t0 = now();
                BlockStore & store = *leaf_stores[L];
                store.reserveBlocks(build_part.chains[L].blocks.size());
                JoinHashTable & ht = leaf_hts[L];
                ht.reserve(build_part.partition_rows[L]);
                for (auto & ob : build_part.chains[L].blocks)
                {
                    Block as_block = outBlockToBlock(std::move(ob));
                    buildOneBlock(std::move(as_block), store, ht, build_hashes);
                }
                build_part.chains[L].blocks.clear();
                my_ns += toNanos(now() - t0);
            }
            ns_build[tid] = my_ns;
        });

    /// -------- PER-THREAD PROBE TIMING + COUNTERS --------
    std::vector<uint64_t> ns_probe_shuffle(threads, 0);
    std::vector<uint64_t> ns_probe(threads, 0);
    std::vector<uint64_t> ns_eviction(threads, 0);
    std::vector<size_t> worker_evictions(threads, 0);
    std::vector<size_t> worker_flushes(threads, 0);

    std::vector<ProbeMaterialiser> mats(threads);
    std::atomic<size_t> global_peak_probe_bytes{0};

    /// -------- PER-THREAD BUFFER + FLUSH-AND-PROBE LOOP --------
    parallelRun(
        threads,
        [&](size_t tid)
        {
            /// Buffered probe input (deep copies). One block per input
            /// block consumed since the last flush. The outer vector is
            /// pmr-rooted at the tracker so the block buffers themselves
            /// (deep copies) are tracked; metadata vectors (hashes,
            /// partitioning) use the default resource because they're
            /// scratch and we don't need them in peak_mem.
            std::pmr::vector<Block> buffered(&tracker);
            std::vector<std::vector<uint64_t>> buf_hashes;
            std::vector<BlockPartitioning> per_block_part;
            size_t buffered_row_bytes = 0;

            /// Scratch reused across flushes.
            std::pmr::vector<uint64_t> sub_hashes(&tracker);
            std::pmr::vector<uint64_t> sub_keys(&tracker);
            std::pmr::vector<RowRefCell> heads(&tracker);
            std::pmr::vector<size_t> probe_idx(&tracker);
            std::pmr::vector<RowRefCell> build_ref(&tracker);

            ProbeMaterialiser & mat = mats[tid];
            mat.init(build.schema, probe.schema, result.output.workers[tid], PIPELINE_BLOCK_ROWS);

            uint64_t my_probe_shuffle_ns = 0;
            uint64_t my_probe_ns = 0;
            uint64_t my_eviction_ns = 0;
            size_t my_flushes = 0;
            size_t my_evictions = 0;

            const size_t n_probe_blocks = probe.blocks.size();
            const size_t probe_start = (n_probe_blocks * tid) / threads;
            const size_t probe_end = (n_probe_blocks * (tid + 1)) / threads;

            /// Partition one buffered block via counting sort on its
            /// precomputed hashes. Charged to `my_probe_shuffle_ns`
            /// since it's the (no-data-movement) equivalent of the
            /// radix scatter.
            auto partitionOne = [&](size_t b)
            {
                const size_t rows = buffered[b].rows;
                BlockPartitioning & bp = per_block_part[b];
                bp.sorted_idx.assign(rows, 0);
                bp.p_offsets.assign(total_leaves + 1, 0);

                /// Count
                const uint64_t * hp = buf_hashes[b].data();
                for (size_t i = 0; i < rows; ++i)
                {
                    const size_t L = static_cast<size_t>((hp[i] >> leaf_shift) & leaf_mask);
                    ++bp.p_offsets[L];
                }
                /// Exclusive prefix sum into p_offsets (so p_offsets[L]
                /// is the start of partition L); the trailing slot
                /// p_offsets[total_leaves] becomes the row total.
                size_t acc = 0;
                for (size_t L = 0; L <= total_leaves; ++L)
                {
                    const size_t cnt = bp.p_offsets[L];
                    bp.p_offsets[L] = static_cast<uint32_t>(acc);
                    acc += cnt;
                }
                /// Fill via a cursor that mutates the offsets and is
                /// restored implicitly by re-walking the offsets from
                /// the prefix sum. We keep a local cursor copy to
                /// avoid the second pass.
                std::vector<uint32_t> cursor(total_leaves);
                std::memcpy(cursor.data(), bp.p_offsets.data(), total_leaves * sizeof(uint32_t));
                for (size_t i = 0; i < rows; ++i)
                {
                    const size_t L = static_cast<size_t>((hp[i] >> leaf_shift) & leaf_mask);
                    bp.sorted_idx[cursor[L]++] = static_cast<uint32_t>(i);
                }
            };

            /// Drain the buffer pool: partition every buffered block,
            /// then probe leaf-by-leaf across all buffered blocks.
            auto flushAndProbe = [&]()
            {
                if (buffered.empty())
                    return;
                ++my_flushes;

                /// Phase A: incremental hash + partition the new
                /// blocks (everything not yet partitioned).
                const TimePoint tps0 = now();
                for (size_t b = 0; b < buffered.size(); ++b)
                {
                    if (buf_hashes[b].size() != buffered[b].rows)
                    {
                        buf_hashes[b].resize(buffered[b].rows);
                        intHash64Batch(buffered[b].keys.data(), buffered[b].rows, buf_hashes[b].data());
                    }
                    if (per_block_part[b].p_offsets.size() != total_leaves + 1)
                        partitionOne(b);
                }
                my_probe_shuffle_ns += toNanos(now() - tps0);

                /// Phase B: probe leaf-by-leaf across all blocks.
                /// Each (leaf, block) cell is a contiguous run inside
                /// `sorted_idx[block]`. The HT for leaf L stays hot
                /// across the inner loop over blocks.
                const TimePoint tp0 = now();
                for (size_t L = 0; L < total_leaves; ++L)
                {
                    const JoinHashTable & ht = leaf_hts[L];
                    const BlockStore & store = *leaf_stores[L];
                    for (size_t b = 0; b < buffered.size(); ++b)
                    {
                        const BlockPartitioning & bp = per_block_part[b];
                        const size_t start = bp.p_offsets[L];
                        const size_t end = bp.p_offsets[L + 1];
                        if (start == end)
                            continue;
                        probeSubsetOfBuffered(
                            buffered[b].view(),
                            bp.sorted_idx.data() + start,
                            end - start,
                            buf_hashes[b].data(),
                            ht,
                            store,
                            mat,
                            sub_hashes,
                            sub_keys,
                            heads,
                            probe_idx,
                            build_ref);
                    }
                }
                my_probe_ns += toNanos(now() - tp0);

                /// Phase C: drop the buffer pool.
                const TimePoint td0 = now();
                buffered.clear();
                buf_hashes.clear();
                per_block_part.clear();
                buffered_row_bytes = 0;
                ++my_evictions;
                my_eviction_ns += toNanos(now() - td0);
            };

            /// -------- MAIN LOOP --------
            for (size_t b = probe_start; b < probe_end; ++b)
            {
                const Block & src = probe.blocks[b];
                if (src.rows == 0)
                    continue;

                /// Deep copy the input block into the worker-local
                /// buffer pool. This is the "deep copy" of Idea 2:
                /// the buffered set is self-contained and stays in
                /// the thread's working set across the flush.
                const TimePoint tps0 = now();
                buffered.emplace_back();
                Block & dst = buffered.back();
                dst.rows = src.rows;
                dst.keys.assign(src.keys.begin(), src.keys.end());
                dst.payloads.resize(src.payloads.size());
                for (size_t c = 0; c < src.payloads.size(); ++c)
                {
                    dst.payloads[c].type = src.payloads[c].type;
                    dst.payloads[c].data.assign(src.payloads[c].data.begin(), src.payloads[c].data.end());
                }
                /// One per-block hash slot + one per-block partition
                /// metadata slot; lazily filled at flush time.
                buf_hashes.emplace_back();
                per_block_part.emplace_back();
                buffered_row_bytes += src.rows * bytes_per_row;
                my_probe_shuffle_ns += toNanos(now() - tps0);

                /// Lock-free update of the GLOBAL peak (the per-worker
                /// budget plus the aggregate is approximate; here we
                /// report this worker's buffered footprint × threads
                /// as a coarse global peak that is directly comparable
                /// to `bep_budget_mib`).
                size_t old_peak = global_peak_probe_bytes.load(std::memory_order_relaxed);
                const size_t cur = buffered_row_bytes * threads;
                while (cur > old_peak && !global_peak_probe_bytes.compare_exchange_weak(old_peak, cur, std::memory_order_relaxed))
                {
                }

                /// Flush when the worker's buffered row-bytes meet or
                /// exceed its per-worker budget share.
                if (buffered_row_bytes >= per_worker_budget)
                    flushAndProbe();
            }

            /// -------- END-OF-INPUT FLUSH --------
            if (!buffered.empty())
                flushAndProbe();

            mat.finish();

            ns_probe_shuffle[tid] = my_probe_shuffle_ns;
            ns_probe[tid] = my_probe_ns;
            ns_eviction[tid] = my_eviction_ns;
            worker_flushes[tid] = my_flushes;
            worker_evictions[tid] = my_evictions;
        });

    const TimePoint t_e2e1 = now();

    /// -------- AGGREGATION --------
    auto maxNs = [](const std::vector<uint64_t> & v) -> uint64_t { return v.empty() ? 0 : *std::max_element(v.begin(), v.end()); };
    auto sumNs = [](const std::vector<uint64_t> & v) -> uint64_t
    {
        uint64_t s = 0;
        for (auto x : v)
            s += x;
        return s;
    };

    const uint64_t max_build = maxNs(ns_build);
    const uint64_t max_probe_shuffle = maxNs(ns_probe_shuffle);
    const uint64_t max_probe = maxNs(ns_probe);
    const uint64_t max_eviction = maxNs(ns_eviction);

    const uint64_t sum_build = sumNs(ns_build);
    const uint64_t sum_probe_shuffle = sumNs(ns_probe_shuffle);
    const uint64_t sum_probe = sumNs(ns_probe);
    const uint64_t sum_eviction = sumNs(ns_eviction);

    const double br = static_cast<double>(build.total_rows);
    const double pr = static_cast<double>(probe.total_rows);

    result.build_shuffle.wall_ms = nanosToMillis(build_shuffle_ns);
    result.build_shuffle.ns_per_row
        = build.total_rows == 0 ? 0.0 : static_cast<double>(build_shuffle_ns) * static_cast<double>(threads) / br;

    result.build.wall_ms = nanosToMillis(max_build);
    result.build.ns_per_row = build.total_rows == 0 ? 0.0 : static_cast<double>(sum_build) / br;

    result.probe_shuffle.wall_ms = nanosToMillis(max_probe_shuffle);
    result.probe_shuffle.ns_per_row = probe.total_rows == 0 ? 0.0 : static_cast<double>(sum_probe_shuffle) / pr;

    result.probe.wall_ms = nanosToMillis(max_probe);
    result.probe.ns_per_row = probe.total_rows == 0 ? 0.0 : static_cast<double>(sum_probe) / pr;

    result.eviction_overhead.wall_ms = nanosToMillis(max_eviction);
    result.eviction_overhead.ns_per_row = probe.total_rows == 0 ? 0.0 : static_cast<double>(sum_eviction) / pr;

    result.e2e_wall_ms = toMillis(t_e2e1 - t_e2e0);

    result.bep_evictions = 0;
    result.bep_refinements = 0;
    result.bep_build_skip_retries = 0;
    for (size_t t = 0; t < threads; ++t)
    {
        result.bep_evictions += worker_evictions[t];
        /// Idea 2 has no radix refinements; report flush count in
        /// `bep_refinements` so the CSV/console column reflects the
        /// algorithm's principal repeating unit.
        result.bep_refinements += worker_flushes[t];
    }
    result.bep_peak_bytes = global_peak_probe_bytes.load(std::memory_order_relaxed);

    result.peak_mem_bytes = tracker.peakBytes();

    return result;
}

}
