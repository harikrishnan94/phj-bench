#include "PHJBep.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <cstdio>
#include <cstdlib>

#include "BlockStore.h"
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


/// Probe a contiguous sequence of OutBlocks against `ht` / `store`.
template <class HT>
[[gnu::always_inline]] inline void probeChain(
    const std::pmr::vector<OutBlock> & blocks,
    const BlockStore & store,
    const HT & ht,
    ProbeMaterialiser & mat,
    std::pmr::vector<uint64_t> & hashes,
    std::pmr::vector<RowRefCell> & heads,
    std::pmr::vector<size_t> & probe_idx,
    std::pmr::vector<RowRefCell> & build_ref)
{
    for (const auto & blk : blocks)
    {
        if (blk.rows == 0)
            continue;
        probeOneBlock(blk.view(), store, ht, mat, hashes, heads, probe_idx, build_ref);
    }
}

}


/// PHJ-BEP — shared, lightest-weight design.
///
/// Three structural choices, each backed by measured overhead from the
/// previous design iteration:
///
///   1. EAGER LEAF HT BUILD. All HTs are constructed up-front via a
///      work-stealing parallelRun across leaves (same pattern as
///      PHJ-PURE's build phase). HTs are read-only for the rest of
///      the run. Eliminates the entire `LeafState`
///      NOT_BUILT/BUILDING/BUILT/STEALING CAS machine that gated
///      every eviction in the original BEP (~22K CAS operations on
///      the user's reference workload), and removes the lazy-build
///      time from the critical path.
///
///   2. SHARED LEAF CHAINS, PER-WORKER UNREFINED. Each worker keeps
///      its OWN pass-1 (`unrefined`) chains (scatter has no
///      contention) but publishes refined OutBlocks into a SINGLE
///      shared `leaves[L]` chain per leaf, protected by one
///      `leaf_mutexes[L]` per leaf. The "shared budget" view of the
///      world is then exact: `global_leaf_rows` is a single atomic
///      summing rows across all leaves, decremented on eviction.
///
///   3. ROUND-ROBIN EVICTION VICTIM. The original BEP scanned
///      `argmax_L (leaf_rows[L])` over all 1024 leaves on every
///      eviction (~10us each, 22K times = ~220ms of pure scan).
///      With uniform key distribution the argmax has small variance
///      so any leaf is roughly as full as any other; we drop the
///      scan and select victims by `next_evict_leaf.fetch_add(1) %
///      total_leaves`. A failed `try_lock` (another worker beat us
///      to this leaf) costs ~10ns and we advance to the next leaf.
///
/// Combined with the row-based budget accounting from main (a9e551d),
/// the per-eviction overhead reduces from ~70us (baseline) to ~few
/// hundred ns of bookkeeping plus the irreducible mutex + chain-move
/// cost. The probe phase itself is unchanged.
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

    /// -------- BUILD SHUFFLE (full leaf depth) --------
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
    const size_t pass1_bits = cfg.pass_bits.front();
    const size_t P = size_t{1} << pass1_bits;
    const size_t leaves_per_p1 = total_leaves / P;
    const uint32_t pass1_shift = static_cast<uint32_t>(64u - pass1_bits);
    const size_t bytes_per_row = sizeof(uint64_t) + probe.schema.rowByteSize();

    const size_t budget_bytes = bep_budget_mib * size_t{1024} * size_t{1024};

    /// Per-worker unrefined budget (worker decides its own refinement
    /// triggers without coordination). Shared global leaf budget.
    const size_t per_worker_budget = budget_bytes / threads;
    const size_t unrefined_high_water = per_worker_budget / 4;
    const size_t unrefined_low_water = unrefined_high_water / 2;
    const size_t global_leaf_high_water = (budget_bytes * 3) / 4;
    const size_t global_leaf_low_water = (global_leaf_high_water * 5) / 8;

    /// -------- EAGER LEAF HT BUILD (work-stealing) --------
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

    /// -------- SHARED LEAF CHAINS + SINGLE EVICTION VICTIM CURSOR --------
    std::pmr::vector<PartitionOut> leaves(&tracker);
    leaves.resize(total_leaves);
    for (auto & po : leaves)
        initPartitionOut(po, probe.schema);
    std::vector<std::mutex> leaf_mutexes(total_leaves);

    /// `leaf_rows[L]` is the count of probe rows currently buffered in
    /// `leaves[L]`. Updated under `leaf_mutexes[L]` so it's exact wrt
    /// the chain. Never read outside the mutex; argmax-style scans are
    /// not needed since eviction uses a round-robin cursor.
    std::vector<size_t> leaf_rows(total_leaves, 0);

    /// Sum of `leaf_rows[L]` across all leaves. Atomic so workers can
    /// poll cheaply to decide when the global leaf trigger fires.
    /// Incremented on refine-publish (delta = added rows), decremented
    /// on evict-steal (delta = stolen rows).
    std::atomic<size_t> global_leaf_rows{0};

    /// Round-robin eviction cursor. Workers `fetch_add(1) %
    /// total_leaves` to pick a candidate victim. With uniform key
    /// distribution all leaves are roughly equal-sized so any one is
    /// as good a victim as any other; this replaces the
    /// argmax-over-1024-leaves scan that dominated `eviction_overhead`
    /// in the baseline.
    std::atomic<size_t> next_evict_leaf{0};

    /// Global peak tracker. Only `global_leaf_rows × bytes_per_row` is
    /// captured (the worker-local unrefined buffers are not summed —
    /// they are bounded by `unrefined_high_water × threads` and so
    /// add a constant ceiling rather than a contended atomic).
    std::atomic<size_t> global_peak_probe_bytes{0};

    /// Worker-liveness counter for cooperative end-of-slice draining.
    /// Each worker decrements this when its input slice is exhausted;
    /// any worker that has finished its own slice keeps draining
    /// global leaves as long as ANY other worker is still scattering.
    /// Without this the slowest scatterer alone absorbs every late
    /// eviction while the fast scatterers idle at the parallelRun
    /// barrier — measured in this iteration as ~2400ms of imbalance
    /// (slowest worker eviction 3906ms vs others idle).
    std::atomic<size_t> scatter_workers_active{threads};

    /// -------- PER-THREAD PROBE TIMING + COUNTERS --------
    std::vector<uint64_t> ns_probe_shuffle(threads, 0);
    std::vector<uint64_t> ns_probe(threads, 0);
    std::vector<uint64_t> ns_eviction(threads, 0);
    std::vector<size_t> worker_evictions(threads, 0);
    std::vector<size_t> worker_refinements(threads, 0);
    std::vector<size_t> worker_skip_retries(threads, 0);

    /// Diagnostic sub-buckets (printed once after the run). Kept as
    /// separate accumulators so we can attribute eviction_overhead
    /// to its actual sources.
    std::vector<uint64_t> ns_main_evict(threads, 0); // mid-stream eviction loop wall
    std::vector<uint64_t> ns_coop_drain(threads, 0); // end-of-slice cooperative drain
    std::vector<uint64_t> ns_finish_refine(threads, 0); // end-of-slice residual refine
    std::vector<uint64_t> ns_worker_total(threads, 0); // per-worker parallelRun #1 wall
    std::vector<uint64_t> ns_scatter(threads, 0); // pure scatter time (excl. refine)
    std::vector<uint64_t> ns_refine(threads, 0); // refinement time
    std::vector<uint64_t> ns_refine_lock(threads, 0); // mutex-held publish-into-leaves time

    std::vector<ProbeMaterialiser> mats(threads);

    /// -------- PROBE LOOP --------
    parallelRun(
        threads,
        [&](size_t tid)
        {
            /// Per-worker pass-1 chains (private, no contention).
            std::pmr::vector<PartitionOut> unrefined(&tracker);
            unrefined.resize(P);
            std::pmr::vector<size_t> unrefined_rows(&tracker);
            unrefined_rows.resize(P, 0);
            std::pmr::vector<PartitionOut> intermediate(&tracker);
            intermediate.resize(leaves_per_p1);

            for (auto & po : unrefined)
                initPartitionOut(po, probe.schema);
            for (auto & po : intermediate)
                initPartitionOut(po, probe.schema);

            size_t unrefined_row_bytes = 0;

            ScatterScratch scatter_scratch(&tracker);
            std::pmr::vector<uint64_t> probe_hashes(&tracker);
            std::pmr::vector<RowRefCell> heads(&tracker);
            std::pmr::vector<size_t> probe_idx(&tracker);
            std::pmr::vector<RowRefCell> build_ref(&tracker);

            ProbeMaterialiser & mat = mats[tid];
            mat.init(build.schema, probe.schema, result.output.workers[tid], PIPELINE_BLOCK_ROWS);

            uint64_t my_probe_shuffle_ns = 0;
            uint64_t my_probe_ns = 0;
            uint64_t my_eviction_ns = 0;
            uint64_t my_main_evict_ns = 0;
            uint64_t my_coop_drain_ns = 0;
            uint64_t my_finish_refine_ns = 0;
            uint64_t my_scatter_ns = 0;
            uint64_t my_refine_ns = 0;
            uint64_t my_refine_lock_ns = 0;
            size_t my_evictions = 0;
            size_t my_refinements = 0;
            size_t my_skip_retries = 0;
            const TimePoint t_worker0 = now();

            const size_t n_probe_blocks = probe.blocks.size();
            const size_t probe_start = (n_probe_blocks * tid) / threads;
            const size_t probe_end = (n_probe_blocks * (tid + 1)) / threads;

            const std::vector<size_t> probe_identity = [&]
            {
                std::vector<size_t> v(probe.schema.types.size());
                for (size_t c = 0; c < v.size(); ++c)
                    v[c] = c;
                return v;
            }();

            /// Pick this worker's largest own pass-1 chain. O(P) but P
            /// is small (typically 64–2048) and the scan touches only
            /// thread-local memory, so this is sub-microsecond even at
            /// P=2048.
            auto argmaxOwnUnrefined = [&]() noexcept -> size_t
            {
                size_t best_p1 = SIZE_MAX;
                size_t best_rows = 0;
                for (size_t p1 = 0; p1 < P; ++p1)
                {
                    const size_t r = unrefined_rows[p1];
                    if (r > best_rows)
                    {
                        best_rows = r;
                        best_p1 = p1;
                    }
                }
                return best_p1;
            };

            /// Refine one own pass-1 chain into the SHARED leaf chains.
            /// Publishes each non-empty intermediate slot under
            /// `leaf_mutexes[L]` with a single atomic add to
            /// `global_leaf_rows` for the whole partition (one atomic
            /// per refine, not one per leaf).
            auto refinePartition = [&](size_t p1)
            {
                for (auto & po : intermediate)
                    initPartitionOut(po, probe.schema);

                refineToLeaves(std::move(unrefined[p1]), probe.schema, cfg.pass_bits, intermediate.data(), scatter_scratch, &tracker);

                initPartitionOut(unrefined[p1], probe.schema);
                const size_t drained_rows = unrefined_rows[p1];
                unrefined_rows[p1] = 0;
                unrefined_row_bytes -= drained_rows * bytes_per_row;

                const size_t base = p1 * leaves_per_p1;
                size_t added_rows_total = 0;
                for (size_t i = 0; i < leaves_per_p1; ++i)
                {
                    PartitionOut & src = intermediate[i];
                    if (src.blocks.empty())
                        continue;
                    size_t added_rows = 0;
                    for (const auto & blk : src.blocks)
                        added_rows += blk.rows;
                    if (added_rows == 0)
                        continue;
                    const size_t L = base + i;
                    const TimePoint tlk0 = now();
                    {
                        std::lock_guard<std::mutex> lock(leaf_mutexes[L]);
                        for (auto & blk : src.blocks)
                        {
                            if (blk.rows > 0)
                                leaves[L].blocks.push_back(std::move(blk));
                        }
                        leaf_rows[L] += added_rows;
                    }
                    my_refine_lock_ns += toNanos(now() - tlk0);
                    src.blocks.clear();
                    src.cur = nullptr;
                    added_rows_total += added_rows;
                }
                if (added_rows_total != 0)
                    global_leaf_rows.fetch_add(added_rows_total, std::memory_order_relaxed);
            };

            /// Attempt to evict one leaf via the round-robin cursor.
            /// Returns the number of rows evicted (0 = skipped or
            /// empty). `my_eviction_ns` records ONLY overhead — the
            /// setup, mutex, atomic, and drop — not the probe time
            /// (added to `my_probe_ns`) and not the outer loop's wait
            /// time (tracked separately as `my_main_evict_ns` /
            /// `my_coop_drain_ns` diagnostics).
            auto tryEvictOneLeaf = [&]() -> size_t
            {
                const TimePoint t_setup0 = now();
                const size_t L = next_evict_leaf.fetch_add(1, std::memory_order_relaxed) % total_leaves;

                std::unique_lock<std::mutex> lock(leaf_mutexes[L], std::try_to_lock);
                if (!lock.owns_lock())
                {
                    ++my_skip_retries;
                    my_eviction_ns += toNanos(now() - t_setup0);
                    return 0;
                }
                const size_t rows = leaf_rows[L];
                if (rows == 0)
                {
                    my_eviction_ns += toNanos(now() - t_setup0);
                    return 0;
                }

                PartitionOut stolen(&tracker);
                stolen.blocks = std::move(leaves[L].blocks);
                leaves[L].blocks.clear();
                leaves[L].cur = nullptr;
                leaf_rows[L] = 0;
                lock.unlock();

                global_leaf_rows.fetch_sub(rows, std::memory_order_relaxed);
                my_eviction_ns += toNanos(now() - t_setup0);

                const TimePoint tp0 = now();
                probeChain(stolen.blocks, *leaf_stores[L], leaf_hts[L], mat, probe_hashes, heads, probe_idx, build_ref);
                my_probe_ns += toNanos(now() - tp0);

                const TimePoint td0 = now();
                dropPartition(stolen);
                my_eviction_ns += toNanos(now() - td0);
                ++my_evictions;
                return rows;
            };

            /// Post-block trigger check.
            auto evictAsNeeded = [&]()
            {
                /// Phase 1: drain own unrefined to low water. Worker-
                /// local; no shared state touched.
                if (unrefined_row_bytes >= unrefined_high_water)
                {
                    while (unrefined_row_bytes >= unrefined_low_water)
                    {
                        const size_t p1 = argmaxOwnUnrefined();
                        if (p1 == SIZE_MAX)
                            break;
                        const TimePoint trf0 = now();
                        refinePartition(p1);
                        const uint64_t r_ns = toNanos(now() - trf0);
                        my_probe_shuffle_ns += r_ns;
                        my_refine_ns += r_ns;
                        ++my_refinements;
                    }
                }

                /// Phase 2: drain shared leaves to low water. Re-read
                /// the global counter on every successful eviction so
                /// we observe other workers' drainage and don't
                /// over-evict. Failed try_lock (skip) is cheap; we
                /// only refresh the counter on success or after a
                /// no-progress streak.
                size_t cached_leaf_bytes
                    = global_leaf_rows.load(std::memory_order_relaxed) * bytes_per_row;
                if (cached_leaf_bytes >= global_leaf_high_water)
                {
                    size_t no_progress = 0;
                    size_t evicts_since_reread = 0;
                    const TimePoint tev0 = now();
                    while (cached_leaf_bytes >= global_leaf_low_water)
                    {
                        const size_t rows = tryEvictOneLeaf();
                        if (rows == 0)
                        {
                            if (++no_progress >= 64)
                            {
                                cached_leaf_bytes
                                    = global_leaf_rows.load(std::memory_order_relaxed) * bytes_per_row;
                                no_progress = 0;
                            }
                            continue;
                        }
                        no_progress = 0;
                        /// Locally decrement on success; periodically
                        /// re-read the global to pick up other workers'
                        /// drainage and avoid over-eviction. Re-read
                        /// every 4 successful evictions keeps the
                        /// contended-atomic load count at ~5K/worker
                        /// (vs ~22K with per-evict reads) while
                        /// bounding over-eviction at 4 × per-leaf rows
                        /// (~50K rows, ~3 MiB).
                        const size_t dec = rows * bytes_per_row;
                        cached_leaf_bytes = cached_leaf_bytes > dec ? cached_leaf_bytes - dec : 0;
                        if (++evicts_since_reread >= 4)
                        {
                            cached_leaf_bytes
                                = global_leaf_rows.load(std::memory_order_relaxed) * bytes_per_row;
                            evicts_since_reread = 0;
                        }
                    }
                    my_main_evict_ns += toNanos(now() - tev0);
                }
            };

            /// -------- MAIN LOOP --------
            for (size_t b = probe_start; b < probe_end; ++b)
            {
                const Block & blk = probe.blocks[b];
                if (blk.rows == 0)
                    continue;

                const TimePoint tps0 = now();
                scatterBatch(blk.view(), probe_identity, probe.schema, pass1_shift, P, unrefined.data(), scatter_scratch);
                size_t added_rows = 0;
                for (size_t p = 0; p < P; ++p)
                {
                    const size_t delta = scatter_scratch.local_hist[p];
                    if (delta == 0)
                        continue;
                    unrefined_rows[p] += delta;
                    added_rows += delta;
                }
                unrefined_row_bytes += added_rows * bytes_per_row;

                /// Global peak — leaf-bytes only (see comment on
                /// `global_peak_probe_bytes`). One relaxed atomic load
                /// + one CAS attempt when higher.
                const size_t cur = global_leaf_rows.load(std::memory_order_relaxed) * bytes_per_row;
                size_t old_peak = global_peak_probe_bytes.load(std::memory_order_relaxed);
                while (cur > old_peak && !global_peak_probe_bytes.compare_exchange_weak(old_peak, cur, std::memory_order_relaxed))
                {
                }
                const uint64_t scatter_ns = toNanos(now() - tps0);
                my_probe_shuffle_ns += scatter_ns;
                my_scatter_ns += scatter_ns;

                evictAsNeeded();
            }

            /// -------- COOPERATIVE END-OF-SLICE DRAIN --------
            /// This worker has finished its input slice. Before
            /// refining its own residual unrefined (which would
            /// publish more rows into shared leaves and re-trigger
            /// other workers' triggers), help drain the shared leaves
            /// as long as any other worker is still scattering. This
            /// converts barrier idle time into useful eviction work
            /// and balances cross-worker probe load. Without it the
            /// slow scatterer alone absorbs all late evictions while
            /// fast scatterers idle (measured: 2.4s of imbalance in
            /// the first iteration of this design).
            scatter_workers_active.fetch_sub(1, std::memory_order_acq_rel);
            {
                const TimePoint tev0 = now();
                while (scatter_workers_active.load(std::memory_order_acquire) > 0)
                {
                    const size_t cur_bytes
                        = global_leaf_rows.load(std::memory_order_relaxed) * bytes_per_row;
                    if (cur_bytes >= global_leaf_low_water)
                    {
                        if (tryEvictOneLeaf() != 0)
                            continue;
                    }
                    std::this_thread::yield();
                }
                /// Diagnostic only: wall time of the cooperative drain
                /// (mostly yield/idle waiting for slow scatterer; the
                /// successful evictions inside it already added their
                /// own overhead to `my_eviction_ns`).
                my_coop_drain_ns += toNanos(now() - tev0);
            }

            /// -------- END-OF-SLICE REFINEMENT --------
            const TimePoint tfin0 = now();

            /// Refine residual own unrefined into shared leaves so the
            /// post-probe drain can find them.
            for (size_t p1 = 0; p1 < P; ++p1)
            {
                if (unrefined_rows[p1] == 0)
                    continue;
                const TimePoint trf0 = now();
                refinePartition(p1);
                const uint64_t r_ns = toNanos(now() - trf0);
                my_probe_shuffle_ns += r_ns;
                my_refine_ns += r_ns;
                ++my_refinements;
            }
            my_finish_refine_ns = toNanos(now() - tfin0);

            ns_probe_shuffle[tid] = my_probe_shuffle_ns;
            ns_probe[tid] = my_probe_ns;
            ns_eviction[tid] = my_eviction_ns;
            worker_evictions[tid] = my_evictions;
            worker_refinements[tid] = my_refinements;
            worker_skip_retries[tid] = my_skip_retries;
            ns_main_evict[tid] = my_main_evict_ns;
            ns_coop_drain[tid] = my_coop_drain_ns;
            ns_finish_refine[tid] = my_finish_refine_ns;
            ns_scatter[tid] = my_scatter_ns;
            ns_refine[tid] = my_refine_ns;
            ns_refine_lock[tid] = my_refine_lock_ns;
            ns_worker_total[tid] = toNanos(now() - t_worker0);
        });

    /// -------- POST-PROBE DRAIN (work-stealing across leaves) --------
    /// parallelRun #1 has joined, so `leaves[L]` and `leaf_rows[L]` are
    /// only modifiable by the drain workers; no lock needed.
    std::atomic<size_t> next_drain_leaf{0};
    parallelRun(
        threads,
        [&](size_t tid)
        {
            std::pmr::vector<uint64_t> probe_hashes(&tracker);
            std::pmr::vector<RowRefCell> heads(&tracker);
            std::pmr::vector<size_t> probe_idx(&tracker);
            std::pmr::vector<RowRefCell> build_ref(&tracker);

            ProbeMaterialiser & mat = mats[tid];

            uint64_t my_probe_ns = 0;
            uint64_t my_eviction_ns = 0;
            size_t my_evictions = 0;

            while (true)
            {
                const size_t L = next_drain_leaf.fetch_add(1, std::memory_order_relaxed);
                if (L >= total_leaves)
                    break;
                if (leaf_rows[L] == 0)
                    continue;

                const TimePoint tp0 = now();
                probeChain(leaves[L].blocks, *leaf_stores[L], leaf_hts[L], mat, probe_hashes, heads, probe_idx, build_ref);
                my_probe_ns += toNanos(now() - tp0);

                const TimePoint td0 = now();
                dropPartition(leaves[L]);
                leaf_rows[L] = 0;
                my_eviction_ns += toNanos(now() - td0);
                ++my_evictions;
            }
            mat.finish();

            ns_probe[tid] += my_probe_ns;
            ns_eviction[tid] += my_eviction_ns;
            worker_evictions[tid] += my_evictions;
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
        result.bep_refinements += worker_refinements[t];
        result.bep_build_skip_retries += worker_skip_retries[t];
    }
    result.bep_peak_bytes = global_peak_probe_bytes.load(std::memory_order_relaxed);

    result.peak_mem_bytes = tracker.peakBytes();

    if (const char * dbg = std::getenv("BEP_DEBUG"); dbg != nullptr && dbg[0] == '1')
    {
        std::fprintf(
            stderr,
            "\n[bep-debug] per-worker (ms): total / scatter / refine (lock) / main_evict / coop_drain / finish\n");
        for (size_t t = 0; t < threads; ++t)
            std::fprintf(
                stderr,
                "  t%02zu: total=%7.1f scatter=%6.1f refine=%6.1f (lock=%5.1f) main_evict=%6.1f coop_drain=%6.1f finish=%5.1f\n",
                t,
                static_cast<double>(ns_worker_total[t]) * 1e-6,
                static_cast<double>(ns_scatter[t]) * 1e-6,
                static_cast<double>(ns_refine[t]) * 1e-6,
                static_cast<double>(ns_refine_lock[t]) * 1e-6,
                static_cast<double>(ns_main_evict[t]) * 1e-6,
                static_cast<double>(ns_coop_drain[t]) * 1e-6,
                static_cast<double>(ns_finish_refine[t]) * 1e-6);
    }

    return result;
}

}
