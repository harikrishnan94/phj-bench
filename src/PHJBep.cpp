#include "PHJBep.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

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

/// Per-leaf HT lifecycle. Transitions form a small state machine
/// claimed via CAS:
///
///   NOT_BUILT  --CAS-->  BUILDING  --store-->  STEALING
///   BUILT      --CAS-->  STEALING  --store-->  BUILT
///   (steal yielded 0 rows)         --store-->  BUILT
///
/// `BUILDING` and `STEALING` are exclusive-ownership claims that
/// gate the per-leaf work (HT construction and chain steal+probe+
/// drop, respectively). `argmaxGlobalLeaf` filters out both states
/// so that 48 workers don't converge on the same leaf — eliminates
/// the per-leaf mutex contention previously caused by all workers
/// racing to steal the same `BUILT` leaf simultaneously. Without
/// this, ~85-91% of `stealLeafChain` calls returned zero rows after
/// waiting on the mutex, because the first stealer drained the
/// chain and the rest woke up to an empty leaf.
///
/// Acquire/release ordering establishes happens-before from the
/// constructor's writes inside `BUILDING` to any subsequent reader
/// that observes `BUILT` or `STEALING`.
enum class LeafState : uint8_t
{
    NOT_BUILT = 0,
    BUILDING = 1,
    BUILT = 2,
    STEALING = 3,
};


/// Per-worker buffer state. After the F-style refactor each worker
/// keeps only its own pass-1 (`unrefined`) chains; refined leaf rows
/// are published into the SHARED `published_leaves[L]` chains under
/// per-leaf mutexes. Refinement first scatters the consumed
/// `unrefined[p1]` into a worker-local `intermediate` array (so
/// `scatterBatch`'s interleaved per-partition writes happen entirely
/// off-shared-memory) and then briefly locks each destination leaf
/// to move blocks into the shared chain.
///
/// The worker caches its own `unrefined` buffered-row-bytes counter
/// (`unrefined_row_bytes` = `sum_blocks(blk.rows) * bytes_per_row`)
/// so the per-block budget check doesn't have to re-walk all pass-1
/// chains. A separate shared `global_unrefined_row_bytes` atomic
/// mirrors the sum across workers for the global peak-tracking path;
/// the worker updates it with the delta whenever its own count
/// changes (`refreshUnrefinedRowBytes`).
///
/// IMPORTANT: the counter sums `blk.rows`, NOT `blk.capacity`. The
/// budget bounds buffered probe DATA, not allocated buffer headroom.
/// Counting capacity made the trigger fire immediately on the first
/// scatter for partition counts × bytes_per_row that pushed even
/// empty initial OutBlocks over the high-water threshold (the radix
/// `initialOutBlockRows(schema)` policy allocates ~4 KiB per chain
/// per partition the moment any row lands in it; for 1024 partitions
/// and a 68 B probe row that's 17 MiB before any data is buffered).
struct WorkerProbeState
{
    std::pmr::vector<PartitionOut> unrefined;
    std::pmr::vector<size_t> unrefined_rows;

    /// Reusable intermediate destination for refinement. Sized to
    /// `leaves_per_p1`. Cleared and re-initialised at the start of
    /// each `refinePartition` invocation so its blocks vector is
    /// empty (the previous refinement's blocks were moved into the
    /// shared chain).
    std::pmr::vector<PartitionOut> intermediate;

    size_t unrefined_row_bytes = 0;
    size_t bytes_per_row = 0;

    explicit WorkerProbeState(std::pmr::memory_resource * mr)
        : unrefined(mr)
        , unrefined_rows(mr)
        , intermediate(mr)
    {
    }
};


struct EvictTarget
{
    /// 0 = unrefined pass-1 partition (per-worker), 1 = published leaf (shared).
    int kind = 0;
    size_t idx = 0;
    size_t rows = 0;
};


/// Release the trailing unused capacity of an `OutBlock` by re-
/// allocating its key and payload buffers at exactly `rows` slots
/// each. Used after the build shuffle: the radix scatter's doubling-
/// grow scheme typically over-allocates the last block per (thread,
/// partition) chain — for a 100M-row build with 2048 partitions and
/// 16 threads that's ~1.4 GiB of trailing capacity that would
/// otherwise propagate verbatim into the per-leaf BlockStores
/// (`outBlockToBlock` does a move, not a shrink). Compacting once
/// here, immediately after radixShuffle, costs one pass of memcpy
/// over the build data (parallelisable per chain) and is invisible
/// to the eviction / probe loops downstream.
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


/// Sum `rows × bytes_per_row` across all buffered unrefined OutBlocks
/// for one worker. This is the "buffered probe data" measure used by
/// the budget; allocated capacity (which can exceed rows by up to
/// ~2× under the doubling-grow scheme) is intentionally NOT counted —
/// see the comment on `WorkerProbeState::unrefined_row_bytes`.
[[gnu::always_inline]] inline size_t unrefinedRowBytesFromScan(const WorkerProbeState & s) noexcept
{
    size_t rows = 0;
    for (const auto & po : s.unrefined)
        for (const auto & blk : po.blocks)
            rows += blk.rows;
    return rows * s.bytes_per_row;
}


/// Probe a contiguous sequence of OutBlocks against `ht` / `store`.
/// Used both by mid-stream eviction (after stealing a leaf's chain
/// out of the shared structure) and by the end-of-input drain. The
/// per-block `probeOneBlock` is called per OutBlock; scratch vectors
/// are reused across iterations so any growth amortises across the
/// entire chain. This is the (C) wrapper: chain-level entry point
/// that keeps the call boundary stable while leaving per-block hash
/// / find / gather as the unit of vectorisation.
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

    /// -------- BUILD SHUFFLE (full leaf depth, same code path as PHJ) --------
    const TimePoint t_bs0 = now();
    PartitionedShuffleOutput build_part = radixShuffle(build, cfg, threads, &tracker);

    /// Compact each chain's trailing partial OutBlock. The radix
    /// scatter's doubling-grow allocates each `OutBlock` at twice
    /// the previous block's capacity (capped at `MAX_OUT_BLOCK_ROWS`),
    /// so the last block per (thread, partition) chain typically
    /// holds significantly fewer rows than its allocated capacity.
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

    /// `bep_budget_mib` is the GLOBAL probe buffer budget, shared
    /// across all worker threads. The worker-level triggers below
    /// derive thresholds from `per_worker_view_budget = budget /
    /// threads`, so the sum of per-worker views (own unrefined +
    /// fair share of global published leaves) stays bounded by
    /// `budget_bytes` modulo a one-batch overshoot. The reported
    /// `bep_peak_mib` is the global probe-buffer peak across the
    /// whole run, directly comparable to `bep_budget_mib`.
    const size_t budget_bytes = bep_budget_mib * size_t{1024} * size_t{1024};
    const size_t per_worker_view_budget = budget_bytes / threads;

    /// Split the per-worker view budget into TWO independent triggers
    /// (option E hysteresis on each):
    ///
    ///   - unrefined: own per-worker pass-1 buffers. Worker T
    ///     refines its own biggest pass-1 chain when its own
    ///     `unrefined_row_bytes` crosses the high-water threshold,
    ///     until it dips back below the low-water threshold. No
    ///     coordination required — each worker manages its own
    ///     pass-1 budget.
    ///
    ///   - leaf share: global published-leaf capacity scaled to
    ///     per-worker share. When the per-worker share crosses
    ///     the leaf high-water threshold, EVERY worker enters
    ///     the cooperative drain loop (their local view of
    ///     global state is identical, modulo memory ordering).
    ///     They each independently pick the global argmax leaf
    ///     and drain it; the per-leaf CAS-claim and chain mutex
    ///     ensure workers settle on distinct leaves naturally.
    ///
    /// Splitting the trigger is the core fix for the work imbalance
    /// that a coupled trigger creates: with a unified
    /// `own_unrefined + leaf_share >= per_worker_view_budget` trigger,
    /// the worker with the larger own_unrefined triggers first and
    /// absorbs the (cross-worker, by design) leaf drain on its own
    /// while other workers continue scattering — turning option A's
    /// cross-worker coalescing into a serialisation bottleneck rather
    /// than a parallelisation win. With separate triggers the leaf
    /// drain is symmetric across workers and (A) actually runs in
    /// parallel.
    ///
    /// Split is 1/4 unrefined, 3/4 leaves of the per-worker view
    /// budget.
    const size_t unrefined_high_water_per_worker = per_worker_view_budget / 4;
    const size_t unrefined_low_water_per_worker = unrefined_high_water_per_worker / 2;
    const size_t leaf_share_high_water_per_worker = (per_worker_view_budget * 3) / 4;
    const size_t leaf_share_low_water_per_worker = (leaf_share_high_water_per_worker * 5) / 8;

    /// -------- SHARED PER-LEAF STATE --------
    std::vector<std::atomic<LeafState>> leaf_states(total_leaves);
    for (size_t l = 0; l < total_leaves; ++l)
        leaf_states[l].store(LeafState::NOT_BUILT, std::memory_order_relaxed);

    /// `JoinHashTable` is move-constructible / default-constructible; we
    /// pre-allocate `total_leaves` empty tables using the tracker so that
    /// the CAS-claimed constructor's cell allocations are tracked.
    std::pmr::vector<JoinHashTable> leaf_hts(&tracker);
    leaf_hts.resize(total_leaves);
    std::vector<std::unique_ptr<BlockStore>> leaf_stores(total_leaves);
    for (auto & p : leaf_stores)
        p = std::make_unique<BlockStore>(&tracker);

    /// Shared per-leaf probe-side chains. Every worker's refinement
    /// publishes into `published_leaves[L]` under `leaf_mutexes[L]`;
    /// eviction steals out of the same structure under the same
    /// mutex. The shared chain is grown ONLY by `push_back` of
    /// already-allocated OutBlocks (no `grow()` calls) — so the
    /// chain's `cur` pointer and `next_cap` stay at their default
    /// initial values and the chain doesn't auto-double its capacity.
    /// That also subsumes (D): there's no per-eviction `next_cap`
    /// reset to do because the doubling progression lives in the
    /// per-worker `intermediate[i]` chain (which is short-lived
    /// per-refinement) rather than the leaf chain itself.
    std::pmr::vector<PartitionOut> published_leaves(&tracker);
    published_leaves.resize(total_leaves);
    for (auto & po : published_leaves)
        initPartitionOut(po, probe.schema);

    std::vector<std::mutex> leaf_mutexes(total_leaves);

    /// Per-leaf row counter. Maintained alongside the chain (under
    /// the same mutex) so it's consistent with `published_leaves[L]
    /// .blocks` at every observable point. Argmax reads it atomically
    /// and uses it as a hint; the per-leaf mutex makes the "rows
    /// match blocks" invariant true. Multiplied by `bytes_per_row`
    /// it gives this leaf's buffered probe-data byte count.
    std::vector<std::atomic<size_t>> published_leaf_rows(total_leaves);
    for (size_t l = 0; l < total_leaves; ++l)
        published_leaf_rows[l].store(0, std::memory_order_relaxed);

    /// Aggregate row sum across all published leaves. Updated lock-
    /// free outside the per-leaf mutex (it is a sum and we don't
    /// need it to be sub-microsecond consistent with any single
    /// leaf's chain — it's a budget hint). Read once per worker per
    /// budget check to compute "share of global leaf buffered data".
    /// Tracks `rows`, NOT `capacity`: a leaf chain's allocated
    /// headroom (the OutBlock `capacity` field, set by the radix
    /// scatter's doubling-grow policy) is irrelevant to the
    /// "buffered probe data" semantic of the budget — only
    /// actually-filled rows count.
    std::atomic<size_t> global_leaf_rows{0};

    /// Aggregate per-worker unrefined buffered-row-bytes summed
    /// across every worker. Maintained via signed deltas: each
    /// worker, when it refreshes its own `unrefined_row_bytes`, adds
    /// the change to this global. Used only for global peak tracking
    /// (see `global_peak_probe_bytes` below) — the worker-local
    /// trigger still consults `s.unrefined_row_bytes` directly.
    std::atomic<size_t> global_unrefined_row_bytes{0};

    /// Running maximum of the GLOBAL probe-buffer footprint observed
    /// during the scatter+evict phase: `global_unrefined_row_bytes +
    /// global_leaf_rows * bytes_per_row`. Updated lock-free with a
    /// CAS loop on every per-block peak bump. This is the metric
    /// reported as `bep_peak_mib`; with `bep_budget_mib` now global
    /// AND row-based, the peak is directly comparable to the budget
    /// and reflects buffered probe DATA (not allocated buffer
    /// headroom, which can exceed rows by up to ~2× under the
    /// doubling-grow scheme).
    std::atomic<size_t> global_peak_probe_bytes{0};

    /// -------- PER-WORKER TIMING + COUNTERS --------
    std::vector<uint64_t> ns_build(threads, 0);
    std::vector<uint64_t> ns_probe_shuffle(threads, 0);
    std::vector<uint64_t> ns_probe(threads, 0);
    std::vector<uint64_t> ns_eviction(threads, 0);
    std::vector<size_t> worker_evictions(threads, 0);
    std::vector<size_t> worker_refinements(threads, 0);
    std::vector<size_t> worker_skip_retries(threads, 0);

    std::vector<WorkerProbeState> worker_states;
    worker_states.reserve(threads);
    for (size_t i = 0; i < threads; ++i)
        worker_states.emplace_back(&tracker);

    std::vector<ProbeMaterialiser> mats(threads);

    /// Drain-phase work-stealing counter. Each leaf is claimed by
    /// exactly one worker via `fetch_add`, restoring PHJ's single-
    /// owner-per-partition build+probe semantics for the drain.
    std::atomic<size_t> next_drain_leaf{0};

    /// Worker liveness counter for cooperative end-of-slice draining.
    /// Each worker decrements this when its input slice is exhausted;
    /// before exiting the scatter parallelRun the worker spins on
    /// global leaf draining as long as ANY worker is still scattering.
    /// Without this, fast workers (those whose input happened to land
    /// in cheap-to-refine partitions, or who were never the eviction
    /// trigger) would idle at the parallelRun barrier while slow
    /// workers carry the remaining eviction work alone — which is the
    /// dominant residual source of probe-phase imbalance once split
    /// triggers have parallelised the in-slice drain.
    std::atomic<size_t> scatter_workers_active{threads};

    /// -------- PER-THREAD PROBE LOOP --------
    parallelRun(
        threads,
        [&](size_t tid)
        {
            WorkerProbeState & s = worker_states[tid];
            s.unrefined.resize(P);
            s.unrefined_rows.resize(P, 0);
            s.intermediate.resize(leaves_per_p1);
            s.bytes_per_row = bytes_per_row;
            for (auto & po : s.unrefined)
                initPartitionOut(po, probe.schema);
            for (auto & po : s.intermediate)
                initPartitionOut(po, probe.schema);

            ScatterScratch scatter_scratch(&tracker);
            std::pmr::vector<uint64_t> build_hashes(&tracker);
            std::pmr::vector<uint64_t> probe_hashes(&tracker);
            std::pmr::vector<RowRefCell> heads(&tracker);
            std::pmr::vector<size_t> probe_idx(&tracker);
            std::pmr::vector<RowRefCell> build_ref(&tracker);

            ProbeMaterialiser & mat = mats[tid];
            mat.init(build.schema, probe.schema, result.output.workers[tid], PIPELINE_BLOCK_ROWS);

            uint64_t my_build_ns = 0;
            uint64_t my_probe_shuffle_ns = 0;
            uint64_t my_probe_ns = 0;
            uint64_t my_eviction_ns = 0;
            size_t my_evictions = 0;
            size_t my_refinements = 0;
            size_t my_skip_retries = 0;

            const size_t n_probe_blocks = probe.blocks.size();
            const size_t probe_start = (n_probe_blocks * tid) / threads;
            const size_t probe_end = (n_probe_blocks * (tid + 1)) / threads;

            /// Per-worker view of buffered probe DATA: own unrefined-
            /// chain rows × bytes_per_row plus a fair share of the
            /// globally published leaf-chain rows × bytes_per_row.
            /// Used for the worker-local triggers (compared against
            /// thresholds derived from `per_worker_view_budget =
            /// bep_budget_mib / threads`).
            auto leafShareBytes
                = [&]() noexcept -> size_t { return (global_leaf_rows.load(std::memory_order_relaxed) * bytes_per_row) / threads; };

            /// Refresh `s.unrefined_row_bytes` from the worker's
            /// chains and mirror the delta into
            /// `global_unrefined_row_bytes`. Called after every event
            /// that changes the worker's own unrefined buffers
            /// (scatter into pass-1, refinement that drains a
            /// pass-1 chain). Cheap: P chains × ~1-3 blocks each.
            auto refreshUnrefinedRowBytes = [&]() noexcept
            {
                const size_t new_val = unrefinedRowBytesFromScan(s);
                if (new_val >= s.unrefined_row_bytes)
                    global_unrefined_row_bytes.fetch_add(new_val - s.unrefined_row_bytes, std::memory_order_relaxed);
                else
                    global_unrefined_row_bytes.fetch_sub(s.unrefined_row_bytes - new_val, std::memory_order_relaxed);
                s.unrefined_row_bytes = new_val;
            };

            /// CAS-loop update of the GLOBAL probe-buffer peak. Every
            /// worker calls this once per input block after refreshing
            /// its own unrefined-row bookkeeping; the load of
            /// `global_unrefined_row_bytes` therefore includes this
            /// worker's latest contribution and a recent (possibly
            /// slightly stale) view of every other worker's.
            auto bumpGlobalPeak = [&]() noexcept
            {
                const size_t cur = global_unrefined_row_bytes.load(std::memory_order_relaxed)
                    + global_leaf_rows.load(std::memory_order_relaxed) * bytes_per_row;
                size_t old_peak = global_peak_probe_bytes.load(std::memory_order_relaxed);
                while (cur > old_peak && !global_peak_probe_bytes.compare_exchange_weak(old_peak, cur, std::memory_order_relaxed))
                {
                }
            };

            /// Pick this worker's largest own pass-1 (unrefined)
            /// chain. Returns SIZE_MAX if unrefined is empty.
            auto argmaxOwnUnrefined = [&]() noexcept -> size_t
            {
                size_t best_p1 = SIZE_MAX;
                size_t best_rows = 0;
                for (size_t p1 = 0; p1 < P; ++p1)
                {
                    const size_t r = s.unrefined_rows[p1];
                    if (r > best_rows)
                    {
                        best_rows = r;
                        best_p1 = p1;
                    }
                }
                return best_p1;
            };

            /// Pick the globally largest published leaf chain that
            /// is in `NOT_BUILT` or `BUILT` state (i.e., not already
            /// claimed by another worker as `BUILDING` or
            /// `STEALING`). Returns SIZE_MAX if no eligible leaf
            /// exists.
            auto argmaxGlobalLeaf = [&]() noexcept -> size_t
            {
                size_t best_l = SIZE_MAX;
                size_t best_rows = 0;
                for (size_t l = 0; l < total_leaves; ++l)
                {
                    const size_t r = published_leaf_rows[l].load(std::memory_order_relaxed);
                    if (r == 0)
                        continue;
                    const LeafState ls = leaf_states[l].load(std::memory_order_acquire);
                    if (ls == LeafState::BUILDING || ls == LeafState::STEALING)
                        continue;
                    if (r > best_rows)
                    {
                        best_rows = r;
                        best_l = l;
                    }
                }
                return best_l;
            };

            /// Force-refine an unrefined pass-1 partition through
            /// passes 2..N onto its leaf range. The intermediate
            /// destination is the worker's reusable `s.intermediate`
            /// array; once `refineToLeaves` returns, each non-empty
            /// `intermediate[i]` is move-published into the shared
            /// `published_leaves[base + i]` chain under the per-leaf
            /// mutex.
            auto refinePartition = [&](size_t p1)
            {
                /// Re-initialise the intermediate slots so their
                /// blocks vectors are empty (previous publication
                /// already moved out the blocks; we still reset
                /// `cur` and `next_cap` to canonical initial
                /// values).
                for (auto & po : s.intermediate)
                    initPartitionOut(po, probe.schema);

                refineToLeaves(std::move(s.unrefined[p1]), probe.schema, cfg.pass_bits, s.intermediate.data(), scatter_scratch, &tracker);

                /// `s.unrefined[p1]` is now in moved-from state with
                /// an empty blocks vector. Re-init it and refresh the
                /// global unrefined counter BEFORE publishing the
                /// refined intermediate into shared leaves. Without
                /// this ordering, `global_unrefined_row_bytes` would
                /// still include `p1`'s rows while `global_leaf_rows`
                /// has already absorbed them — momentarily double-
                /// counting `p1` in any concurrent global-peak CAS
                /// by another worker.
                initPartitionOut(s.unrefined[p1], probe.schema);
                s.unrefined_rows[p1] = 0;
                refreshUnrefinedRowBytes();

                const size_t base = p1 * leaves_per_p1;
                size_t global_added_rows = 0;
                for (size_t i = 0; i < leaves_per_p1; ++i)
                {
                    PartitionOut & src = s.intermediate[i];
                    if (src.blocks.empty())
                        continue;

                    size_t added_rows = 0;
                    for (const auto & blk : src.blocks)
                        added_rows += blk.rows;
                    if (added_rows == 0)
                        continue;

                    const size_t L = base + i;
                    {
                        std::lock_guard<std::mutex> lock(leaf_mutexes[L]);
                        for (auto & blk : src.blocks)
                        {
                            if (blk.rows > 0)
                                published_leaves[L].blocks.push_back(std::move(blk));
                        }
                        published_leaf_rows[L].fetch_add(added_rows, std::memory_order_relaxed);
                    }
                    src.blocks.clear();
                    src.cur = nullptr;
                    global_added_rows += added_rows;
                }
                if (global_added_rows != 0)
                    global_leaf_rows.fetch_add(global_added_rows, std::memory_order_relaxed);
            };

            /// CAS-claim leaf L and (if claimed) construct its HT
            /// from `build_part.chains[L]`. The transition to
            /// `BUILT` uses release ordering so subsequent acquire
            /// loads see the constructor's writes.
            auto buildLeafHt = [&](size_t leaf)
            {
                BlockStore & store = *leaf_stores[leaf];
                store.reserveBlocks(build_part.chains[leaf].blocks.size());
                JoinHashTable & ht = leaf_hts[leaf];
                if (build_part.partition_rows[leaf] > 0)
                    ht.reserve(build_part.partition_rows[leaf]);
                for (auto & ob : build_part.chains[leaf].blocks)
                {
                    Block as_block = outBlockToBlock(std::move(ob));
                    buildOneBlock(std::move(as_block), store, ht, build_hashes);
                }
                build_part.chains[leaf].blocks.clear();
            };

            /// Steal the shared leaf chain's blocks into a worker-local
            /// PartitionOut (so the lock is released as soon as the
            /// vector moves are done, before the probe sweep starts).
            /// Returns the number of rows stolen (0 if another worker
            /// raced in and drained first).
            auto stealLeafChain = [&](size_t leaf, PartitionOut & dst) noexcept -> size_t
            {
                size_t stolen_rows = 0;
                {
                    std::lock_guard<std::mutex> lock(leaf_mutexes[leaf]);
                    for (auto & blk : published_leaves[leaf].blocks)
                    {
                        if (blk.rows > 0)
                        {
                            stolen_rows += blk.rows;
                            dst.blocks.push_back(std::move(blk));
                        }
                    }
                    published_leaves[leaf].blocks.clear();
                    published_leaves[leaf].cur = nullptr;
                    /// Drain the per-leaf counter inside the mutex so
                    /// the (chain, counter) pair stays consistent.
                    published_leaf_rows[leaf].fetch_sub(stolen_rows, std::memory_order_relaxed);
                }
                if (stolen_rows != 0)
                    global_leaf_rows.fetch_sub(stolen_rows, std::memory_order_relaxed);
                return stolen_rows;
            };

            /// Process one leaf-eviction step: pick the globally
            /// largest published leaf, build its HT if needed,
            /// steal its chain (under per-leaf mutex), probe it,
            /// drop. Returns true if forward progress was made,
            /// false on a skip (claim/build collision or empty
            /// chain after the steal). On `false` the caller
            /// re-checks the budget condition; on a sustained
            /// no-progress streak the caller yields and retries.
            auto evictOneGlobalLeaf = [&]() -> bool
            {
                const TimePoint te0 = now();
                const size_t leaf = argmaxGlobalLeaf();
                my_eviction_ns += toNanos(now() - te0);
                if (leaf == SIZE_MAX)
                    return false;

                /// Claim the leaf for the ENTIRE evict operation
                /// (build-if-needed + steal + probe + drop) via the
                /// `LeafState` machine. `argmaxGlobalLeaf` already
                /// filtered out leaves in `BUILDING` / `STEALING`,
                /// but a concurrent worker may have transitioned
                /// the leaf between our argmax and our CAS — that's
                /// the natural retry point.
                ///
                /// CRITICAL: this is what eliminates the cross-
                /// worker mutex contention on the per-leaf chain.
                /// Without the `STEALING` claim 48 workers
                /// converged on the same `BUILT` leaf, contended
                /// on `leaf_mutexes[L]`, and 47 of them stole an
                /// empty chain (~85-91% waste rate, ~3.2 s/thread
                /// in the 1 GiB regime).
                const TimePoint ts0 = now();
                LeafState st = leaf_states[leaf].load(std::memory_order_acquire);
                my_eviction_ns += toNanos(now() - ts0);

                if (st == LeafState::BUILDING || st == LeafState::STEALING)
                {
                    ++my_skip_retries;
                    return false;
                }
                if (st == LeafState::NOT_BUILT)
                {
                    const TimePoint tc0 = now();
                    LeafState expected = LeafState::NOT_BUILT;
                    const bool claimed
                        = leaf_states[leaf].compare_exchange_strong(expected, LeafState::BUILDING, std::memory_order_acq_rel);
                    my_eviction_ns += toNanos(now() - tc0);
                    if (!claimed)
                    {
                        ++my_skip_retries;
                        return false;
                    }
                    const TimePoint tb0 = now();
                    buildLeafHt(leaf);
                    /// Build complete. Transition straight to
                    /// `STEALING` rather than `BUILT` — we're about
                    /// to consume the leaf ourselves, so there's no
                    /// `BUILT`-window for another worker to race
                    /// against.
                    leaf_states[leaf].store(LeafState::STEALING, std::memory_order_release);
                    my_build_ns += toNanos(now() - tb0);
                }
                else
                {
                    /// `st == BUILT`. CAS-claim `BUILT -> STEALING`
                    /// so we hold the leaf exclusively for the
                    /// steal+probe+drop. A failed CAS means another
                    /// worker beat us to the claim; retry from the
                    /// next argmax.
                    const TimePoint tc0 = now();
                    LeafState expected = LeafState::BUILT;
                    const bool claimed
                        = leaf_states[leaf].compare_exchange_strong(expected, LeafState::STEALING, std::memory_order_acq_rel);
                    my_eviction_ns += toNanos(now() - tc0);
                    if (!claimed)
                    {
                        ++my_skip_retries;
                        return false;
                    }
                }

                /// We hold the `STEALING` claim on `leaf`. No other
                /// worker can be in `stealLeafChain(leaf, ...)`
                /// concurrently; publishers (refining workers) can
                /// still publish into `published_leaves[L]` under
                /// the per-leaf mutex, but the stealer is unique.
                PartitionOut stolen(&tracker);
                const TimePoint tst0 = now();
                const size_t stolen_rows = stealLeafChain(leaf, stolen);
                my_eviction_ns += toNanos(now() - tst0);

                if (stolen_rows == 0)
                {
                    /// Nothing to steal (the published rows we
                    /// observed pre-CAS were drained between the
                    /// argmax and the steal — possible if a
                    /// publisher cleared its `intermediate[i]`
                    /// after our argmax read `published_leaf_rows`
                    /// but before the publish actually committed
                    /// blocks; see the relaxed-counter ordering in
                    /// `refinePartition`). Release the claim back
                    /// to `BUILT` so future argmax can pick it up
                    /// once new rows accrue.
                    leaf_states[leaf].store(LeafState::BUILT, std::memory_order_release);
                    return false;
                }

                const TimePoint tp0 = now();
                const JoinHashTable & ht = leaf_hts[leaf];
                const BlockStore & store = *leaf_stores[leaf];
                probeChain(stolen.blocks, store, ht, mat, probe_hashes, heads, probe_idx, build_ref);
                my_probe_ns += toNanos(now() - tp0);

                const TimePoint td0 = now();
                dropPartition(stolen);
                /// Release the `STEALING` claim. `BUILT` is the
                /// canonical "idle, HT present" state.
                leaf_states[leaf].store(LeafState::BUILT, std::memory_order_release);
                my_eviction_ns += toNanos(now() - td0);
                ++my_evictions;
                return true;
            };

            /// Post-input-block trigger check. Two independent
            /// hysteresis loops; refinement runs before leaf drain
            /// because refinement publishes rows from unrefined
            /// into the shared leaf chains and may push the leaf
            /// share over the leaf high-water threshold.
            auto evictAsNeeded = [&]()
            {
                /// Phase 1: drain own unrefined to low water.
                if (s.unrefined_row_bytes >= unrefined_high_water_per_worker)
                {
                    while (s.unrefined_row_bytes >= unrefined_low_water_per_worker)
                    {
                        const size_t p1 = argmaxOwnUnrefined();
                        if (p1 == SIZE_MAX)
                            break;
                        const TimePoint trf0 = now();
                        refinePartition(p1);
                        my_probe_shuffle_ns += toNanos(now() - trf0);
                        ++my_refinements;
                    }
                }

                /// Phase 2: cooperative drain of global leaves.
                /// Bounded no-progress backoff so a worker doesn't
                /// spin forever if every non-empty leaf is being
                /// built elsewhere; the brief sleep matches the
                /// behaviour of the prior single-trigger loop.
                if (leafShareBytes() >= leaf_share_high_water_per_worker)
                {
                    size_t consecutive_no_progress = 0;
                    while (leafShareBytes() >= leaf_share_low_water_per_worker)
                    {
                        if (evictOneGlobalLeaf())
                        {
                            consecutive_no_progress = 0;
                        }
                        else if (++consecutive_no_progress >= 16)
                        {
                            const TimePoint tb0 = now();
                            std::this_thread::sleep_for(std::chrono::microseconds(50));
                            my_eviction_ns += toNanos(now() - tb0);
                            consecutive_no_progress = 0;
                        }
                    }
                }
            };

            /// -------- MAIN LOOP --------
            const std::vector<size_t> probe_identity = [&]
            {
                std::vector<size_t> v(probe.schema.types.size());
                for (size_t c = 0; c < v.size(); ++c)
                    v[c] = c;
                return v;
            }();

            for (size_t b = probe_start; b < probe_end; ++b)
            {
                const Block & blk = probe.blocks[b];
                if (blk.rows == 0)
                    continue;

                /// Pass-1 scatter of one input block into unrefined chains.
                const TimePoint tps0 = now();
                scatterBatch(blk.view(), probe_identity, probe.schema, pass1_shift, P, s.unrefined.data(), scatter_scratch);
                for (size_t p = 0; p < P; ++p)
                {
                    const size_t delta = scatter_scratch.local_hist[p];
                    if (delta == 0)
                        continue;
                    s.unrefined_rows[p] += delta;
                }
                /// Refresh cached unrefined row-bytes after scatter,
                /// mirror the delta into the global aggregate, then
                /// refresh the global peak. Row-based accounting so
                /// the budget bounds buffered probe DATA, not
                /// allocated buffer headroom (see
                /// `unrefinedRowBytesFromScan`).
                refreshUnrefinedRowBytes();
                bumpGlobalPeak();
                my_probe_shuffle_ns += toNanos(now() - tps0);

                evictAsNeeded();
            }

            /// -------- COOPERATIVE END-OF-SLICE DRAIN --------
            /// This worker has finished its input slice. Before
            /// refining its own leftover unrefined (which would
            /// publish more rows into shared leaves and then trip
            /// other workers' triggers), help drain the global
            /// leaves as long as at least one other worker is
            /// still scattering. This converts the natural barrier
            /// idle time into useful eviction work and balances
            /// the cross-worker probe load.
            scatter_workers_active.fetch_sub(1, std::memory_order_acq_rel);
            while (scatter_workers_active.load(std::memory_order_acquire) > 0)
            {
                if (leafShareBytes() >= leaf_share_low_water_per_worker)
                {
                    if (!evictOneGlobalLeaf())
                    {
                        const TimePoint tb0 = now();
                        std::this_thread::yield();
                        my_eviction_ns += toNanos(now() - tb0);
                    }
                }
                else
                {
                    const TimePoint tb0 = now();
                    std::this_thread::yield();
                    my_eviction_ns += toNanos(now() - tb0);
                }
            }

            /// -------- END-OF-INPUT REFINEMENT --------
            /// Force-refine every pass-1 partition that still holds
            /// rows. Publishes into shared leaves under per-leaf
            /// mutexes (other workers may still be draining their
            /// own pass-1 buffers here, but contention on the same
            /// leaf is rare and brief).
            for (size_t p1 = 0; p1 < P; ++p1)
            {
                if (s.unrefined_rows[p1] == 0)
                    continue;
                const TimePoint trf0 = now();
                refinePartition(p1);
                my_probe_shuffle_ns += toNanos(now() - trf0);
                ++my_refinements;
            }

            ns_build[tid] = my_build_ns;
            ns_probe_shuffle[tid] = my_probe_shuffle_ns;
            ns_probe[tid] = my_probe_ns;
            ns_eviction[tid] = my_eviction_ns;
            worker_evictions[tid] = my_evictions;
            worker_refinements[tid] = my_refinements;
            worker_skip_retries[tid] = my_skip_retries;
        });

    /// -------- POST-SCATTER HT CLEANUP --------
    /// parallelRun #1 has joined, so no concurrent reader of `leaf_hts`
    /// or `leaf_stores` can exist. Eagerly free leaves whose published
    /// chain holds no residual: those are leaves that mid-stream
    /// eviction processed completely (HT was built but no probe rows
    /// remain for the drain). Without this pass, all `total_leaves`
    /// HTs that were built during scatter would stay alive into the
    /// drain.
    for (size_t L = 0; L < total_leaves; ++L)
    {
        if (published_leaf_rows[L].load(std::memory_order_relaxed) != 0)
            continue;
        /// Tracker-backed assignment so the cells buffer is actually
        /// released. Assigning from a default-constructed JHT (which
        /// uses `get_default_resource()`) would NOT free the buffer
        /// because `polymorphic_allocator::POCMA = false`.
        leaf_hts[L] = JoinHashTable(&tracker);
        leaf_stores[L].reset();
    }

    /// -------- WORK-STEALING DRAIN (PHJ-EQUIVALENT) --------
    /// Mid-stream eviction has already processed (and dropped) the
    /// slices it was forced to flush; whatever remains lives in the
    /// shared `published_leaves[L]` chains. We drain that residual
    /// in PHJ-style: one fetch_add-claimed leaf per worker, build
    /// HT if NOT_BUILT, probe end-to-end.
    parallelRun(
        threads,
        [&](size_t tid)
        {
            std::pmr::vector<uint64_t> build_hashes(&tracker);
            std::pmr::vector<uint64_t> probe_hashes(&tracker);
            std::pmr::vector<RowRefCell> heads(&tracker);
            std::pmr::vector<size_t> probe_idx(&tracker);
            std::pmr::vector<RowRefCell> build_ref(&tracker);

            ProbeMaterialiser & mat = mats[tid];

            uint64_t my_build_ns = 0;
            uint64_t my_probe_ns = 0;
            uint64_t my_eviction_ns = 0;
            size_t my_evictions = 0;

            auto buildLeafHt = [&](size_t leaf)
            {
                BlockStore & store = *leaf_stores[leaf];
                store.reserveBlocks(build_part.chains[leaf].blocks.size());
                JoinHashTable & ht = leaf_hts[leaf];
                if (build_part.partition_rows[leaf] > 0)
                    ht.reserve(build_part.partition_rows[leaf]);
                for (auto & ob : build_part.chains[leaf].blocks)
                {
                    Block as_block = outBlockToBlock(std::move(ob));
                    buildOneBlock(std::move(as_block), store, ht, build_hashes);
                }
                build_part.chains[leaf].blocks.clear();
            };

            while (true)
            {
                const size_t L = next_drain_leaf.fetch_add(1, std::memory_order_relaxed);
                if (L >= total_leaves)
                    break;

                /// Steal the leaf's residual chain. parallelRun #1
                /// has joined, so `published_leaves[L]` is now
                /// only modifiable by us — no lock needed.
                PartitionOut stolen(&tracker);
                size_t stolen_rows = 0;
                for (auto & blk : published_leaves[L].blocks)
                {
                    if (blk.rows > 0)
                    {
                        stolen_rows += blk.rows;
                        stolen.blocks.push_back(std::move(blk));
                    }
                }
                published_leaves[L].blocks.clear();
                published_leaves[L].cur = nullptr;

                if (stolen_rows == 0)
                {
                    /// No residual probe rows. HT and store may
                    /// already have been freed in the post-scatter
                    /// cleanup; if not (because the leaf was built
                    /// mid-stream and had subsequent residual that
                    /// got cleared between cleanup and drain), free
                    /// them now.
                    leaf_hts[L] = JoinHashTable(&tracker);
                    leaf_stores[L].reset();
                    continue;
                }

                if (leaf_states[L].load(std::memory_order_acquire) == LeafState::NOT_BUILT)
                {
                    const TimePoint tb0 = now();
                    buildLeafHt(L);
                    leaf_states[L].store(LeafState::BUILT, std::memory_order_release);
                    my_build_ns += toNanos(now() - tb0);
                }

                const TimePoint tp0 = now();
                const JoinHashTable & ht = leaf_hts[L];
                const BlockStore & store = *leaf_stores[L];
                probeChain(stolen.blocks, store, ht, mat, probe_hashes, heads, probe_idx, build_ref);
                my_probe_ns += toNanos(now() - tp0);

                const TimePoint td0 = now();
                dropPartition(stolen);
                /// Tracker-backed assignment so the move-assign can
                /// steal and free the cells buffer.
                leaf_hts[L] = JoinHashTable(&tracker);
                leaf_stores[L].reset();
                my_eviction_ns += toNanos(now() - td0);
                ++my_evictions;
            }

            mat.finish();

            ns_build[tid] += my_build_ns;
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

    return result;
}

}
