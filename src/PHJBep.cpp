#include "PHJBep.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "BlockStore.h"
#include "HashTable.h"
#include "JoinOps.h"
#include "Threading.h"
#include "Timer.h"


namespace phj
{

namespace
{

/// Per-leaf HT lifecycle. Strictly monotonic transitions
/// `NOT_BUILT -> BUILDING -> BUILT`, claimed via CAS by the first
/// worker that selects the leaf for eviction. Acquire/release atomic
/// ordering establishes happens-before from the constructor's writes
/// to `leaf_hts[L]` / `leaf_stores[L]` (within `BUILDING`) to any
/// subsequent reader that observes `BUILT`.
enum class LeafState : uint8_t
{
    NOT_BUILT = 0,
    BUILDING = 1,
    BUILT = 2,
};


/// Per-worker probe-side buffer state. The worker holds two parallel
/// arrays of `PartitionOut` chains drawn from the radix scatter
/// machinery:
///
///   - `unrefined[p1]`  — pass-1 buckets; receives every scatter_pass1
///                        output for pass-1 partition id `p1`. A pass-1
///                        bucket can accumulate rows again after being
///                        previously refined, since pass 1 is the only
///                        scatter applied to incoming probe blocks.
///   - `leaf[l]`        — leaf buckets; populated only by refinement.
///                        Refining `p1` appends its currently-buffered
///                        rows into `leaf[p1 * leaves_per_p1 .. ...)`,
///                        coexisting with whatever earlier refinements
///                        of the same bucket already deposited.
///
/// `unrefined_rows[p1]` and `leaf_rows[l]` mirror each chain's filled
/// row count; the eviction argmax reads them directly and the trigger
/// uses `total_rows * bytes_per_row` against the budget.
struct WorkerProbeState
{
    /// Pass-1 chains, indexed by pass-1 partition id `p1 \in [0, P)`.
    std::vector<PartitionOut> unrefined;
    std::vector<size_t> unrefined_rows;

    /// Leaf chains, indexed by global leaf id `l \in [0, total_leaves)`.
    std::vector<PartitionOut> leaf;
    std::vector<size_t> leaf_rows;

    /// Running totals (used for both the trigger check and argmax).
    size_t total_rows = 0;
    size_t peak_rows = 0;

    /// Probe-row byte size: `sizeof(uint64_t)` for the key column plus
    /// `probe_schema.rowByteSize()`. Counts toward the per-worker
    /// memory budget M; refinement is row-preserving so we use a row
    /// count rather than a byte count internally.
    size_t bytes_per_row = 0;
};


struct EvictTarget
{
    /// 0 = unrefined pass-1 partition, 1 = leaf.
    int kind = 0;
    size_t idx = 0;
    size_t rows = 0;
};


[[gnu::always_inline]] inline bool overBudget(const WorkerProbeState & s, size_t budget_bytes) noexcept
{
    return s.total_rows * s.bytes_per_row >= budget_bytes;
}


/// Pick the largest buffered partition across both structural states.
/// Leaves currently in `BUILDING` are skipped; their state is loaded
/// with acquire ordering so a subsequent successful probe load of
/// `BUILT` will see the constructor's HT / store writes.
EvictTarget argmaxBuffered(const WorkerProbeState & s, const std::vector<std::atomic<LeafState>> & leaf_states) noexcept
{
    EvictTarget best{};
    const size_t P = s.unrefined.size();
    for (size_t p1 = 0; p1 < P; ++p1)
    {
        const size_t r = s.unrefined_rows[p1];
        if (r == 0)
            continue;
        if (r > best.rows)
            best = {0, p1, r};
    }
    const size_t L = s.leaf.size();
    for (size_t l = 0; l < L; ++l)
    {
        const size_t r = s.leaf_rows[l];
        if (r == 0)
            continue;
        if (leaf_states[l].load(std::memory_order_acquire) == LeafState::BUILDING)
            continue;
        if (r > best.rows)
            best = {1, l, r};
    }
    return best;
}

}


PhjBepResult runPhjBep(const BlockStream & build, const BlockStream & probe, const RadixConfig & cfg, size_t threads, size_t bep_budget_mib)
{
    if (threads == 0)
        threads = 1;

    PhjBepResult result;
    result.output.left_schema = build.schema;
    result.output.right_schema = probe.schema;
    result.output.workers.resize(threads);
    result.bep_budget_mib = bep_budget_mib;

    const TimePoint t_e2e0 = now();

    /// -------- BUILD SHUFFLE (full leaf depth, same code path as PHJ) --------
    const TimePoint t_bs0 = now();
    PartitionedShuffleOutput build_part = radixShuffle(build, cfg, threads);
    const TimePoint t_bs1 = now();
    const uint64_t build_shuffle_ns = toNanos(t_bs1 - t_bs0);

    const size_t total_leaves = build_part.partitions;
    const size_t pass1_bits = cfg.pass_bits.front();
    const size_t P = size_t{1} << pass1_bits;
    const size_t leaves_per_p1 = total_leaves / P;
    const uint32_t pass1_shift = static_cast<uint32_t>(64u - pass1_bits);
    const size_t bytes_per_row = sizeof(uint64_t) + probe.schema.rowByteSize();
    const size_t budget_bytes = bep_budget_mib * size_t{1024} * size_t{1024};

    /// -------- SHARED PER-LEAF STATE (HT lifecycle) --------
    std::vector<std::atomic<LeafState>> leaf_states(total_leaves);
    for (size_t l = 0; l < total_leaves; ++l)
        leaf_states[l].store(LeafState::NOT_BUILT, std::memory_order_relaxed);

    /// `JoinHashTable` is move-constructible / default-constructible; we
    /// pre-allocate `total_leaves` empty tables and the CAS-claimed
    /// constructor populates table `L` in place. `BlockStore` is neither
    /// movable nor copyable (carries a mutex), so we hold each store via
    /// `unique_ptr`.
    std::vector<JoinHashTable> leaf_hts(total_leaves);
    std::vector<std::unique_ptr<BlockStore>> leaf_stores(total_leaves);
    for (auto & p : leaf_stores)
        p = std::make_unique<BlockStore>();

    /// -------- PER-WORKER TIMING + COUNTERS --------
    std::vector<uint64_t> ns_build(threads, 0);
    std::vector<uint64_t> ns_probe_shuffle(threads, 0);
    std::vector<uint64_t> ns_probe(threads, 0);
    std::vector<uint64_t> ns_eviction(threads, 0);
    std::vector<size_t> worker_evictions(threads, 0);
    std::vector<size_t> worker_refinements(threads, 0);
    std::vector<size_t> worker_peak_rows(threads, 0);
    std::vector<size_t> worker_skip_retries(threads, 0);

    /// Hoisted per-worker state. The work-stealing drain below runs in
    /// a second `parallelRun`, so the worker buffers and the per-thread
    /// materialiser must outlive the scatter lambda: the drain reads
    /// every worker's `s.leaf[L]` chains (to merge them per leaf) and
    /// keeps appending output rows through the same materialiser.
    std::vector<WorkerProbeState> worker_states(threads);
    std::vector<ProbeMaterialiser> mats(threads);

    /// Drain-phase work-stealing counter. Each leaf is claimed by
    /// exactly one worker via `fetch_add`, restoring PHJ's single-
    /// owner-per-partition build+probe semantics for the drain.
    std::atomic<size_t> next_drain_leaf{0};

    /// -------- PER-THREAD PROBE LOOP --------
    parallelRun(
        threads,
        [&](size_t tid)
        {
            WorkerProbeState & s = worker_states[tid];
            s.unrefined.assign(P, PartitionOut{});
            s.unrefined_rows.assign(P, 0);
            s.leaf.assign(total_leaves, PartitionOut{});
            s.leaf_rows.assign(total_leaves, 0);
            s.bytes_per_row = bytes_per_row;
            for (auto & po : s.unrefined)
                initPartitionOut(po, probe.schema);
            for (auto & po : s.leaf)
                initPartitionOut(po, probe.schema);

            ScatterScratch scatter_scratch;
            std::vector<uint64_t> build_hashes;
            std::vector<uint64_t> probe_hashes;
            std::vector<RowRefCell> heads;
            std::vector<size_t> probe_idx;
            std::vector<RowRefCell> build_ref;

            ProbeMaterialiser & mat = mats[tid];
            mat.init(build.schema, probe.schema, result.output.workers[tid], PIPELINE_BLOCK_ROWS);

            uint64_t my_build_ns = 0;
            uint64_t my_probe_shuffle_ns = 0;
            uint64_t my_probe_ns = 0;
            uint64_t my_eviction_ns = 0;
            size_t my_evictions = 0;
            size_t my_refinements = 0;
            size_t my_skip_retries = 0;

            /// Worker's slice of probe input blocks.
            const size_t n_probe_blocks = probe.blocks.size();
            const size_t probe_start = (n_probe_blocks * tid) / threads;
            const size_t probe_end = (n_probe_blocks * (tid + 1)) / threads;

            /// Force-refine an unrefined pass-1 partition through passes
            /// 2..N onto leaves under its leaf range. Refinement is
            /// row-preserving across the rows it consumed from
            /// `unrefined[p1]`. The leaf chains may already hold rows
            /// from a previous refinement of the same `p1`; the new
            /// rows are appended onto them. Total per-worker rows are
            /// unchanged.
            auto refinePartition = [&](size_t p1)
            {
                const size_t base = p1 * leaves_per_p1;
                /// Snapshot the per-leaf row counts before refinement
                /// so we can compute the per-leaf delta cheaply (rather
                /// than scanning every leaf's chain from scratch).
                std::vector<size_t> before(leaves_per_p1);
                for (size_t i = 0; i < leaves_per_p1; ++i)
                    before[i] = partitionRows(s.leaf[base + i]);

                refineToLeaves(std::move(s.unrefined[p1]), probe.schema, cfg.pass_bits, s.leaf.data() + base, scatter_scratch);

                for (size_t i = 0; i < leaves_per_p1; ++i)
                {
                    const size_t after = partitionRows(s.leaf[base + i]);
                    s.leaf_rows[base + i] += (after - before[i]);
                }
                /// `s.unrefined[p1]` is now moved-from (an empty chain
                /// with possibly a dangling `cur`). Re-initialise so
                /// future pass-1 scatters of incoming probe blocks
                /// land into a valid chain — the same bucket can be
                /// re-refined later when it accumulates again.
                initPartitionOut(s.unrefined[p1], probe.schema);
                s.unrefined_rows[p1] = 0;
            };

            /// CAS-claim leaf L and (if claimed) construct its HT from
            /// `build_part.chains[L]`. After construction, transition to
            /// `BUILT` with release ordering so subsequent acquire
            /// readers see the HT / store writes.
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

            /// Probe-side processing: scan the worker's local leaf chain
            /// against the shared `leaf_hts[leaf]` / `leaf_stores[leaf]`.
            auto processPartition = [&](size_t leaf)
            {
                if (s.leaf[leaf].blocks.empty())
                    return;
                const JoinHashTable & ht = leaf_hts[leaf];
                const BlockStore & store = *leaf_stores[leaf];
                for (const auto & ob : s.leaf[leaf].blocks)
                {
                    if (ob.rows == 0)
                        continue;
                    probeOneBlock(ob.view(), store, ht, mat, probe_hashes, heads, probe_idx, build_ref);
                }
            };

            auto dropLeafProbeBuffers = [&](size_t leaf)
            {
                s.total_rows -= s.leaf_rows[leaf];
                s.leaf_rows[leaf] = 0;
                dropPartition(s.leaf[leaf]);
            };

            auto bumpPeak = [&]() noexcept { s.peak_rows = std::max(s.peak_rows, s.total_rows); };

            /// Inner eviction loop: invoked at every input-block boundary
            /// (and once at end of input, via the drain phase below).
            /// Returns when total buffered bytes drop below `M`.
            auto evictUntilUnderBudget = [&]()
            {
                while (overBudget(s, budget_bytes))
                {
                    TimePoint te0 = now();
                    EvictTarget target = argmaxBuffered(s, leaf_states);

                    if (target.rows == 0)
                    {
                        /// Every non-empty leaf is `BUILDING` elsewhere
                        /// and no unrefined data remains. Brief backoff.
                        /// The sleep counts toward `eviction_overhead_ns`
                        /// per spec ("`BUILDING`-skip / backoff").
                        ++my_skip_retries;
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                        my_eviction_ns += toNanos(now() - te0);
                        continue;
                    }
                    my_eviction_ns += toNanos(now() - te0);

                    if (target.kind == 0)
                    {
                        const TimePoint trf0 = now();
                        refinePartition(target.idx);
                        my_probe_shuffle_ns += toNanos(now() - trf0);
                        ++my_refinements;
                        continue;
                    }

                    /// Leaf. Acquire-load state and dispatch.
                    const TimePoint ts0 = now();
                    LeafState st = leaf_states[target.idx].load(std::memory_order_acquire);
                    my_eviction_ns += toNanos(now() - ts0);

                    if (st == LeafState::BUILT)
                    {
                        const TimePoint tp0 = now();
                        processPartition(target.idx);
                        my_probe_ns += toNanos(now() - tp0);
                        const TimePoint td0 = now();
                        dropLeafProbeBuffers(target.idx);
                        my_eviction_ns += toNanos(now() - td0);
                        ++my_evictions;
                        continue;
                    }
                    if (st == LeafState::NOT_BUILT)
                    {
                        const TimePoint tc0 = now();
                        LeafState expected = LeafState::NOT_BUILT;
                        const bool claimed
                            = leaf_states[target.idx].compare_exchange_strong(expected, LeafState::BUILDING, std::memory_order_acq_rel);
                        my_eviction_ns += toNanos(now() - tc0);
                        if (claimed)
                        {
                            const TimePoint tb0 = now();
                            buildLeafHt(target.idx);
                            leaf_states[target.idx].store(LeafState::BUILT, std::memory_order_release);
                            my_build_ns += toNanos(now() - tb0);

                            const TimePoint tp0 = now();
                            processPartition(target.idx);
                            my_probe_ns += toNanos(now() - tp0);
                            const TimePoint td0 = now();
                            dropLeafProbeBuffers(target.idx);
                            my_eviction_ns += toNanos(now() - td0);
                            ++my_evictions;
                            continue;
                        }
                        ++my_skip_retries;
                        continue;
                    }
                    /// Raced into `BUILDING` between argmax and load.
                    ++my_skip_retries;
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
                /// `scatter_scratch.local_hist[p]` is the per-partition
                /// row delta for this batch.
                for (size_t p = 0; p < P; ++p)
                {
                    const size_t delta = scatter_scratch.local_hist[p];
                    if (delta == 0)
                        continue;
                    s.unrefined_rows[p] += delta;
                    s.total_rows += delta;
                }
                bumpPeak();
                my_probe_shuffle_ns += toNanos(now() - tps0);

                evictUntilUnderBudget();
            }

            /// -------- END-OF-INPUT DRAIN --------
            /// Force-refine every pass-1 partition that still holds
            /// rows. Per-worker only, no cross-worker contention.
            for (size_t p1 = 0; p1 < P; ++p1)
            {
                if (s.unrefined_rows[p1] == 0)
                    continue;
                const TimePoint trf0 = now();
                refinePartition(p1);
                my_probe_shuffle_ns += toNanos(now() - trf0);
                ++my_refinements;
            }

            /// End of parallelRun #1. The old per-worker leaf drain
            /// (offset-stepped walk with CAS-claimed builds and a
            /// pending/backoff loop) has been replaced by the second
            /// parallelRun below, which restores PHJ's single-owner-
            /// per-partition build+probe by work-stealing leaves and
            /// merging the per-worker `s.leaf[L]` chains lazily under
            /// fetch_add-exclusive ownership.
            ns_build[tid] = my_build_ns;
            ns_probe_shuffle[tid] = my_probe_shuffle_ns;
            ns_probe[tid] = my_probe_ns;
            ns_eviction[tid] = my_eviction_ns;
            worker_evictions[tid] = my_evictions;
            worker_refinements[tid] = my_refinements;
            worker_peak_rows[tid] = s.peak_rows;
            worker_skip_retries[tid] = my_skip_retries;
        });

    /// -------- WORK-STEALING DRAIN (PHJ-EQUIVALENT) --------
    /// Mid-stream eviction has already processed (and dropped) the
    /// slices it was forced to flush; whatever remains lives in each
    /// worker's `s.leaf[L]` chains. We now drain that residual exactly
    /// like PHJ drains its post-shuffle partitions:
    ///
    ///   1. A shared atomic `next_drain_leaf` work-steals leaves. Each
    ///      successful `fetch_add` gives the calling worker exclusive
    ///      ownership of leaf L for the drain phase — no other worker
    ///      will read `worker_states[*].leaf[L]` once this one starts.
    ///
    ///   2. The worker lazily cross-merges leaf L by walking every
    ///      `worker_states[t].leaf[L]` and moving its OutBlocks into a
    ///      local chain. Since the merge moves blocks (no copies) and
    ///      the per-worker chains are unobserved by anyone else, no
    ///      synchronisation is required.
    ///
    ///   3. If mid-stream eviction never built leaf L's HT, the
    ///      claiming worker builds it now. The fetch_add already
    ///      guarantees uniqueness, so the CAS dance from the old drain
    ///      collapses to a plain store of `BUILT`.
    ///
    ///   4. The merged chain is probed end-to-end against the leaf's
    ///      shared HT, with build and probe staying on the same worker
    ///      for any leaf that was lazy-built in step 3 (HT-hot cache,
    ///      matching PHJ). For leaves whose HT was already built mid-
    ///      stream the worker pays the same cold-HT cost this BEP path
    ///      has always paid for those leaves.
    parallelRun(
        threads,
        [&](size_t tid)
        {
            std::vector<uint64_t> build_hashes;
            std::vector<uint64_t> probe_hashes;
            std::vector<RowRefCell> heads;
            std::vector<size_t> probe_idx;
            std::vector<RowRefCell> build_ref;

            ProbeMaterialiser & mat = mats[tid];

            uint64_t my_build_ns = 0;
            uint64_t my_probe_ns = 0;
            uint64_t my_probe_shuffle_ns = 0;
            uint64_t my_eviction_ns = 0;
            size_t my_evictions = 0;

            /// Re-declared here (rather than hoisted as a free function)
            /// so each drain worker has its own scratch hashes vector
            /// captured by reference.
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

                /// Lazy cross-worker merge for leaf L. fetch_add gave
                /// us exclusive ownership of `worker_states[*].leaf[L]`
                /// for the drain phase, so the moves below need no
                /// synchronisation. The merge time is attributed to
                /// `probe_shuffle` since it's the analogue of PHJ's
                /// probe-side shuffle for this leaf.
                const TimePoint tm0 = now();
                PartitionOut merged;
                size_t merged_rows = 0;
                for (size_t t = 0; t < threads; ++t)
                {
                    auto & src = worker_states[t].leaf[L];
                    for (auto & blk : src.blocks)
                    {
                        if (blk.rows > 0)
                        {
                            merged_rows += blk.rows;
                            merged.blocks.push_back(std::move(blk));
                        }
                    }
                    src.blocks.clear();
                    src.cur = nullptr;
                }
                my_probe_shuffle_ns += toNanos(now() - tm0);

                if (merged_rows == 0)
                {
                    /// No residual probe rows for this leaf — either
                    /// it was empty all along (build had no rows
                    /// either) or mid-stream eviction already processed
                    /// every worker's slice. Either way, the HT and
                    /// store are no longer needed; release them so the
                    /// drain doesn't keep up to `total_leaves` HTs
                    /// alive simultaneously.
                    leaf_hts[L] = JoinHashTable{};
                    leaf_stores[L].reset();
                    continue;
                }

                /// Acquire-load tracks whether mid-stream eviction
                /// already built this leaf's HT (it may have done so on
                /// another worker, in which case we just probe). The
                /// fetch_add above plus the parallelRun #1 join provides
                /// the necessary happens-before so no CAS-claim is
                /// required to write `BUILT`.
                if (leaf_states[L].load(std::memory_order_acquire) == LeafState::NOT_BUILT)
                {
                    const TimePoint tb0 = now();
                    buildLeafHt(L);
                    leaf_states[L].store(LeafState::BUILT, std::memory_order_release);
                    my_build_ns += toNanos(now() - tb0);
                }

                /// Probe the merged chain end-to-end. For leaves we
                /// just built, HT + BlockStore are hot in this worker's
                /// cache (PHJ-equivalent behaviour); for leaves built
                /// mid-stream by another worker they arrive cold.
                const TimePoint tp0 = now();
                const JoinHashTable & ht = leaf_hts[L];
                const BlockStore & store = *leaf_stores[L];
                for (const auto & ob : merged.blocks)
                {
                    if (ob.rows == 0)
                        continue;
                    probeOneBlock(ob.view(), store, ht, mat, probe_hashes, heads, probe_idx, build_ref);
                }
                my_probe_ns += toNanos(now() - tp0);

                /// Releasing the merged chain destroys its OutBlocks
                /// (and their keys/payload vectors). We also free the
                /// leaf's HT and build-side store now that the drain
                /// is done with them: `ProbeMaterialiser` has already
                /// copied every matched build payload by value into
                /// the output blocks, so no dangling reference remains.
                /// This matches PHJ's stack-local HT lifetime — only
                /// the leaves currently in flight occupy memory,
                /// rather than 2048 simultaneously-allocated HTs
                /// piling up heap fragmentation across the drain.
                /// Counted as eviction overhead to match the old
                /// drain's accounting of `dropLeafProbeBuffers` calls.
                const TimePoint td0 = now();
                dropPartition(merged);
                leaf_hts[L] = JoinHashTable{};
                leaf_stores[L].reset();
                my_eviction_ns += toNanos(now() - td0);
                ++my_evictions;
            }

            mat.finish();

            /// The drain runs after parallelRun #1 has already written
            /// the mid-stream tallies; accumulate the drain's per-
            /// thread totals on top.
            ns_build[tid] += my_build_ns;
            ns_probe_shuffle[tid] += my_probe_shuffle_ns;
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
    result.bep_peak_buffered_rows = 0;
    for (size_t t = 0; t < threads; ++t)
    {
        result.bep_evictions += worker_evictions[t];
        result.bep_refinements += worker_refinements[t];
        result.bep_build_skip_retries += worker_skip_retries[t];
        result.bep_peak_buffered_rows = std::max(result.bep_peak_buffered_rows, worker_peak_rows[t]);
    }

    return result;
}

}
