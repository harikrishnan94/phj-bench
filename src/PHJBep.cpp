#include "PHJBep.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
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
/// to the probe loop downstream.
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
/// Used both by mid-stream eviction (after the per-thread leaf has
/// been chosen for drain) and by the end-of-input residual drain.
/// The per-block `probeOneBlock` is called per OutBlock; scratch
/// vectors are reused across iterations so any growth amortises
/// across the entire chain.
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


/// PHJ-BEP — IDEA 1: eager build + thread-local probe (no sync).
///
/// Departure from the baseline BEP:
///   - All leaf HTs are constructed up-front (work-stealing across
///     leaves) before any probe block is consumed. The HTs are then
///     read-only for the entire probe phase.
///   - Each worker maintains its OWN unrefined chains AND its OWN
///     per-leaf chains. No `published_leaves`, no per-leaf mutex, no
///     state machine, no cross-worker draining, no cooperative
///     end-of-slice loop.
///   - Worker-local budget: `bep_budget_mib / threads`. Same 1/4
///     unrefined / 3/4 leaf-share split with hysteresis.
///   - On unrefined high water: refine own argmax pass-1 chain into
///     own leaf chains.
///   - On leaf high water: probe own argmax leaf chain against the
///     pre-built HT, dropping the chain. No coordination with other
///     workers — they own different probe rows and so see disjoint
///     leaf chains.
///   - At end of input: refine residual unrefined → leaf chains,
///     probe each non-empty own leaf chain against its (already
///     built) HT, drop. Single parallelRun; no second pass.
///
/// Hypothesis: removing the per-leaf mutex, the LeafState CAS, the
/// global atomic counters, the argmax-over-1024-leaves scans, and
/// the cooperative-drain spin loop yields a structurally simpler
/// probe-phase with fewer pipeline stalls and no contention.
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
    const size_t pass1_bits = cfg.pass_bits.front();
    const size_t P = size_t{1} << pass1_bits;
    const size_t leaves_per_p1 = total_leaves / P;
    const uint32_t pass1_shift = static_cast<uint32_t>(64u - pass1_bits);
    const size_t bytes_per_row = sizeof(uint64_t) + probe.schema.rowByteSize();

    /// Worker-local view of the global budget. There is no sharing,
    /// so the per-worker view is `bep_budget_mib / threads` MiB
    /// strictly, split 1/4 unrefined / 3/4 leaves.
    const size_t budget_bytes = bep_budget_mib * size_t{1024} * size_t{1024};
    const size_t per_worker_view_budget = budget_bytes / threads;
    const size_t unrefined_high_water = per_worker_view_budget / 4;
    const size_t unrefined_low_water = unrefined_high_water / 2;
    const size_t leaves_high_water = (per_worker_view_budget * 3) / 4;
    const size_t leaves_low_water = (leaves_high_water * 5) / 8;

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
    std::vector<size_t> worker_refinements(threads, 0);

    std::vector<ProbeMaterialiser> mats(threads);

    /// Running maximum of the GLOBAL probe-buffer footprint observed
    /// during the scatter+evict phase, equal to the sum of every
    /// worker's own (unrefined_row_bytes + leaves_row_bytes). Updated
    /// lock-free with a CAS loop on every per-block peak bump.
    std::atomic<size_t> global_unrefined_row_bytes{0};
    std::atomic<size_t> global_leaves_row_bytes{0};
    std::atomic<size_t> global_peak_probe_bytes{0};

    /// -------- PER-THREAD PROBE LOOP (no sync) --------
    parallelRun(
        threads,
        [&](size_t tid)
        {
            /// All buffers are per-worker; nothing is shared.
            std::pmr::vector<PartitionOut> unrefined(&tracker);
            unrefined.resize(P);
            std::pmr::vector<size_t> unrefined_rows(&tracker);
            unrefined_rows.resize(P, 0);
            std::pmr::vector<PartitionOut> leaves(&tracker);
            leaves.resize(total_leaves);
            std::pmr::vector<size_t> leaf_rows(&tracker);
            leaf_rows.resize(total_leaves, 0);
            std::pmr::vector<PartitionOut> intermediate(&tracker);
            intermediate.resize(leaves_per_p1);

            for (auto & po : unrefined)
                initPartitionOut(po, probe.schema);
            for (auto & po : leaves)
                initPartitionOut(po, probe.schema);
            for (auto & po : intermediate)
                initPartitionOut(po, probe.schema);

            size_t unrefined_row_bytes = 0;
            size_t leaves_row_bytes = 0;

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
            size_t my_evictions = 0;
            size_t my_refinements = 0;

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

            /// Pick the worker's largest own pass-1 chain.
            auto argmaxUnrefined = [&]() noexcept -> size_t
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

            /// Pick the worker's largest own leaf chain.
            auto argmaxLeaf = [&]() noexcept -> size_t
            {
                size_t best_L = SIZE_MAX;
                size_t best_rows = 0;
                for (size_t L = 0; L < total_leaves; ++L)
                {
                    const size_t r = leaf_rows[L];
                    if (r > best_rows)
                    {
                        best_rows = r;
                        best_L = L;
                    }
                }
                return best_L;
            };

            /// Lock-free update of the GLOBAL probe-buffer peak.
            /// Each worker contributes its delta to the global
            /// row-byte aggregates; the peak is the running max of
            /// the sum across workers.
            auto bumpGlobalPeak = [&]() noexcept
            {
                const size_t cur = global_unrefined_row_bytes.load(std::memory_order_relaxed)
                    + global_leaves_row_bytes.load(std::memory_order_relaxed);
                size_t old_peak = global_peak_probe_bytes.load(std::memory_order_relaxed);
                while (cur > old_peak && !global_peak_probe_bytes.compare_exchange_weak(old_peak, cur, std::memory_order_relaxed))
                {
                }
            };

            /// Refine a single own pass-1 chain into the own leaves.
            /// Pure thread-local work; no synchronization.
            auto refinePartition = [&](size_t p1)
            {
                for (auto & po : intermediate)
                    initPartitionOut(po, probe.schema);

                refineToLeaves(std::move(unrefined[p1]), probe.schema, cfg.pass_bits, intermediate.data(), scatter_scratch, &tracker);

                /// `unrefined[p1]` is now in moved-from state. Re-init.
                initPartitionOut(unrefined[p1], probe.schema);
                const size_t drained_rows = unrefined_rows[p1];
                unrefined_rows[p1] = 0;
                const size_t drained_bytes = drained_rows * bytes_per_row;
                unrefined_row_bytes -= drained_bytes;
                global_unrefined_row_bytes.fetch_sub(drained_bytes, std::memory_order_relaxed);

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
                    for (auto & blk : src.blocks)
                    {
                        if (blk.rows > 0)
                            leaves[L].blocks.push_back(std::move(blk));
                    }
                    /// We only `push_back` already-allocated OutBlocks
                    /// into `leaves[L]`; nothing ever calls `grow()`
                    /// on it. Reset `cur` defensively.
                    leaves[L].cur = nullptr;
                    src.blocks.clear();
                    src.cur = nullptr;
                    leaf_rows[L] += added_rows;
                    added_rows_total += added_rows;
                }
                if (added_rows_total != 0)
                {
                    const size_t added_bytes = added_rows_total * bytes_per_row;
                    leaves_row_bytes += added_bytes;
                    global_leaves_row_bytes.fetch_add(added_bytes, std::memory_order_relaxed);
                }
            };

            /// Probe one own leaf chain against the (pre-built) HT
            /// for that leaf, then drop. Returns true on progress.
            auto evictOwnLeaf = [&]() -> bool
            {
                const TimePoint te0 = now();
                const size_t L = argmaxLeaf();
                my_eviction_ns += toNanos(now() - te0);
                if (L == SIZE_MAX)
                    return false;
                const TimePoint tp0 = now();
                probeChain(leaves[L].blocks, *leaf_stores[L], leaf_hts[L], mat, probe_hashes, heads, probe_idx, build_ref);
                my_probe_ns += toNanos(now() - tp0);

                const TimePoint td0 = now();
                const size_t evicted_rows = leaf_rows[L];
                leaf_rows[L] = 0;
                const size_t evicted_bytes = evicted_rows * bytes_per_row;
                leaves_row_bytes -= evicted_bytes;
                global_leaves_row_bytes.fetch_sub(evicted_bytes, std::memory_order_relaxed);
                dropPartition(leaves[L]);
                my_eviction_ns += toNanos(now() - td0);
                ++my_evictions;
                return true;
            };

            /// Post-input-block trigger check.
            auto evictAsNeeded = [&]()
            {
                /// Phase 1: drain own unrefined to low water.
                if (unrefined_row_bytes >= unrefined_high_water)
                {
                    while (unrefined_row_bytes >= unrefined_low_water)
                    {
                        const size_t p1 = argmaxUnrefined();
                        if (p1 == SIZE_MAX)
                            break;
                        const TimePoint trf0 = now();
                        refinePartition(p1);
                        my_probe_shuffle_ns += toNanos(now() - trf0);
                        ++my_refinements;
                    }
                }

                /// Phase 2: drain own leaves to low water (probe +
                /// drop the argmax chain). No synchronization — own
                /// leaves are disjoint from every other worker's.
                if (leaves_row_bytes >= leaves_high_water)
                {
                    while (leaves_row_bytes >= leaves_low_water)
                    {
                        if (!evictOwnLeaf())
                            break;
                    }
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
                const size_t added_bytes = added_rows * bytes_per_row;
                unrefined_row_bytes += added_bytes;
                global_unrefined_row_bytes.fetch_add(added_bytes, std::memory_order_relaxed);
                bumpGlobalPeak();
                my_probe_shuffle_ns += toNanos(now() - tps0);

                evictAsNeeded();
            }

            /// -------- END-OF-INPUT REFINEMENT + DRAIN --------
            /// Refine residual unrefined into own leaves.
            for (size_t p1 = 0; p1 < P; ++p1)
            {
                if (unrefined_rows[p1] == 0)
                    continue;
                const TimePoint trf0 = now();
                refinePartition(p1);
                my_probe_shuffle_ns += toNanos(now() - trf0);
                ++my_refinements;
            }
            /// Probe all remaining own leaf chains against their
            /// pre-built HTs.
            for (size_t L = 0; L < total_leaves; ++L)
            {
                if (leaf_rows[L] == 0)
                    continue;
                const TimePoint tp0 = now();
                probeChain(leaves[L].blocks, *leaf_stores[L], leaf_hts[L], mat, probe_hashes, heads, probe_idx, build_ref);
                my_probe_ns += toNanos(now() - tp0);
                const TimePoint td0 = now();
                const size_t bytes = leaf_rows[L] * bytes_per_row;
                leaves_row_bytes -= bytes;
                global_leaves_row_bytes.fetch_sub(bytes, std::memory_order_relaxed);
                leaf_rows[L] = 0;
                dropPartition(leaves[L]);
                my_eviction_ns += toNanos(now() - td0);
                ++my_evictions;
            }
            mat.finish();

            ns_probe_shuffle[tid] = my_probe_shuffle_ns;
            ns_probe[tid] = my_probe_ns;
            ns_eviction[tid] = my_eviction_ns;
            worker_evictions[tid] = my_evictions;
            worker_refinements[tid] = my_refinements;
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
    }
    result.bep_peak_bytes = global_peak_probe_bytes.load(std::memory_order_relaxed);

    result.peak_mem_bytes = tracker.peakBytes();

    return result;
}

}
