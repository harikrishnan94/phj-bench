#include "PHJ.h"

#include <algorithm>
#include <atomic>
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

/// Convert one `OutBlock` (from a partition's chain) into a `Block`
/// suitable for the build operator: same rows, same column-major
/// layout, with vectors truncated to the actual filled row count.
/// The OutBlock is consumed (its buffers are moved); the resulting
/// Block owns the storage that the partition's `BlockStore` will then
/// retain across the build/probe lifetime.
Block outBlockToBlock(OutBlock && ob)
{
    Block b;
    b.rows = ob.rows;
    b.keys = std::move(ob.keys);
    b.keys.resize(ob.rows);
    b.payloads.resize(ob.payloads.size());
    for (size_t c = 0; c < ob.payloads.size(); ++c)
    {
        b.payloads[c].type = ob.payloads[c].type;
        b.payloads[c].data = std::move(ob.payloads[c].data);
        b.payloads[c].data.resize(ob.rows * payloadTypeSize(b.payloads[c].type));
    }
    return b;
}


/// Non-owning view of an OutBlock as a BlockView. The OutBlock's
/// `keys`/`payloads` buffers may have trailing capacity beyond `rows`;
/// the view exposes only `rows` worth.
BlockView viewOf(const OutBlock & ob) noexcept
{
    return {ob.rows, ob.keys.data(), ob.payloads.data(), ob.payloads.size()};
}

}


PhjResult runPHJ(const BlockStream & build, const BlockStream & probe, const RadixConfig & cfg, size_t threads)
{
    if (threads == 0)
        threads = 1;

    PhjResult result;
    result.output.left_schema = build.schema;
    result.output.right_schema = probe.schema;
    result.output.workers.resize(threads);

    const TimePoint t_e2e0 = now();

    /// -------- BUILD SHUFFLE --------
    const TimePoint t_bs0 = now();
    PartitionedShuffleOutput build_part = radixShuffle(build, cfg, threads);
    const TimePoint t_bs1 = now();

    /// -------- PROBE SHUFFLE --------
    const TimePoint t_ps0 = now();
    PartitionedShuffleOutput probe_part = radixShuffle(probe, cfg, threads);
    const TimePoint t_ps1 = now();

    const size_t partitions = build_part.partitions;

    std::vector<uint64_t> build_ns(threads, 0);
    std::vector<uint64_t> probe_ns(threads, 0);

    std::atomic<size_t> next_partition{0};

    /// -------- BUILD + PROBE (work-stealing per partition) --------
    parallelRun(
        threads,
        [&](size_t tid)
        {
            uint64_t my_build_ns = 0;
            uint64_t my_probe_ns = 0;

            ProbeMaterialiser mat;
            mat.init(build.schema, probe.schema, result.output.workers[tid], PIPELINE_BLOCK_ROWS);

            std::vector<uint64_t> hashes;
            std::vector<uint64_t> probe_hashes;
            std::vector<RowRefCell> heads;
            std::vector<size_t> probe_idx;
            std::vector<RowRefCell> build_ref;

            while (true)
            {
                const size_t p = next_partition.fetch_add(1, std::memory_order_relaxed);
                if (p >= partitions)
                    break;

                /// ----- BUILD this partition -----
                const TimePoint tb0 = now();

                BlockStore store;
                store.reserveBlocks(build_part.chains[p].blocks.size());
                JoinHashTable ht;
                if (build_part.partition_rows[p] > 0)
                    ht.reserve(build_part.partition_rows[p]);

                /// Drain the shuffle output blocks one at a time into
                /// the partition's block store, vectorised-hashing and
                /// batched-inserting as we go.
                for (auto & ob : build_part.chains[p].blocks)
                {
                    Block as_block = outBlockToBlock(std::move(ob));
                    buildOneBlock(std::move(as_block), store, ht, hashes);
                }
                build_part.chains[p].blocks.clear();

                const TimePoint tb1 = now();
                my_build_ns += toNanos(tb1 - tb0);

                /// ----- PROBE this partition -----
                const TimePoint tp0 = now();

                if (store.numBlocks() > 0 && probe_part.partition_rows[p] > 0)
                {
                    for (const auto & ob : probe_part.chains[p].blocks)
                    {
                        if (ob.rows == 0)
                            continue;
                        probeOneBlock(viewOf(ob), store, ht, mat, probe_hashes, heads, probe_idx, build_ref);
                    }
                }

                const TimePoint tp1 = now();
                my_probe_ns += toNanos(tp1 - tp0);
            }

            mat.finish();
            build_ns[tid] = my_build_ns;
            probe_ns[tid] = my_probe_ns;
        });

    const TimePoint t_e2e1 = now();

    const uint64_t max_build_ns = build_ns.empty() ? 0 : *std::max_element(build_ns.begin(), build_ns.end());
    const uint64_t max_probe_ns = probe_ns.empty() ? 0 : *std::max_element(probe_ns.begin(), probe_ns.end());
    uint64_t sum_build_ns = 0;
    for (auto v : build_ns)
        sum_build_ns += v;
    uint64_t sum_probe_ns = 0;
    for (auto v : probe_ns)
        sum_probe_ns += v;

    result.build_shuffle.wall_ms = toMillis(t_bs1 - t_bs0);
    result.build_shuffle.ns_per_row = build.total_rows == 0
        ? 0.0
        : static_cast<double>(toNanos(t_bs1 - t_bs0)) * static_cast<double>(threads) / static_cast<double>(build.total_rows);

    result.probe_shuffle.wall_ms = toMillis(t_ps1 - t_ps0);
    result.probe_shuffle.ns_per_row = probe.total_rows == 0
        ? 0.0
        : static_cast<double>(toNanos(t_ps1 - t_ps0)) * static_cast<double>(threads) / static_cast<double>(probe.total_rows);

    result.build.wall_ms = nanosToMillis(max_build_ns);
    result.build.ns_per_row = build.total_rows == 0 ? 0.0 : static_cast<double>(sum_build_ns) / static_cast<double>(build.total_rows);

    result.probe.wall_ms = nanosToMillis(max_probe_ns);
    result.probe.ns_per_row = probe.total_rows == 0 ? 0.0 : static_cast<double>(sum_probe_ns) / static_cast<double>(probe.total_rows);

    result.e2e_wall_ms = toMillis(t_e2e1 - t_e2e0);

    return result;
}

}
