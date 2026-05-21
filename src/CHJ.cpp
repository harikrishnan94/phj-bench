#include "CHJ.h"

#include <vector>

#include "BlockStore.h"
#include "JoinOps.h"
#include "Threading.h"
#include "Timer.h"
#include "TwoLevelHashTable.h"


namespace phj
{

ChjResult runCHJ(const BlockStream & build, const BlockStream & probe, size_t threads)
{
    if (threads == 0)
        threads = 1;

    ChjResult result;
    result.output.left_schema = build.schema;
    result.output.right_schema = probe.schema;
    result.output.workers.resize(threads);

    BlockStore store;
    store.reserveBlocks(build.blocks.size());
    TwoLevelJoinHashTable ht;

    std::vector<uint64_t> build_ns(threads, 0);
    std::vector<uint64_t> probe_ns(threads, 0);

    const TimePoint t_e2e0 = now();

    /// -------- BUILD --------
    const TimePoint t_build0 = now();

    parallelRun(
        threads,
        [&](size_t tid)
        {
            const TimePoint t0 = now();
            const size_t n = build.blocks.size();
            const size_t start = (n * tid) / threads;
            const size_t end = (n * (tid + 1)) / threads;
            std::vector<uint64_t> hashes;
            for (size_t b = start; b < end; ++b)
            {
                /// We need a mutable copy to move into the store. The
                /// underlying source `build.blocks` stays intact across
                /// reps (and across the optional --check pass), so we
                /// deep-copy at block granularity here. This is the
                /// "append the block to the build-side block store"
                /// step in the spec; the build store ultimately owns
                /// the block's storage.
                Block clone = build.blocks[b];
                buildOneBlock(std::move(clone), store, ht, hashes);
            }
            build_ns[tid] = toNanos(now() - t0);
        });

    const TimePoint t_build1 = now();

    /// -------- PROBE --------
    const TimePoint t_probe0 = now();

    parallelRun(
        threads,
        [&](size_t tid)
        {
            const TimePoint t0 = now();
            const size_t n = probe.blocks.size();
            const size_t start = (n * tid) / threads;
            const size_t end = (n * (tid + 1)) / threads;

            ProbeMaterialiser mat;
            mat.init(build.schema, probe.schema, result.output.workers[tid], PIPELINE_BLOCK_ROWS);

            std::vector<uint64_t> hashes;
            std::vector<RowRefCell> heads;
            std::vector<size_t> probe_idx;
            std::vector<RowRefCell> build_ref;

            for (size_t b = start; b < end; ++b)
                probeOneBlock(probe.blocks[b].view(), store, ht, mat, hashes, heads, probe_idx, build_ref);

            mat.finish();
            probe_ns[tid] = toNanos(now() - t0);
        });

    const TimePoint t_probe1 = now();
    const TimePoint t_e2e1 = t_probe1;

    uint64_t sum_build_ns = 0;
    for (auto v : build_ns)
        sum_build_ns += v;
    uint64_t sum_probe_ns = 0;
    for (auto v : probe_ns)
        sum_probe_ns += v;

    result.build.wall_ms = toMillis(t_build1 - t_build0);
    result.build.ns_per_row = build.total_rows == 0 ? 0.0 : static_cast<double>(sum_build_ns) / static_cast<double>(build.total_rows);
    result.probe.wall_ms = toMillis(t_probe1 - t_probe0);
    result.probe.ns_per_row = probe.total_rows == 0 ? 0.0 : static_cast<double>(sum_probe_ns) / static_cast<double>(probe.total_rows);
    result.e2e_wall_ms = toMillis(t_e2e1 - t_e2e0);

    return result;
}

}
