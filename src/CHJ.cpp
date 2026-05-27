#include "CHJ.h"

#include <algorithm>
#include <vector>

#include "JoinOps.h"
#include "MemTracker.h"
#include "Threading.h"
#include "Timer.h"
#include "TwoLevelHashTable.h"


namespace phj
{

ChjResult runCHJ(const BlockStream & build, const BlockStream & probe, size_t threads)
{
    if (threads == 0)
        threads = 1;

    MemTracker tracker;

    ChjResult result;
    result.output.left_schema = build.schema;
    result.output.right_schema = probe.schema;
    result.output.workers.resize(threads);

    /// CH-style: min(threads, 256) slots rounded to a power-of-2.
    /// Each slot gets its own BlockStore and JoinHashTable; no shared
    /// data structure exists during the build phase.
    const size_t n_slots = std::min(threads, ChjSlottedTable::MAX_SLOTS);
    ChjSlottedTable table(n_slots, &tracker);

    /// Pre-reserve each slot's block-vector capacity. Every input block
    /// scatters at most one sub-block per slot, so `build.blocks.size()`
    /// is a tight upper bound on the number of blocks per slot.
    table.reserveSlotBlocks(build.blocks.size());

    std::vector<uint64_t> build_ns(threads, 0);
    std::vector<uint64_t> probe_ns(threads, 0);

    const TimePoint t_e2e0 = now();

    /// -------- BUILD --------
    /// Each worker holds a slice of the input build blocks and scatters
    /// each block's rows across slots by hash. A slot's mutex is held
    /// only for the duration of one sub-block insert; workers that hash
    /// to different slots proceed in parallel.
    const TimePoint t_build0 = now();

    parallelRun(
        threads,
        [&](size_t tid)
        {
            const TimePoint t0 = now();
            const size_t n = build.blocks.size();
            const size_t start = (n * tid) / threads;
            const size_t end = (n * (tid + 1)) / threads;

            std::pmr::vector<uint64_t> hashes(&tracker);
            std::vector<std::vector<size_t>> slot_rows(table.numSlots());

            for (size_t b = start; b < end; ++b)
                buildOneBlockCHJ(build.blocks[b], table, hashes, slot_rows, &tracker);

            build_ns[tid] = toNanos(now() - t0);
        });

    const TimePoint t_build1 = now();

    /// -------- PROBE --------
    /// After the build barrier the table is read-only. Each worker scatters
    /// its probe blocks by the same hash bits and queries each slot's
    /// JoinHashTable without any locking.
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

            std::pmr::vector<uint64_t> hashes(&tracker);
            std::pmr::vector<size_t> probe_idx(&tracker);
            std::pmr::vector<RowRefCell> build_ref(&tracker);
            std::vector<std::vector<size_t>> slot_rows(table.numSlots());

            for (size_t b = start; b < end; ++b)
                probeOneBlockCHJ(probe.blocks[b].view(), table, mat, hashes, probe_idx, build_ref, slot_rows);

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
    result.peak_mem_bytes = tracker.peakBytes();

    return result;
}

}
