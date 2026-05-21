#include "CHJ.h"

#include <cstring>
#include <vector>

#include "Hash.h"
#include "Threading.h"
#include "Timer.h"
#include "TwoLevelHashTable.h"
#include "Types.h"


namespace phj
{

namespace
{

[[gnu::always_inline]] inline void copySized(std::byte * dst, const std::byte * src, size_t sz) noexcept
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

}


ChjResult runCHJ(const ColumnSet & build_cs, const ColumnSet & probe_cs, size_t threads)
{
    if (threads == 0)
        threads = 1;

    ChjResult result;
    result.output.left_schema = build_cs.schema;
    result.output.right_schema = probe_cs.schema;
    result.output.workers.resize(threads);
    for (auto & w : result.output.workers)
        w.init(build_cs.schema, probe_cs.schema);

    TwoLevelJoinHashTable ht;
    std::vector<RowIndex> next_row_idx(build_cs.rows, INVALID_ROW);

    /// Pre-sizing is NOT done: the spec says CHJ does not use reserve.
    /// Sub-tables resize independently as inserts arrive.

    const size_t left_n = build_cs.schema.types.size();
    const size_t right_n = probe_cs.schema.types.size();
    std::vector<size_t> left_sizes(left_n);
    std::vector<size_t> right_sizes(right_n);
    for (size_t c = 0; c < left_n; ++c)
        left_sizes[c] = payloadTypeSize(build_cs.schema.types[c]);
    for (size_t c = 0; c < right_n; ++c)
        right_sizes[c] = payloadTypeSize(probe_cs.schema.types[c]);

    std::vector<uint64_t> build_ns(threads, 0);
    std::vector<uint64_t> probe_ns(threads, 0);

    /// -------- BUILD --------
    const TimePoint t_build0 = now();

    parallelRun(
        threads,
        [&](size_t tid)
        {
            const TimePoint t0 = now();
            const size_t start = (build_cs.rows * tid) / threads;
            const size_t end = (build_cs.rows * (tid + 1)) / threads;
            const uint64_t * keys = build_cs.keyData();
            RowIndex * next = next_row_idx.data();
            for (size_t i = start; i < end; ++i)
            {
                const uint64_t k = keys[i];
                const uint64_t h = intHash64(k);
                const RowIndex prev = ht.insertLocked(h, k, static_cast<RowIndex>(i));
                next[i] = prev;
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
            const size_t start = (probe_cs.rows * tid) / threads;
            const size_t end = (probe_cs.rows * (tid + 1)) / threads;
            const uint64_t * keys = probe_cs.keyData();
            const RowIndex * next = next_row_idx.data();

            std::vector<const std::byte *> left_bases(left_n);
            for (size_t c = 0; c < left_n; ++c)
                left_bases[c] = build_cs.payloads[c].data.get();
            std::vector<const std::byte *> right_bases(right_n);
            for (size_t c = 0; c < right_n; ++c)
                right_bases[c] = probe_cs.payloads[c].data.get();

            OutputWorker & ow = result.output.workers[tid];

            for (size_t i = start; i < end; ++i)
            {
                const uint64_t k = keys[i];
                const uint64_t h = intHash64(k);
                RowIndex r = ht.find(h, k);
                while (r != INVALID_ROW)
                {
                    *ow.keys.nextSlot() = k;
                    for (size_t c = 0; c < left_n; ++c)
                        copySized(ow.left[c].nextSlot(), left_bases[c] + static_cast<size_t>(r) * left_sizes[c], left_sizes[c]);
                    for (size_t c = 0; c < right_n; ++c)
                        copySized(ow.right[c].nextSlot(), right_bases[c] + i * right_sizes[c], right_sizes[c]);
                    r = next[r];
                }
            }
            probe_ns[tid] = toNanos(now() - t0);
        });

    const TimePoint t_probe1 = now();

    uint64_t sum_build_ns = 0;
    for (auto v : build_ns)
        sum_build_ns += v;
    uint64_t sum_probe_ns = 0;
    for (auto v : probe_ns)
        sum_probe_ns += v;

    result.build.wall_ms = toMillis(t_build1 - t_build0);
    result.build.ns_per_row = build_cs.rows == 0 ? 0.0 : static_cast<double>(sum_build_ns) / static_cast<double>(build_cs.rows);
    result.probe.wall_ms = toMillis(t_probe1 - t_probe0);
    result.probe.ns_per_row = probe_cs.rows == 0 ? 0.0 : static_cast<double>(sum_probe_ns) / static_cast<double>(probe_cs.rows);

    return result;
}

}
