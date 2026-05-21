#include "PHJ.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <vector>

#include "Hash.h"
#include "HashTable.h"
#include "Threading.h"
#include "Timer.h"
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


std::vector<const PayloadColumn *> payloadPtrs(const ColumnSet & cs)
{
    std::vector<const PayloadColumn *> v;
    v.reserve(cs.payloads.size());
    for (const auto & p : cs.payloads)
        v.push_back(&p);
    return v;
}


std::vector<size_t> identityMap(size_t n)
{
    std::vector<size_t> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = i;
    return v;
}


/// Build-side compact representation of one partition: keys + per-column
/// payload buffers, all flat and column-major. The row index returned by
/// the partition's HT is an offset into these flat buffers.
struct CompactPartition
{
    size_t rows = 0;
    std::unique_ptr<uint64_t[]> keys;
    std::vector<std::unique_ptr<std::byte[]>> payloads;
    std::vector<size_t> sizes;
};


CompactPartition compactBuildPartition(const PartitionedShuffleOutput & shuffled, size_t partition_idx)
{
    CompactPartition out;
    const size_t rows = shuffled.partition_rows[partition_idx];
    out.rows = rows;
    const size_t n_payload = shuffled.schema.types.size();

    out.keys = std::make_unique<uint64_t[]>(rows == 0 ? 1 : rows);
    out.payloads.resize(n_payload);
    out.sizes.resize(n_payload);
    for (size_t c = 0; c < n_payload; ++c)
    {
        out.sizes[c] = payloadTypeSize(shuffled.schema.types[c]);
        out.payloads[c] = std::make_unique<std::byte[]>(rows == 0 ? out.sizes[c] : rows * out.sizes[c]);
    }

    size_t cursor = 0;
    for (size_t t = 0; t < shuffled.threads; ++t)
    {
        const ThreadPartitionBuffers & tpb = shuffled.data[partition_idx][t];
        const size_t t_rows = tpb.keys.rows;
        if (t_rows == 0)
            continue;

        KeyChainReader key_cur(tpb.keys);
        std::vector<PayloadChainReader> payload_curs;
        payload_curs.reserve(n_payload);
        for (size_t c = 0; c < n_payload; ++c)
            payload_curs.emplace_back(tpb.payloads[c]);

        for (size_t r = 0; r < t_rows; ++r)
        {
            out.keys.get()[cursor + r] = key_cur.next();
            for (size_t c = 0; c < n_payload; ++c)
            {
                const size_t sz = out.sizes[c];
                std::memcpy(out.payloads[c].get() + (cursor + r) * sz, payload_curs[c].next(), sz);
            }
        }
        cursor += t_rows;
    }
    return out;
}

}


PhjResult runPHJ(const ColumnSet & build_cs, const ColumnSet & probe_cs, const RadixConfig & cfg, size_t threads)
{
    if (threads == 0)
        threads = 1;

    PhjResult result;
    result.output.left_schema = build_cs.schema;
    result.output.right_schema = probe_cs.schema;
    result.output.workers.resize(threads);
    for (auto & w : result.output.workers)
        w.init(build_cs.schema, probe_cs.schema);

    const std::vector<const PayloadColumn *> build_payload_ptrs = payloadPtrs(build_cs);
    const std::vector<const PayloadColumn *> probe_payload_ptrs = payloadPtrs(probe_cs);
    const std::vector<size_t> build_id = identityMap(build_cs.schema.types.size());
    const std::vector<size_t> probe_id = identityMap(probe_cs.schema.types.size());

    /// -------- BUILD SHUFFLE --------
    const TimePoint t_bs0 = now();
    PartitionedShuffleOutput build_part
        = radixShuffle(build_cs.keyData(), build_cs.rows, build_payload_ptrs, build_cs.schema, build_id, cfg, threads);
    const TimePoint t_bs1 = now();

    /// -------- PROBE SHUFFLE --------
    const TimePoint t_ps0 = now();
    PartitionedShuffleOutput probe_part
        = radixShuffle(probe_cs.keyData(), probe_cs.rows, probe_payload_ptrs, probe_cs.schema, probe_id, cfg, threads);
    const TimePoint t_ps1 = now();

    const size_t partitions = build_part.partitions;
    const size_t left_n = build_cs.schema.types.size();
    const size_t right_n = probe_cs.schema.types.size();
    std::vector<size_t> right_sizes(right_n);
    for (size_t c = 0; c < right_n; ++c)
        right_sizes[c] = payloadTypeSize(probe_cs.schema.types[c]);

    std::vector<uint64_t> build_ns(threads, 0);
    std::vector<uint64_t> probe_ns(threads, 0);

    std::atomic<size_t> next_partition{0};

    /// -------- BUILD + PROBE (work-stealing per partition) --------
    parallelRun(
        threads,
        [&](size_t tid)
        {
            OutputWorker & ow = result.output.workers[tid];
            uint64_t my_build_ns = 0;
            uint64_t my_probe_ns = 0;

            while (true)
            {
                const size_t p = next_partition.fetch_add(1, std::memory_order_relaxed);
                if (p >= partitions)
                    break;

                /// ----- BUILD this partition -----
                const TimePoint tb0 = now();

                CompactPartition build_p = compactBuildPartition(build_part, p);
                const size_t b_rows = build_p.rows;

                JoinHashTable ht;
                std::vector<RowIndex> next_row(b_rows, INVALID_ROW);
                if (b_rows > 0)
                {
                    ht.reserve(b_rows);
                    const uint64_t * keys = build_p.keys.get();
                    for (size_t i = 0; i < b_rows; ++i)
                    {
                        const uint64_t k = keys[i];
                        const uint64_t h = intHash64(k);
                        const RowIndex prev = ht.insert(h, k, static_cast<RowIndex>(i));
                        next_row[i] = prev;
                    }
                }

                const TimePoint tb1 = now();
                my_build_ns += toNanos(tb1 - tb0);

                /// ----- PROBE this partition -----
                const TimePoint tp0 = now();

                if (b_rows > 0 && probe_part.partition_rows[p] > 0)
                {
                    std::vector<const std::byte *> left_bases(left_n);
                    std::vector<size_t> left_sizes(left_n);
                    for (size_t c = 0; c < left_n; ++c)
                    {
                        left_bases[c] = build_p.payloads[c].get();
                        left_sizes[c] = build_p.sizes[c];
                    }

                    for (size_t t = 0; t < probe_part.threads; ++t)
                    {
                        const ThreadPartitionBuffers & tpb = probe_part.data[p][t];
                        const size_t pt_rows = tpb.keys.rows;
                        if (pt_rows == 0)
                            continue;

                        KeyChainReader key_cur(tpb.keys);
                        std::vector<PayloadChainReader> probe_payload_curs;
                        probe_payload_curs.reserve(right_n);
                        for (size_t c = 0; c < right_n; ++c)
                            probe_payload_curs.emplace_back(tpb.payloads[c]);

                        for (size_t r = 0; r < pt_rows; ++r)
                        {
                            const uint64_t k = key_cur.next();
                            const uint64_t h = intHash64(k);

                            /// Snapshot probe payload pointers for this row exactly once
                            /// (multi-match emits multiple output rows referencing the
                            /// same probe row).
                            std::array<const std::byte *, 32> probe_row_payloads{};
                            for (size_t c = 0; c < right_n; ++c)
                                probe_row_payloads[c] = probe_payload_curs[c].next();

                            RowIndex match = ht.find(h, k);
                            while (match != INVALID_ROW)
                            {
                                *ow.keys.nextSlot() = k;
                                for (size_t c = 0; c < left_n; ++c)
                                    copySized(
                                        ow.left[c].nextSlot(), left_bases[c] + static_cast<size_t>(match) * left_sizes[c], left_sizes[c]);
                                for (size_t c = 0; c < right_n; ++c)
                                    copySized(ow.right[c].nextSlot(), probe_row_payloads[c], right_sizes[c]);
                                match = next_row[match];
                            }
                        }
                    }
                }

                const TimePoint tp1 = now();
                my_probe_ns += toNanos(tp1 - tp0);
            }

            build_ns[tid] = my_build_ns;
            probe_ns[tid] = my_probe_ns;
        });

    const uint64_t max_build_ns = *std::max_element(build_ns.begin(), build_ns.end());
    const uint64_t max_probe_ns = *std::max_element(probe_ns.begin(), probe_ns.end());
    uint64_t sum_build_ns = 0;
    for (auto v : build_ns)
        sum_build_ns += v;
    uint64_t sum_probe_ns = 0;
    for (auto v : probe_ns)
        sum_probe_ns += v;

    result.build_shuffle.wall_ms = toMillis(t_bs1 - t_bs0);
    result.build_shuffle.ns_per_row = build_cs.rows == 0
        ? 0.0
        : static_cast<double>(toNanos(t_bs1 - t_bs0)) * static_cast<double>(threads) / static_cast<double>(build_cs.rows);

    result.probe_shuffle.wall_ms = toMillis(t_ps1 - t_ps0);
    result.probe_shuffle.ns_per_row = probe_cs.rows == 0
        ? 0.0
        : static_cast<double>(toNanos(t_ps1 - t_ps0)) * static_cast<double>(threads) / static_cast<double>(probe_cs.rows);

    result.build.wall_ms = nanosToMillis(max_build_ns);
    result.build.ns_per_row = build_cs.rows == 0 ? 0.0 : static_cast<double>(sum_build_ns) / static_cast<double>(build_cs.rows);

    result.probe.wall_ms = nanosToMillis(max_probe_ns);
    result.probe.ns_per_row = probe_cs.rows == 0 ? 0.0 : static_cast<double>(sum_probe_ns) / static_cast<double>(probe_cs.rows);

    return result;
}

}
