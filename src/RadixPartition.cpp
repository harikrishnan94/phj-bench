#include "RadixPartition.h"

#include <array>
#include <cstring>
#include <stdexcept>

#include "Hash.h"
#include "Threading.h"


namespace phj
{

namespace
{

/// Inlined memcpy for small fixed sizes. The size dispatch lets the
/// compiler turn each branch into a single load/store pair.
[[gnu::always_inline]] inline void copyPayload(std::byte * dst, const std::byte * src, size_t size) noexcept
{
    switch (size)
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
            std::memcpy(dst, src, size);
            return;
    }
}


PartitionedShuffleOutput allocateOutput(size_t partitions, size_t threads, const PayloadSchema & schema)
{
    PartitionedShuffleOutput out;
    out.partitions = partitions;
    out.threads = threads;
    out.schema = schema;
    out.data.resize(partitions);
    for (auto & p : out.data)
    {
        p.resize(threads);
        for (auto & tpb : p)
            tpb.init(schema);
    }
    out.partition_rows.assign(partitions, 0);
    return out;
}


/// Single radix pass over flat input.
/// `parts` = number of output partitions for this pass.
/// `shift` = right shift applied to the 64-bit hash before masking.
PartitionedShuffleOutput shufflePassFlat(
    const uint64_t * keys,
    size_t rows,
    const std::vector<const PayloadColumn *> & in_payloads,
    const PayloadSchema & out_schema,
    const std::vector<size_t> & out_to_in_col,
    size_t parts,
    uint32_t shift,
    size_t threads)
{
    PartitionedShuffleOutput out = allocateOutput(parts, threads, out_schema);

    const size_t mask = parts - 1;
    const size_t n_payload = out_schema.types.size();

    parallelRun(
        threads,
        [&](size_t tid)
        {
            const size_t start = (rows * tid) / threads;
            const size_t end = (rows * (tid + 1)) / threads;

            std::vector<ThreadPartitionBuffers *> ptbs(parts);
            for (size_t p = 0; p < parts; ++p)
                ptbs[p] = &out.data[p][tid];

            std::vector<const std::byte *> src_bases(n_payload);
            std::vector<size_t> src_sizes(n_payload);
            for (size_t c = 0; c < n_payload; ++c)
            {
                src_bases[c] = in_payloads[out_to_in_col[c]]->data.get();
                src_sizes[c] = payloadTypeSize(out_schema.types[c]);
            }

            for (size_t i = start; i < end; ++i)
            {
                const uint64_t key = keys[i];
                const uint64_t h = intHash64(key);
                const size_t p = static_cast<size_t>(h >> shift) & mask;
                ThreadPartitionBuffers * tpb = ptbs[p];
                *tpb->keys.nextSlot() = key;
                for (size_t c = 0; c < n_payload; ++c)
                {
                    std::byte * dst = tpb->payloads[c].nextSlot();
                    copyPayload(dst, src_bases[c] + i * src_sizes[c], src_sizes[c]);
                }
                ++tpb->rows;
            }
        });

    for (size_t p = 0; p < parts; ++p)
    {
        size_t sum = 0;
        for (const auto & tpb : out.data[p])
            sum += tpb.rows;
        out.partition_rows[p] = sum;
    }
    return out;
}


/// Sub-radix pass over already-partitioned chained input. Each input
/// partition (from the previous pass) is processed by exactly one pass-N
/// thread; that thread emits `sub_parts` sub-partitions for each input
/// partition. Final partition index = `ip * sub_parts + sub`.
PartitionedShuffleOutput shufflePassChained(const PartitionedShuffleOutput & in, size_t sub_parts, uint32_t shift, size_t threads)
{
    const size_t out_parts = in.partitions * sub_parts;
    PartitionedShuffleOutput out = allocateOutput(out_parts, threads, in.schema);

    const size_t sub_mask = sub_parts - 1;
    const size_t n_payload = in.schema.types.size();
    std::vector<size_t> src_sizes(n_payload);
    for (size_t c = 0; c < n_payload; ++c)
        src_sizes[c] = payloadTypeSize(in.schema.types[c]);

    parallelRun(
        threads,
        [&](size_t tid)
        {
            for (size_t ip = tid; ip < in.partitions; ip += threads)
            {
                std::vector<ThreadPartitionBuffers *> sub_ptbs(sub_parts);
                for (size_t s = 0; s < sub_parts; ++s)
                    sub_ptbs[s] = &out.data[ip * sub_parts + s][tid];

                for (size_t it = 0; it < in.threads; ++it)
                {
                    const ThreadPartitionBuffers & in_tpb = in.data[ip][it];
                    const size_t total_rows = in_tpb.keys.rows;
                    if (total_rows == 0)
                        continue;

                    KeyChainReader key_cur(in_tpb.keys);
                    std::vector<PayloadChainReader> payload_curs;
                    payload_curs.reserve(n_payload);
                    for (size_t c = 0; c < n_payload; ++c)
                        payload_curs.emplace_back(in_tpb.payloads[c]);

                    for (size_t r = 0; r < total_rows; ++r)
                    {
                        const uint64_t key = key_cur.next();
                        const uint64_t h = intHash64(key);
                        const size_t s = static_cast<size_t>(h >> shift) & sub_mask;
                        ThreadPartitionBuffers * tpb = sub_ptbs[s];
                        *tpb->keys.nextSlot() = key;
                        for (size_t c = 0; c < n_payload; ++c)
                        {
                            std::byte * dst = tpb->payloads[c].nextSlot();
                            copyPayload(dst, payload_curs[c].next(), src_sizes[c]);
                        }
                        ++tpb->rows;
                    }
                }
            }
        });

    for (size_t p = 0; p < out_parts; ++p)
    {
        size_t sum = 0;
        for (const auto & tpb : out.data[p])
            sum += tpb.rows;
        out.partition_rows[p] = sum;
    }
    return out;
}

}


PartitionedShuffleOutput radixShuffle(
    const uint64_t * keys,
    size_t rows,
    const std::vector<const PayloadColumn *> & in_payloads,
    const PayloadSchema & out_schema,
    const std::vector<size_t> & out_to_in_col,
    const RadixConfig & cfg,
    size_t threads)
{
    if (cfg.pass_bits.empty())
        throw std::invalid_argument("radixShuffle: at least one pass required");
    const uint8_t total = cfg.totalBits();
    if (total == 0 || total > 32)
        throw std::invalid_argument("radixShuffle: total partition bits must be in [1, 32]");

    /// Bits are consumed from the high end of the hash, earliest pass first.
    /// After `consumed` bits have been used, the next pass takes the bits
    /// at positions [64 - consumed - bi, 64 - consumed), i.e. right-shift
    /// by `64 - consumed - bi` and mask by `(1 << bi) - 1`.
    uint8_t consumed = 0;

    const uint8_t b0 = cfg.pass_bits[0];
    const uint32_t shift0 = static_cast<uint32_t>(64u - b0);
    PartitionedShuffleOutput result = shufflePassFlat(keys, rows, in_payloads, out_schema, out_to_in_col, size_t{1} << b0, shift0, threads);
    consumed = b0;

    for (size_t pass = 1; pass < cfg.pass_bits.size(); ++pass)
    {
        const uint8_t bi = cfg.pass_bits[pass];
        const uint32_t shift = static_cast<uint32_t>(64u - consumed - bi);
        result = shufflePassChained(result, size_t{1} << bi, shift, threads);
        consumed = static_cast<uint8_t>(consumed + bi);
    }

    return result;
}

}
