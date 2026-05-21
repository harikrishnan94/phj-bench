#include "DataGen.h"

#include <cstring>

#include "Hash.h"
#include "Threading.h"


namespace phj
{

namespace
{

constexpr uint64_t BUILD_KEY_SALT = 0x6f1bda28e54e2bd1ULL;
constexpr uint64_t PROBE_PICK_SALT = 0xc52f01f0e0c46cd3ULL;
constexpr uint64_t BUILD_PAYLOAD_SALT = 0xf3e9aa18b6517527ULL;
constexpr uint64_t PROBE_PAYLOAD_SALT = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t COLUMN_SALT_STRIDE = 0x9ddfea08eb382d69ULL;

[[gnu::always_inline]] inline void writePayloadValue(std::byte * dst, PayloadType type, uint64_t lo, uint64_t hi) noexcept
{
    switch (type)
    {
        case PayloadType::UInt8: {
            auto v = static_cast<uint8_t>(lo);
            std::memcpy(dst, &v, sizeof(v));
            return;
        }
        case PayloadType::UInt16: {
            auto v = static_cast<uint16_t>(lo);
            std::memcpy(dst, &v, sizeof(v));
            return;
        }
        case PayloadType::UInt32: {
            auto v = static_cast<uint32_t>(lo);
            std::memcpy(dst, &v, sizeof(v));
            return;
        }
        case PayloadType::UInt64: {
            std::memcpy(dst, &lo, sizeof(lo));
            return;
        }
        case PayloadType::UInt128: {
            std::memcpy(dst, &lo, sizeof(lo));
            std::memcpy(dst + sizeof(lo), &hi, sizeof(hi));
            return;
        }
    }
}


void allocateStream(BlockStream & out, size_t rows, const PayloadSchema & schema, size_t block_rows)
{
    out.schema = schema;
    out.total_rows = rows;
    if (rows == 0)
        return;
    const size_t n_blocks = (rows + block_rows - 1) / block_rows;
    out.blocks.resize(n_blocks);
    for (size_t b = 0; b < n_blocks; ++b)
    {
        const size_t start = b * block_rows;
        const size_t end = std::min(start + block_rows, rows);
        out.blocks[b].init(schema, end - start);
    }
}


/// Given a flat row index `i` within a stream, return the (block, row)
/// pair via integer division. Block rows are constant except the last.
struct BlockCoord
{
    size_t block;
    size_t row;
};

[[gnu::always_inline]] inline BlockCoord coordOf(size_t i, size_t block_rows) noexcept
{
    return {i / block_rows, i % block_rows};
}


void fillBuildSlice(BlockStream & stream, size_t start, size_t end, uint64_t seed, size_t block_rows)
{
    const uint64_t key_salt = seed ^ BUILD_KEY_SALT;
    const uint64_t pay_salt = seed ^ BUILD_PAYLOAD_SALT;
    const size_t n_payload = stream.schema.types.size();

    for (size_t i = start; i < end; ++i)
    {
        const auto [b, r] = coordOf(i, block_rows);
        Block & blk = stream.blocks[b];
        blk.keys[r] = intHash64(key_salt + i);
        for (size_t c = 0; c < n_payload; ++c)
        {
            const PayloadType type = stream.schema.types[c];
            const size_t element_size = payloadTypeSize(type);
            const uint64_t col_salt = pay_salt ^ (COLUMN_SALT_STRIDE * (c + 1));
            const uint64_t lo = intHash64(col_salt ^ i);
            const uint64_t hi = (type == PayloadType::UInt128) ? intHash64(col_salt ^ (i + (1ULL << 33))) : 0;
            writePayloadValue(blk.payloads[c].raw() + r * element_size, type, lo, hi);
        }
    }
}


[[gnu::always_inline]] inline uint64_t buildKeyFromStream(const BlockStream & stream, size_t flat_row, size_t block_rows) noexcept
{
    const auto [b, r] = coordOf(flat_row, block_rows);
    return stream.blocks[b].keyData()[r];
}


void fillProbeSlice(BlockStream & stream, const BlockStream & build_stream, size_t start, size_t end, uint64_t seed, size_t block_rows)
{
    const uint64_t pick_salt = seed ^ PROBE_PICK_SALT;
    const uint64_t pay_salt = seed ^ PROBE_PAYLOAD_SALT;
    const size_t build_rows = build_stream.total_rows;
    const size_t n_payload = stream.schema.types.size();

    for (size_t i = start; i < end; ++i)
    {
        const auto [b, r] = coordOf(i, block_rows);
        Block & blk = stream.blocks[b];
        const uint64_t pick = intHash64(pick_salt + i);
        const size_t bi = static_cast<size_t>(pick % build_rows);
        blk.keys[r] = buildKeyFromStream(build_stream, bi, block_rows);
        for (size_t c = 0; c < n_payload; ++c)
        {
            const PayloadType type = stream.schema.types[c];
            const size_t element_size = payloadTypeSize(type);
            const uint64_t col_salt = pay_salt ^ (COLUMN_SALT_STRIDE * (c + 1));
            const uint64_t lo = intHash64(col_salt ^ i);
            const uint64_t hi = (type == PayloadType::UInt128) ? intHash64(col_salt ^ (i + (1ULL << 33))) : 0;
            writePayloadValue(blk.payloads[c].raw() + r * element_size, type, lo, hi);
        }
    }
}

}


BlockStream generateBuild(size_t rows, const PayloadSchema & schema, uint64_t seed, size_t threads, size_t block_rows)
{
    BlockStream out;
    allocateStream(out, rows, schema, block_rows);
    if (rows == 0)
        return out;

    parallelRun(
        threads,
        [&](size_t tid)
        {
            const size_t start = (rows * tid) / threads;
            const size_t end = (rows * (tid + 1)) / threads;
            fillBuildSlice(out, start, end, seed, block_rows);
        });
    return out;
}


BlockStream
generateProbe(size_t rows, const PayloadSchema & schema, uint64_t seed, const BlockStream & build_stream, size_t threads, size_t block_rows)
{
    BlockStream out;
    allocateStream(out, rows, schema, block_rows);
    if (rows == 0 || build_stream.total_rows == 0)
        return out;

    parallelRun(
        threads,
        [&](size_t tid)
        {
            const size_t start = (rows * tid) / threads;
            const size_t end = (rows * (tid + 1)) / threads;
            fillProbeSlice(out, build_stream, start, end, seed, block_rows);
        });
    return out;
}

}
