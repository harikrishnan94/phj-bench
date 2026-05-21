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

void fillPayloads(ColumnSet & cs, size_t start, size_t end, uint64_t base_salt, uint64_t row_salt)
{
    const size_t n = cs.schema.types.size();
    for (size_t c = 0; c < n; ++c)
    {
        const PayloadType type = cs.schema.types[c];
        const size_t element_size = payloadTypeSize(type);
        const uint64_t col_salt = base_salt ^ (COLUMN_SALT_STRIDE * (c + 1));
        std::byte * data = cs.payloads[c].data.get();
        for (size_t i = start; i < end; ++i)
        {
            const uint64_t lo = intHash64(col_salt ^ (row_salt + i));
            const uint64_t hi = (type == PayloadType::UInt128) ? intHash64(col_salt ^ (row_salt + i + (1ULL << 33))) : 0;
            writePayloadValue(data + i * element_size, type, lo, hi);
        }
    }
}

}


ColumnSet generateBuild(size_t rows, const PayloadSchema & schema, uint64_t seed, size_t threads)
{
    ColumnSet cs = ColumnSet::allocate(rows, schema);
    parallelRun(
        threads,
        [&](size_t tid)
        {
            const size_t start = (rows * tid) / threads;
            const size_t end = (rows * (tid + 1)) / threads;
            uint64_t * keys = cs.keys.get();
            const uint64_t key_salt = seed ^ BUILD_KEY_SALT;
            for (size_t i = start; i < end; ++i)
                keys[i] = intHash64(key_salt + i);
            fillPayloads(cs, start, end, seed ^ BUILD_PAYLOAD_SALT, 0);
        });
    return cs;
}


ColumnSet
generateProbe(size_t rows, const PayloadSchema & schema, uint64_t seed, const uint64_t * build_keys, size_t build_rows, size_t threads)
{
    ColumnSet cs = ColumnSet::allocate(rows, schema);
    if (rows == 0 || build_rows == 0)
        return cs;
    parallelRun(
        threads,
        [&](size_t tid)
        {
            const size_t start = (rows * tid) / threads;
            const size_t end = (rows * (tid + 1)) / threads;
            uint64_t * keys = cs.keys.get();
            const uint64_t pick_salt = seed ^ PROBE_PICK_SALT;
            for (size_t i = start; i < end; ++i)
            {
                const uint64_t pick = intHash64(pick_salt + i);
                const size_t bi = static_cast<size_t>(pick % build_rows);
                keys[i] = build_keys[bi];
            }
            fillPayloads(cs, start, end, seed ^ PROBE_PAYLOAD_SALT, 0);
        });
    return cs;
}

}
