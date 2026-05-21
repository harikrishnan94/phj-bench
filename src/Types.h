#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>


namespace phj
{

using BlockNo = uint32_t;
using RowNo = uint32_t;

inline constexpr BlockNo INVALID_BLOCK = ~BlockNo{0};


struct UInt128
{
    uint64_t low;
    uint64_t high;

    friend bool operator==(const UInt128 &, const UInt128 &) noexcept = default;
};
static_assert(sizeof(UInt128) == 16);


enum class PayloadType : uint8_t
{
    UInt8 = 0,
    UInt16 = 1,
    UInt32 = 2,
    UInt64 = 3,
    UInt128 = 4,
};

inline constexpr size_t payloadTypeSize(PayloadType t) noexcept
{
    switch (t)
    {
        case PayloadType::UInt8:
            return 1;
        case PayloadType::UInt16:
            return 2;
        case PayloadType::UInt32:
            return 4;
        case PayloadType::UInt64:
            return 8;
        case PayloadType::UInt128:
            return 16;
    }
    return 0;
}

inline constexpr const char * payloadTypeName(PayloadType t) noexcept
{
    switch (t)
    {
        case PayloadType::UInt8:
            return "u8";
        case PayloadType::UInt16:
            return "u16";
        case PayloadType::UInt32:
            return "u32";
        case PayloadType::UInt64:
            return "u64";
        case PayloadType::UInt128:
            return "u128";
    }
    return "?";
}


struct PayloadSchema
{
    std::vector<PayloadType> types;

    [[nodiscard]] size_t rowByteSize() const noexcept
    {
        size_t s = 0;
        for (auto t : types)
            s += payloadTypeSize(t);
        return s;
    }
};


/// Position of a build-side row in the build-side block store.
/// Cells in the hashtable hold one of these alongside the key. `block_no`
/// indexes into the partition's (PHJ) or the shared (CHJ) Vector<Block>
/// and `row_no` is the within-block row index. An empty cell is signalled
/// by `block_no == INVALID_BLOCK`.
struct RowRefCell
{
    BlockNo block_no = INVALID_BLOCK;
    RowNo row_no = 0;

    [[nodiscard]] bool valid() const noexcept { return block_no != INVALID_BLOCK; }
};
static_assert(sizeof(RowRefCell) == 8);

inline constexpr RowRefCell INVALID_REF{INVALID_BLOCK, 0};


/// Pipeline block size in rows. The data generator emits blocks of this
/// size; the radix partition, build, and probe operators consume input
/// blocks of this size as their batch unit; the probe operator emits
/// output blocks of (up to) this size.
inline constexpr size_t PIPELINE_BLOCK_ROWS = 10'000;

}
