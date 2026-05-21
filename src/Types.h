#pragma once

#include <cstddef>
#include <cstdint>


namespace phj
{

using RowIndex = uint32_t;
inline constexpr RowIndex INVALID_ROW = ~RowIndex{0};


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


/// Reference into externally-owned column-major payload storage.
/// Stored alongside the key in the hashtable cell. Holds the head row
/// index of the multi-match chain; the next-row pointers live in an
/// external array.
struct RowRefCell
{
    RowIndex row_idx = INVALID_ROW;
};
static_assert(sizeof(RowRefCell) == 4);

}
