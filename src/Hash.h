#pragma once

#include <cstdint>


namespace phj
{

/// CH-style integer-mix hash. This is the Murmur3 finalizer twisted by a
/// constant xor at entry, as used in ClickHouse's intHash64. Both schemes
/// use this single hash function unchanged so the comparison isolates
/// partitioning strategy, not hashing.
[[gnu::always_inline]] inline uint64_t intHash64(uint64_t x) noexcept
{
    x ^= 0x4cf5ad432745937fULL;
    x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
    x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

}
