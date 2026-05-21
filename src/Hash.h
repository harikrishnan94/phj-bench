#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#    include <immintrin.h>
#endif


namespace phj
{

/// CH-style integer-mix hash. The Murmur3 finalizer twisted by a constant
/// xor at entry, as used in ClickHouse's `intHash64`. Both schemes use
/// this hash unchanged so the comparison isolates partitioning strategy,
/// not hashing.
[[gnu::always_inline]] inline uint64_t intHash64(uint64_t x) noexcept
{
    x ^= 0x4cf5ad432745937fULL;
    x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
    x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}


/// Vectorised batch hash. Writes `out[i] = intHash64(in[i])` for `i` in
/// `[0, n)`. The build, probe, and radix-partition operators all call
/// into this at block granularity (n is the input block's row count).
/// AVX-512F + AVX-512DQ is used when available (VPMULLQ); falls back to
/// a tight scalar loop that the compiler auto-vectorises.
inline void intHash64Batch(const uint64_t * in, size_t n, uint64_t * out) noexcept
{
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    const __m512i salt = _mm512_set1_epi64(static_cast<int64_t>(0x4cf5ad432745937fULL));
    const __m512i m1 = _mm512_set1_epi64(static_cast<int64_t>(0xff51afd7ed558ccdULL));
    const __m512i m2 = _mm512_set1_epi64(static_cast<int64_t>(0xc4ceb9fe1a85ec53ULL));

    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m512i x = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(in + i));
        x = _mm512_xor_si512(x, salt);
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 33));
        x = _mm512_mullo_epi64(x, m1);
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 33));
        x = _mm512_mullo_epi64(x, m2);
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 33));
        _mm512_storeu_si512(reinterpret_cast<__m512i *>(out + i), x);
    }
    for (; i < n; ++i)
        out[i] = intHash64(in[i]);
#else
    for (size_t i = 0; i < n; ++i)
        out[i] = intHash64(in[i]);
#endif
}

}
