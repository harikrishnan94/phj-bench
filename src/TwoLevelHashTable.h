#pragma once

#include <array>
#include <cstddef>
#include <mutex>

#include "HashTable.h"


namespace phj
{

/// CH-style two-level hashtable: 256 sub-tables, top 8 hash bits select
/// the sub-table. Each sub-table has its own mutex and resizes
/// independently. `insertLocked` is the parallel-safe insert path;
/// `find` is mutex-free (assumes the table is read-only after the
/// global build/probe barrier).
class TwoLevelJoinHashTable
{
public:
    static constexpr size_t SUB_TABLES = 256;
    static_assert((SUB_TABLES & (SUB_TABLES - 1)) == 0);

    using Hash = JoinHashTable::Hash;
    using Key = JoinHashTable::Key;
    using Ref = JoinHashTable::Ref;

    [[gnu::always_inline]] static constexpr size_t bucketOf(Hash h) noexcept { return static_cast<size_t>(h >> 56); }

    [[gnu::always_inline]] Ref insertLocked(Hash h, Key k, Ref r)
    {
        const size_t b = bucketOf(h);
        const std::scoped_lock lock(mutexes[b]);
        return subs[b].insert(h, k, r);
    }

    [[gnu::always_inline]] Ref find(Hash h, Key k) const noexcept { return subs[bucketOf(h)].find(h, k); }

    /// Batched find. Mutex-free (assumes post-barrier read-only state).
    /// Open-addressing probe sequences remain per-cell.
    void batchFind(const Hash * hashes, const Key * keys, Ref * out, size_t n) const noexcept
    {
        for (size_t i = 0; i < n; ++i)
            out[i] = find(hashes[i], keys[i]);
    }

private:
    std::array<JoinHashTable, SUB_TABLES> subs;
    std::array<std::mutex, SUB_TABLES> mutexes;
};

}
