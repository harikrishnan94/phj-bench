#pragma once

#include <array>
#include <cstddef>
#include <mutex>

#include "HashTable.h"


namespace phj
{

/// CH-style two-level hashtable: 256 sub-tables, top 8 hash bits select the
/// sub-table. Each sub-table has its own mutex and resizes independently.
/// `insertLocked` is the parallel-safe insert path; `find` is mutex-free
/// (assumes the table is read-only after the global build/probe barrier).
class TwoLevelJoinHashTable
{
public:
    static constexpr size_t SUB_TABLES = 256;
    static_assert((SUB_TABLES & (SUB_TABLES - 1)) == 0);

    using Hash = JoinHashTable::Hash;
    using Key = JoinHashTable::Key;
    using Row = JoinHashTable::Row;

    [[gnu::always_inline]] static constexpr size_t bucketOf(Hash h) noexcept { return static_cast<size_t>(h >> 56); }

    [[gnu::always_inline]] Row insertLocked(Hash h, Key k, Row r)
    {
        const size_t b = bucketOf(h);
        const std::scoped_lock lock(mutexes[b]);
        return subs[b].insert(h, k, r);
    }

    [[gnu::always_inline]] Row find(Hash h, Key k) const noexcept { return subs[bucketOf(h)].find(h, k); }

    JoinHashTable & sub(size_t i) noexcept { return subs[i]; }
    const JoinHashTable & sub(size_t i) const noexcept { return subs[i]; }

private:
    std::array<JoinHashTable, SUB_TABLES> subs;
    std::array<std::mutex, SUB_TABLES> mutexes;
};

}
