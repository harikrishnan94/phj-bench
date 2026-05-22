#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <vector>

#include "Hash.h"
#include "Types.h"


namespace phj
{

/// Open-addressing linear-probe join hashtable. Cells are 16 bytes:
/// (key, RowRefCell). Multi-match chaining is the caller's
/// responsibility via the build-side block store's next-chain matrix;
/// the HT records only the head `RowRefCell` per key. Resize uses
/// `intHash64(key)` to recompute positions; both join schemes use the
/// same hash unchanged so this is well-defined.
///
/// JoinHashTable is allocator-aware (allocator_type =
/// polymorphic_allocator<>) so that pmr containers of hash tables
/// (e.g., the per-leaf HT vector in PHJBep) propagate their resource
/// into each table's cell array at construction time.
class JoinHashTable
{
public:
    using allocator_type = std::pmr::polymorphic_allocator<>;
    using Hash = uint64_t;
    using Key = uint64_t;
    using Ref = RowRefCell;

    struct Cell
    {
        Key key;
        Ref ref;
    };
    static_assert(sizeof(Cell) == 16);

    JoinHashTable()
        : cells(std::pmr::get_default_resource())
    {
    }
    explicit JoinHashTable(std::pmr::memory_resource * mr)
        : cells(mr)
    {
    }

    /// Allocator-extended constructors used by pmr containers.
    JoinHashTable(std::allocator_arg_t, allocator_type const & alloc)
        : cells(alloc)
    {
    }

    JoinHashTable(JoinHashTable && o, allocator_type const & alloc)
        : cells(std::move(o.cells), alloc)
        , num_cells(o.num_cells)
    {
        o.num_cells = 0;
    }

    JoinHashTable(const JoinHashTable &) = delete;
    JoinHashTable & operator=(const JoinHashTable &) = delete;
    JoinHashTable(JoinHashTable &&) noexcept = default;
    JoinHashTable & operator=(JoinHashTable &&) noexcept = default;
    ~JoinHashTable() = default;

    /// Pre-size to comfortably hold `expected_rows` entries (target
    /// ~50% load).
    void reserve(size_t expected_rows)
    {
        const size_t want = expected_rows * 2;
        const size_t cap = std::bit_ceil(want);
        ensureCapacity(cap);
    }

    /// Insert `(key, ref)` keyed on the precomputed `hash`. Returns
    /// the previous head ref for `key` (INVALID_REF if this is the
    /// first insertion for `key`). The caller stores the return value
    /// into the block store's next-chain at `ref` to thread multi-
    /// match rows. Triggers grow at 50% load if needed.
    [[gnu::always_inline]] inline Ref insert(Hash hash, Key key, Ref ref)
    {
        if (num_cells * 2 >= cells.size()) [[unlikely]]
            ensureCapacity(cells.empty() ? DEFAULT_CAPACITY : cells.size() * 2);

        const size_t mask = cells.size() - 1;
        size_t pos = static_cast<size_t>(hash) & mask;
        while (true)
        {
            Cell & c = cells[pos];
            if (!c.ref.valid())
            {
                c.key = key;
                c.ref = ref;
                ++num_cells;
                return INVALID_REF;
            }
            if (c.key == key)
            {
                const Ref prev = c.ref;
                c.ref = ref;
                return prev;
            }
            pos = (pos + 1) & mask;
        }
    }

    /// Find the head ref for `key`, or INVALID_REF if not present.
    [[gnu::always_inline]] inline Ref find(Hash hash, Key key) const noexcept
    {
        if (cells.empty()) [[unlikely]]
            return INVALID_REF;
        const size_t mask = cells.size() - 1;
        size_t pos = static_cast<size_t>(hash) & mask;
        while (true)
        {
            const Cell & c = cells[pos];
            if (!c.ref.valid())
                return INVALID_REF;
            if (c.key == key)
                return c.ref;
            pos = (pos + 1) & mask;
        }
    }

    /// Batched find: for each `i` in `[0, n)`, write the head ref for
    /// `keys[i]` (keyed on `hashes[i]`) to `out[i]`. Operates one row
    /// at a time internally; the batch interface is the caller-visible
    /// dispatch granularity, not a SIMD primitive — open-addressing
    /// probe sequences remain per-cell.
    void batchFind(const Hash * hashes, const Key * keys, Ref * out, size_t n) const noexcept
    {
        for (size_t i = 0; i < n; ++i)
            out[i] = find(hashes[i], keys[i]);
    }

private:
    static constexpr size_t DEFAULT_CAPACITY = 128 * 1024;

    std::pmr::vector<Cell> cells;
    size_t num_cells = 0;

    void ensureCapacity(size_t new_cap)
    {
        if (new_cap <= cells.size())
            return;
        std::pmr::vector<Cell> old_cells = std::move(cells);
        cells.assign(new_cap, Cell{0, INVALID_REF});
        num_cells = 0;
        const size_t new_mask = new_cap - 1;
        for (Cell & oc : old_cells)
        {
            if (!oc.ref.valid())
                continue;
            const uint64_t h = intHash64(oc.key);
            size_t pos = static_cast<size_t>(h) & new_mask;
            while (cells[pos].ref.valid())
                pos = (pos + 1) & new_mask;
            cells[pos] = oc;
            ++num_cells;
        }
    }
};

}
