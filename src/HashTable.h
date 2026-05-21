#pragma once

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "Hash.h"
#include "Types.h"


namespace phj
{

/// Open-addressing linear-probe join hashtable. Cells are 16 bytes:
/// (key, RowRefCell). Multi-match chaining is the caller's responsibility
/// via an external `next_row_idx` array; the HT records only the head row
/// per key. Resize uses `intHash64(key)` to recompute positions; both join
/// schemes use the same hash unchanged so this is well-defined.
class JoinHashTable
{
public:
    using Hash = uint64_t;
    using Key = uint64_t;
    using Row = RowIndex;

    struct Cell
    {
        Key key;
        RowRefCell ref;
        uint32_t pad;
    };
    static_assert(sizeof(Cell) == 16, "Cell expected to be 16 bytes");

    JoinHashTable() = default;
    JoinHashTable(const JoinHashTable &) = delete;
    JoinHashTable & operator=(const JoinHashTable &) = delete;
    JoinHashTable(JoinHashTable &&) noexcept = default;
    JoinHashTable & operator=(JoinHashTable &&) noexcept = default;
    ~JoinHashTable() = default;

    /// Pre-size to comfortably hold `expected_rows` entries (target ~50% load).
    void reserve(size_t expected_rows)
    {
        const size_t want = std::max<size_t>(MIN_CAPACITY, expected_rows * 2);
        const size_t cap = std::bit_ceil(want);
        ensureCapacity(cap);
    }

    /// Insert `key` with row index `row_idx`. Returns the previous head row
    /// for `key` (INVALID_ROW if this is the first insertion for `key`).
    /// The caller stores the return value into next_row_idx[row_idx] to
    /// chain multi-match rows. Triggers grow at 50% load if needed.
    [[gnu::always_inline]] inline Row insert(Hash hash, Key key, Row row_idx)
    {
        if (num_cells * 2 >= cells.size()) [[unlikely]]
            ensureCapacity(cells.empty() ? MIN_CAPACITY : cells.size() * 2);

        const size_t mask = cells.size() - 1;
        size_t pos = static_cast<size_t>(hash) & mask;
        while (true)
        {
            Cell & c = cells[pos];
            if (c.ref.row_idx == INVALID_ROW)
            {
                c.key = key;
                c.ref.row_idx = row_idx;
                ++num_cells;
                return INVALID_ROW;
            }
            if (c.key == key)
            {
                const Row prev = c.ref.row_idx;
                c.ref.row_idx = row_idx;
                return prev;
            }
            pos = (pos + 1) & mask;
        }
    }

    /// Find the head row index for `key`, or INVALID_ROW if not present.
    [[gnu::always_inline]] inline Row find(Hash hash, Key key) const noexcept
    {
        if (cells.empty()) [[unlikely]]
            return INVALID_ROW;
        const size_t mask = cells.size() - 1;
        size_t pos = static_cast<size_t>(hash) & mask;
        while (true)
        {
            const Cell & c = cells[pos];
            if (c.ref.row_idx == INVALID_ROW)
                return INVALID_ROW;
            if (c.key == key)
                return c.ref.row_idx;
            pos = (pos + 1) & mask;
        }
    }

    [[nodiscard]] size_t size() const noexcept { return num_cells; }
    [[nodiscard]] size_t capacity() const noexcept { return cells.size(); }

private:
    static constexpr size_t MIN_CAPACITY = 256 * 1024;

    std::vector<Cell> cells;
    size_t num_cells = 0;

    void ensureCapacity(size_t new_cap)
    {
        if (new_cap <= cells.size())
            return;
        std::vector<Cell> old_cells = std::move(cells);
        cells.assign(new_cap, Cell{0, RowRefCell{INVALID_ROW}, 0});
        num_cells = 0;
        const size_t new_mask = new_cap - 1;
        for (Cell & oc : old_cells)
        {
            if (oc.ref.row_idx == INVALID_ROW)
                continue;
            const uint64_t h = intHash64(oc.key);
            size_t pos = static_cast<size_t>(h) & new_mask;
            while (cells[pos].ref.row_idx != INVALID_ROW)
                pos = (pos + 1) & new_mask;
            cells[pos] = oc;
            ++num_cells;
        }
    }
};

}
