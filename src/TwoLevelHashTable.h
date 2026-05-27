#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <vector>

#include "BlockStore.h"
#include "HashTable.h"


namespace phj
{

/// One independent build-side partition in a CH-style ConcurrentHashJoin.
///
/// Each slot owns its own BlockStore and JoinHashTable. The mutex serialises
/// concurrent build inserts from multiple workers that scatter their input
/// blocks into this slot; only one writer is active per slot at a time so
/// the inner JoinHashTable never needs per-cell locking.
struct ChjSlot
{
    std::mutex mu;
    BlockStore store;
    JoinHashTable ht;

    explicit ChjSlot(std::pmr::memory_resource * mr)
        : store(mr)
        , ht(mr)
    {
    }

    ChjSlot(const ChjSlot &) = delete;
    ChjSlot & operator=(const ChjSlot &) = delete;
    ChjSlot(ChjSlot &&) = delete;
    ChjSlot & operator=(ChjSlot &&) = delete;
};


/// CH-style slotted table for concurrent hash join.
///
/// Mirrors ClickHouse's `ConcurrentHashJoin`: `n_slots` (power-of-2, ≤ 256)
/// independent build partitions selected by `hash & (n_slots - 1)`.
///
/// Build: every worker hashes its build block, bins rows by slot index, then
/// for each non-empty bin acquires that slot's mutex and inserts — a plain
/// single-writer JoinHashTable insert with no per-cell locking. Workers can
/// build into different slots simultaneously.
///
/// Probe: each probe block is scattered by the same low hash bits, and each
/// slot's table is queried independently. No locking needed after the build
/// barrier since every slot's table is read-only.
class ChjSlottedTable
{
public:
    static constexpr size_t MAX_SLOTS = 256;

    ChjSlottedTable(size_t n_slots, std::pmr::memory_resource * mr)
    {
        n_slots = std::max(n_slots, size_t{1});
        n_slots = std::bit_ceil(n_slots);
        n_slots = std::min(n_slots, MAX_SLOTS);
        n_slots_ = n_slots;

        slots_.resize(n_slots_);
        for (auto & s : slots_)
            s = std::make_unique<ChjSlot>(mr);
    }

    [[nodiscard]] size_t numSlots() const noexcept { return n_slots_; }

    [[nodiscard]] size_t slotOf(uint64_t hash) const noexcept { return static_cast<size_t>(hash) & (n_slots_ - 1); }

    [[nodiscard]] ChjSlot & slot(size_t i) noexcept { return *slots_[i]; }
    [[nodiscard]] const ChjSlot & slot(size_t i) const noexcept { return *slots_[i]; }

    /// Pre-reserve block-vector capacity in every slot's store.
    void reserveSlotBlocks(size_t blocks_per_slot)
    {
        for (auto & s : slots_)
            s->store.reserveBlocks(blocks_per_slot);
    }

private:
    size_t n_slots_ = 1;
    std::vector<std::unique_ptr<ChjSlot>> slots_;
};

} // namespace phj
