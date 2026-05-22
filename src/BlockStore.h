#pragma once

#include <cstddef>
#include <memory_resource>
#include <mutex>
#include <utility>
#include <vector>

#include "Block.h"
#include "Types.h"


namespace phj
{

/// Build-side block store: a Vector<Block> appended to during build,
/// plus a parallel next-chain matrix that mirrors the block layout and
/// holds, for each (block_no, row_no), the previous `RowRefCell` in the
/// multi-match chain (or INVALID_REF if this row is the chain head).
///
/// In PHJ each partition owns its own BlockStore (single-writer; the
/// work-stealing worker that claimed the partition). In CHJ a single
/// BlockStore is shared across the 256 sub-tables; concurrent appends
/// from multiple build workers serialise on an internal mutex. Reads
/// of `blocks` and `next_chain` after the build/probe barrier are
/// unsynchronised; each appended block's `next_chain` row is written
/// only by the worker that appended that block (the previous-ref
/// returned by `insertLocked` is a value, not a pointer; no other
/// worker writes that slot).
///
/// All internal allocations (block storage and next-chain row vectors)
/// are routed through the `memory_resource` supplied at construction,
/// enabling per-run peak-memory tracking via `MemTracker`.
class BlockStore
{
public:
    BlockStore()
        : BlockStore(std::pmr::get_default_resource())
    {
    }

    explicit BlockStore(std::pmr::memory_resource * mr)
        : blocks_(mr)
        , next_chain_(mr)
    {
    }

    BlockStore(const BlockStore &) = delete;
    BlockStore & operator=(const BlockStore &) = delete;
    BlockStore(BlockStore &&) = delete;
    BlockStore & operator=(BlockStore &&) = delete;
    ~BlockStore() = default;

    /// Append a block (single-threaded path). Returns the assigned
    /// `block_no`. The block's contents are moved in.
    BlockNo append(Block && block)
    {
        const BlockNo block_no = static_cast<BlockNo>(blocks_.size());
        const size_t rows = block.rows;
        blocks_.push_back(std::move(block));
        next_chain_.emplace_back(rows, INVALID_REF);
        return block_no;
    }

    /// Append a block with mutex serialisation (CHJ path). Returns
    /// the assigned `block_no`.
    BlockNo appendLocked(Block && block)
    {
        const std::scoped_lock lock(mu_);
        return append(std::move(block));
    }

    /// Reserve capacity for `n` blocks in advance. Avoids reallocations
    /// during concurrent CHJ appends.
    void reserveBlocks(size_t n)
    {
        blocks_.reserve(n);
        next_chain_.reserve(n);
    }

    [[nodiscard]] size_t numBlocks() const noexcept { return blocks_.size(); }

    [[nodiscard]] const Block & block(BlockNo i) const noexcept { return blocks_[i]; }

    /// Write the previous chain ref for `(block_no, row_no)`.
    void setPrev(RowRefCell ref, RowRefCell prev) noexcept { next_chain_[ref.block_no][ref.row_no] = prev; }

    /// Read the previous chain ref for `(block_no, row_no)`.
    [[nodiscard]] RowRefCell getPrev(RowRefCell ref) const noexcept { return next_chain_[ref.block_no][ref.row_no]; }

private:
    std::pmr::vector<Block> blocks_;
    std::pmr::vector<std::pmr::vector<RowRefCell>> next_chain_;
    std::mutex mu_;
};

}
