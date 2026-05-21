#pragma once

#include <cstdint>
#include <vector>

#include "Block.h"
#include "Types.h"


namespace phj
{

/// Multi-pass radix configuration. `pass_bits` lists the number of hash
/// bits consumed at each pass; the sum is the total log2 of the
/// partition count. Single-pass is `pass_bits = {N}`; two-pass is
/// `pass_bits = {N1, N2}`. All bits come from the TOP of the 64-bit
/// hash; the highest bits are consumed first so the lower bits remain
/// well-distributed for the in-partition hashtable.
struct RadixConfig
{
    std::vector<uint8_t> pass_bits;

    [[nodiscard]] uint8_t totalBits() const noexcept
    {
        uint8_t sum = 0;
        for (auto b : pass_bits)
            sum = static_cast<uint8_t>(sum + b);
        return sum;
    }

    [[nodiscard]] size_t partitions() const noexcept { return size_t{1} << totalBits(); }
};


/// One block in a partition's output chain. Holds the key column plus K
/// payload columns, each `capacity` slots wide, `rows` of which are
/// filled. The doubling-capacity scheme starts at ~16 KiB per buffer
/// (per the prior spec) and doubles on each new block subject to
/// `MAX_OUT_BLOCK_ROWS`.
struct OutBlock
{
    size_t rows = 0;
    size_t capacity = 0;
    std::vector<uint64_t> keys;
    std::vector<Column> payloads;
};


/// Per-partition output of the radix shuffle: an append-only sequence
/// of `OutBlock`s held in the order they were grown (oldest first;
/// equivalent to the linked-list chain in the reference). The build
/// phase appends these blocks one at a time into the partition's
/// `BlockStore`, recording each block's assigned `block_no` so the HT
/// cells encode `(key, RowRefCell{block_no, row_no})`.
struct OutBlockChain
{
    std::vector<OutBlock> blocks;
    size_t total_rows = 0;
};


/// Output of the radix shuffle: per-partition chain (rolled-up across
/// the per-thread shards used during scatter).
struct PartitionedShuffleOutput
{
    size_t partitions = 0;
    PayloadSchema schema;
    /// chains[partition]; populated only once, after the shuffle returns.
    std::vector<OutBlockChain> chains;
    std::vector<size_t> partition_rows;
};


/// Run a radix shuffle over the input block stream. The block-pipelined
/// operator consumes each `block` of the stream as a single batch and
/// dispatches the canonical 5-phase pipeline (SIMD hash -> histogram ->
/// pre-grow + commit -> column-first scatter -> next batch).
///
/// Both `keys` and the columns whose types are listed in `schema`
/// (mapped through `in_col_index`) are scattered into the per-partition
/// `OutBlock` chains. `cfg` lists the bits per pass.
PartitionedShuffleOutput radixShuffle(
    const BlockStream & input,
    const PayloadSchema & out_schema,
    const std::vector<size_t> & in_col_index,
    const RadixConfig & cfg,
    size_t threads);


/// Convenience: shuffle a stream whose schema matches its own output
/// (identity column map).
PartitionedShuffleOutput radixShuffle(const BlockStream & input, const RadixConfig & cfg, size_t threads);

}
