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

    /// Non-owning view of the OutBlock as a `BlockView`. Exposes only
    /// the first `rows` slots even though the underlying buffers may
    /// have trailing capacity. Lets probe and refinement operators
    /// consume `OutBlock`s through the same interface as pipeline
    /// `Block`s without copying.
    [[nodiscard]] BlockView view() const noexcept { return {rows, keys.data(), payloads.data(), payloads.size()}; }
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


/// ============================================================
/// Reusable per-worker scatter machinery
/// ------------------------------------------------------------
/// The radix scatter is structured as a 5-phase per-batch pipeline
/// against a per-(thread, partition) `PartitionOut` chain. The
/// pieces below expose that machinery for callers that drive the
/// scatter outside of the full `radixShuffle` orchestrator — in
/// particular PHJ-BEP, which runs the first pass on incoming
/// probe blocks and the trailing passes on a single already-
/// buffered partition under a memory-budget eviction loop.
/// ============================================================

/// Initial per-column buffer rows for a newly-allocated `OutBlock`.
/// Sized so that the column with the largest element type yields a
/// ~16 KiB buffer; smaller-element columns occupy proportionally less.
[[nodiscard]] size_t initialOutBlockRows(const PayloadSchema & schema) noexcept;


/// Per-(thread, partition) growable output chain. The chain matches
/// the radix scatter's structure (singly-linked block list, kept as
/// a vector here for cache-friendlier downstream traversal). The
/// doubling-capacity scheme starts at ~16 KiB per buffer and doubles
/// on each new block, capped at `MAX_OUT_BLOCK_ROWS`.
struct PartitionOut
{
    std::vector<OutBlock> blocks;
    OutBlock * cur = nullptr;
    size_t next_cap = 0;

    /// Initialise an empty chain with `initial_cap` as the next-to-be-
    /// allocated `OutBlock`'s capacity. Must be called before any
    /// scatter into this chain.
    void init(size_t initial_cap) noexcept;

    /// Allocate a new `OutBlock` big enough to hold at least
    /// `min_required` rows, doubling `next_cap` for the next-to-be-
    /// allocated block.
    void grow(const PayloadSchema & schema, size_t min_required);
};


/// Convenience: initialise `out` with the configured per-block initial
/// capacity for `schema`. Equivalent to `out.init(initialOutBlockRows(schema))`.
void initPartitionOut(PartitionOut & out, const PayloadSchema & schema);


/// Total filled rows across all `OutBlock`s in the chain.
[[nodiscard]] size_t partitionRows(const PartitionOut & po) noexcept;


/// Drop all buffered `OutBlock`s, releasing their memory. `next_cap` is
/// preserved so subsequent fills retain the doubling progression.
void dropPartition(PartitionOut & po) noexcept;


/// Scratch buffers reused across `scatterBatch` calls within one worker.
/// Each vector is resized on demand to the maximum size seen.
struct ScatterScratch
{
    std::vector<uint64_t> hashes;
    std::vector<uint32_t> pids;
    std::vector<uint32_t> local_hist;
    std::vector<uint64_t *> key_ptrs;
    std::vector<std::byte *> ptrs_flat;
};


/// One scatter pass over one batch of rows.
///   `in`           — input batch view (keys + payloads, `in.rows` wide).
///   `in_col_index` — per output-schema column, the position of the
///                    corresponding input payload column inside `in`.
///                    Identity for same-schema scatter.
///   `out_schema`   — destination payload schema (same as input order
///                    when `in_col_index` is identity).
///   `shift`        — number of low bits to skip in the 64-bit hash
///                    before consuming the partition-bit window.
///   `parts`        — power-of-2 partition count.
///   `out_chains`   — `parts` per-partition output chains; the
///                    function appends rows to whichever the hash
///                    selects.
///   `scratch`      — reused working buffers, resized as needed.
///
/// On return, `scratch.local_hist[p]` is the number of rows scattered
/// into partition `p` during this batch. Callers that need a per-
/// partition row delta read it directly from there.
void scatterBatch(
    BlockView in,
    const std::vector<size_t> & in_col_index,
    const PayloadSchema & out_schema,
    uint32_t shift,
    size_t parts,
    PartitionOut * out_chains,
    ScatterScratch & scratch);


/// Refine a single already-buffered partition through the trailing
/// passes of a multi-pass radix configuration, splitting its rows into
/// `leaves_per_partition` leaf chains.
///   `input`         — chain consumed (its blocks are moved through
///                     intermediate buffers); empty on return.
///   `pass_bits`     — full pass_bits list; `pass_bits[0]` is the
///                     bit-width of the pass already consumed by the
///                     caller (the first pass). The function runs the
///                     trailing passes `[1..pass_bits.size())`.
///   `leaves_out`    — array of `prod(2^pass_bits[1..end])` chains
///                     populated in leaf order within this partition's
///                     leaf range.
///   `scratch`       — reused working buffers.
///
/// Refinement preserves the total filled row count (and total byte
/// content) — it is a pure structural restripe.
void refineToLeaves(
    PartitionOut && input,
    const PayloadSchema & schema,
    const std::vector<uint8_t> & pass_bits,
    PartitionOut * leaves_out,
    ScatterScratch & scratch);


/// Convert one `OutBlock` from a partition's chain into a pipeline
/// `Block` suitable for the build operator: same column-major layout,
/// vectors truncated to the actual filled row count. The `OutBlock`
/// is consumed (its key and payload buffers are moved); the resulting
/// `Block` owns the storage that the build-side `BlockStore` will
/// then retain across the build/probe lifetime.
[[nodiscard]] Block outBlockToBlock(OutBlock && ob);

}
