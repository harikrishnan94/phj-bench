#pragma once

#include <cstdint>
#include <memory_resource>
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
///
/// OutBlock is allocator-aware so that pmr containers of OutBlocks
/// propagate their resource into each block's key and payload vectors.
struct OutBlock
{
    using allocator_type = std::pmr::polymorphic_allocator<>;

    size_t rows = 0;
    size_t capacity = 0;
    std::pmr::vector<uint64_t> keys;
    std::pmr::vector<Column> payloads;

    OutBlock()
        : keys(std::pmr::get_default_resource())
        , payloads(std::pmr::get_default_resource())
    {
    }
    explicit OutBlock(std::pmr::memory_resource * mr)
        : keys(mr)
        , payloads(mr)
    {
    }

    OutBlock(std::allocator_arg_t, allocator_type const & alloc)
        : keys(alloc)
        , payloads(alloc)
    {
    }

    OutBlock(const OutBlock & o, allocator_type const & alloc)
        : rows(o.rows)
        , capacity(o.capacity)
        , keys(o.keys, alloc)
        , payloads(o.payloads, alloc)
    {
    }

    OutBlock(OutBlock && o, allocator_type const & alloc)
        : rows(o.rows)
        , capacity(o.capacity)
        , keys(std::move(o.keys), alloc)
        , payloads(std::move(o.payloads), alloc)
    {
    }

    OutBlock(const OutBlock &) = default;
    OutBlock(OutBlock &&) noexcept = default;
    OutBlock & operator=(const OutBlock &) = default;
    OutBlock & operator=(OutBlock &&) noexcept = default;

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
///
/// OutBlockChain is allocator-aware so pmr containers propagate their
/// resource into the chain's block vector.
struct OutBlockChain
{
    using allocator_type = std::pmr::polymorphic_allocator<>;

    std::pmr::vector<OutBlock> blocks;
    size_t total_rows = 0;

    OutBlockChain()
        : blocks(std::pmr::get_default_resource())
    {
    }
    explicit OutBlockChain(std::pmr::memory_resource * mr)
        : blocks(mr)
    {
    }

    OutBlockChain(std::allocator_arg_t, allocator_type const & alloc)
        : blocks(alloc)
    {
    }

    OutBlockChain(const OutBlockChain & o, allocator_type const & alloc)
        : blocks(o.blocks, alloc)
        , total_rows(o.total_rows)
    {
    }

    OutBlockChain(OutBlockChain && o, allocator_type const & alloc)
        : blocks(std::move(o.blocks), alloc)
        , total_rows(o.total_rows)
    {
    }

    OutBlockChain(const OutBlockChain &) = default;
    OutBlockChain(OutBlockChain &&) noexcept = default;
    OutBlockChain & operator=(const OutBlockChain &) = default;
    OutBlockChain & operator=(OutBlockChain &&) noexcept = default;
};


/// Output of the radix shuffle: per-partition chain (rolled-up across
/// the per-thread shards used during scatter).
struct PartitionedShuffleOutput
{
    size_t partitions = 0;
    PayloadSchema schema;
    std::pmr::vector<OutBlockChain> chains;
    std::pmr::vector<size_t> partition_rows;

    PartitionedShuffleOutput()
        : chains(std::pmr::get_default_resource())
        , partition_rows(std::pmr::get_default_resource())
    {
    }

    explicit PartitionedShuffleOutput(std::pmr::memory_resource * mr)
        : chains(mr)
        , partition_rows(mr)
    {
    }

    PartitionedShuffleOutput(const PartitionedShuffleOutput &) = default;
    PartitionedShuffleOutput(PartitionedShuffleOutput &&) noexcept = default;
    PartitionedShuffleOutput & operator=(const PartitionedShuffleOutput &) = default;
    PartitionedShuffleOutput & operator=(PartitionedShuffleOutput &&) noexcept = default;
};


/// Run a radix shuffle over the input block stream. The block-pipelined
/// operator consumes each `block` of the stream as a single batch and
/// dispatches the canonical 5-phase pipeline (SIMD hash -> histogram ->
/// pre-grow + commit -> column-first scatter -> next batch).
///
/// Both `keys` and the columns whose types are listed in `schema`
/// (mapped through `in_col_index`) are scattered into the per-partition
/// `OutBlock` chains. `cfg` lists the bits per pass. All intermediate
/// and output allocations are made through `mr` (defaults to
/// `new_delete_resource()`).
PartitionedShuffleOutput radixShuffle(
    const BlockStream & input,
    const PayloadSchema & out_schema,
    const std::vector<size_t> & in_col_index,
    const RadixConfig & cfg,
    size_t threads,
    std::pmr::memory_resource * mr = std::pmr::get_default_resource());


/// Convenience: shuffle a stream whose schema matches its own output
/// (identity column map).
PartitionedShuffleOutput radixShuffle(
    const BlockStream & input, const RadixConfig & cfg, size_t threads, std::pmr::memory_resource * mr = std::pmr::get_default_resource());


/// ============================================================
/// Reusable per-worker scatter machinery
/// ------------------------------------------------------------
/// The radix scatter is structured as a 5-phase per-batch pipeline
/// against a per-(thread, partition) `PartitionOut` chain. The
/// pieces below expose that machinery for callers that drive the
/// scatter outside of the full `radixShuffle` orchestrator — in
/// particular the BEP-backed PHJ path, which runs the first pass on incoming
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
///
/// PartitionOut is allocator-aware so that pmr containers of
/// PartitionOuts propagate their resource into each chain's block
/// vector (and transitively into each OutBlock's key/payload vectors).
struct PartitionOut
{
    using allocator_type = std::pmr::polymorphic_allocator<>;

    std::pmr::vector<OutBlock> blocks;
    OutBlock * cur = nullptr;
    size_t next_cap = 0;

    PartitionOut()
        : blocks(std::pmr::get_default_resource())
    {
    }
    explicit PartitionOut(std::pmr::memory_resource * mr)
        : blocks(mr)
    {
    }

    PartitionOut(std::allocator_arg_t, allocator_type const & alloc)
        : blocks(alloc)
    {
    }

    PartitionOut(const PartitionOut & o, allocator_type const & alloc)
        : blocks(o.blocks, alloc)
        , cur(nullptr)
        , next_cap(o.next_cap)
    {
    }

    /// Move with (possibly different) allocator. When `alloc` matches
    /// the source's resource the blocks buffer is stolen O(1); when
    /// different the blocks are copied. `cur` is always reset to null
    /// after a move-with-allocator to avoid a stale pointer.
    PartitionOut(PartitionOut && o, allocator_type const & alloc)
        : blocks(std::move(o.blocks), alloc)
        , cur(nullptr)
        , next_cap(o.next_cap)
    {
        o.cur = nullptr;
    }

    /// Regular move: steal the buffer (allocator is transferred for
    /// pmr move-ctor) and null out `cur` on the source.
    PartitionOut(PartitionOut && o) noexcept
        : blocks(std::move(o.blocks))
        , cur(o.cur)
        , next_cap(o.next_cap)
    {
        o.cur = nullptr;
    }

    PartitionOut(const PartitionOut &) = default;
    PartitionOut & operator=(const PartitionOut &) = default;
    PartitionOut & operator=(PartitionOut && o) noexcept
    {
        if (this != &o)
        {
            blocks = std::move(o.blocks);
            cur = nullptr;
            next_cap = o.next_cap;
            o.cur = nullptr;
        }
        return *this;
    }

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
/// Each vector is resized on demand to the maximum size seen. Constructed
/// with an explicit resource so all scratch allocations are tracked.
struct ScatterScratch
{
    std::pmr::vector<uint64_t> hashes;
    std::pmr::vector<uint32_t> pids;
    std::pmr::vector<uint32_t> local_hist;
    std::pmr::vector<uint64_t *> key_ptrs;
    std::pmr::vector<std::byte *> ptrs_flat;

    ScatterScratch()
        : ScatterScratch(std::pmr::get_default_resource())
    {
    }

    explicit ScatterScratch(std::pmr::memory_resource * mr)
        : hashes(mr)
        , pids(mr)
        , local_hist(mr)
        , key_ptrs(mr)
        , ptrs_flat(mr)
    {
    }
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
///   `mr`            — memory resource for any intermediate chains
///                     allocated during 3+ pass refinement.
///
/// Refinement preserves the total filled row count (and total byte
/// content) — it is a pure structural restripe.
void refineToLeaves(
    PartitionOut && input,
    const PayloadSchema & schema,
    const std::vector<uint8_t> & pass_bits,
    PartitionOut * leaves_out,
    ScatterScratch & scratch,
    std::pmr::memory_resource * mr = std::pmr::get_default_resource());


/// Convert one `OutBlock` from a partition's chain into a pipeline
/// `Block` suitable for the build operator: same column-major layout,
/// vectors truncated to the actual filled row count. The `OutBlock`
/// is consumed (its key and payload buffers are moved); the resulting
/// `Block` owns the storage that the build-side `BlockStore` will
/// then retain across the build/probe lifetime.
///
/// The returned Block uses the same memory resource as `ob` so that
/// the buffer transfer is an O(1) pointer-steal (same allocator) rather
/// than a copy.
[[nodiscard]] Block outBlockToBlock(OutBlock && ob);

}
