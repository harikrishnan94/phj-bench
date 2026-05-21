#include "RadixPartition.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "Hash.h"
#include "Threading.h"


namespace phj
{

namespace
{

/// Per-(column, partition) live write pointer. The reference uses one
/// flat ptrs_flat[K*P] array; we keep the same layout here.
using PtrsFlat = std::vector<std::byte *>;


/// Function-pointer scatter: writes `n` values from `src` into the
/// per-partition column buffers via the column-major branch-free
/// inner loop `*ptrs[pids[j]]++ = src[j]`. One specialisation per
/// supported width (1/2/4/8/16 bytes). The reference file's
/// `UInt64Column::scatter_direct` is the prototype.
using ScatterFn = void (*)(std::byte ** col_ptrs, const uint32_t * pids, const std::byte * src_bytes, size_t n);

template <typename T>
[[gnu::hot]] void scatterTyped(std::byte ** col_ptrs, const uint32_t * pids, const std::byte * src_bytes, size_t n)
{
    auto ** typed_ptrs = reinterpret_cast<T **>(col_ptrs);
    const auto * src = reinterpret_cast<const T *>(src_bytes);
    for (size_t j = 0; j < n; ++j)
        *typed_ptrs[pids[j]]++ = src[j];
}

[[nodiscard]] ScatterFn scatterFor(PayloadType t) noexcept
{
    switch (t)
    {
        case PayloadType::UInt8:
            return scatterTyped<uint8_t>;
        case PayloadType::UInt16:
            return scatterTyped<uint16_t>;
        case PayloadType::UInt32:
            return scatterTyped<uint32_t>;
        case PayloadType::UInt64:
            return scatterTyped<uint64_t>;
        case PayloadType::UInt128:
            return scatterTyped<UInt128>;
    }
    return nullptr;
}


constexpr size_t TARGET_INITIAL_BUFFER_BYTES = 16 * 1024;
constexpr size_t MAX_OUT_BLOCK_ROWS = 65'536;


void initOutBlock(OutBlock & blk, const PayloadSchema & schema, size_t capacity)
{
    blk.rows = 0;
    blk.capacity = capacity;
    blk.keys.assign(capacity, 0);
    blk.payloads.assign(schema.types.size(), {});
    for (size_t c = 0; c < schema.types.size(); ++c)
    {
        blk.payloads[c].type = schema.types[c];
        blk.payloads[c].data.assign(capacity * payloadTypeSize(schema.types[c]), std::byte{});
    }
}


/// A `ThreadShard` is a contiguous span of per-partition `PartitionOut`
/// chains, one per partition. The radix shuffle owns one per worker.
struct ThreadShard
{
    /// chains[partition]
    std::vector<PartitionOut> chains;

    void init(size_t partitions, const PayloadSchema & schema)
    {
        const size_t initial_cap = initialOutBlockRows(schema);
        chains.assign(partitions, {});
        for (auto & c : chains)
            c.init(initial_cap);
    }
};


/// Sizes the column-element-size lookup table once per pass.
[[nodiscard]] std::vector<size_t> elementSizes(const PayloadSchema & schema)
{
    std::vector<size_t> v(schema.types.size());
    for (size_t c = 0; c < schema.types.size(); ++c)
        v[c] = payloadTypeSize(schema.types[c]);
    return v;
}


[[nodiscard]] std::vector<ScatterFn> scatterTable(const PayloadSchema & schema)
{
    std::vector<ScatterFn> v(schema.types.size());
    for (size_t c = 0; c < schema.types.size(); ++c)
        v[c] = scatterFor(schema.types[c]);
    return v;
}


/// Concatenate per-thread shards into final per-partition chains. Each
/// partition's output is the concatenation of all threads' OutBlocks
/// for that partition (in thread-id order; the within-thread order is
/// preserved). Each block keeps its grown capacity; `rows` is the
/// number of slots filled.
PartitionedShuffleOutput collectChains(std::vector<ThreadShard> shards, const PayloadSchema & out_schema)
{
    PartitionedShuffleOutput out;
    const size_t parts = shards.empty() ? 0 : shards.front().chains.size();
    out.partitions = parts;
    out.schema = out_schema;
    out.chains.assign(parts, {});
    out.partition_rows.assign(parts, 0);
    for (size_t p = 0; p < parts; ++p)
    {
        OutBlockChain & chain = out.chains[p];
        size_t total = 0;
        for (auto & sh : shards)
        {
            for (auto & blk : sh.chains[p].blocks)
            {
                total += blk.rows;
                if (blk.rows > 0)
                    chain.blocks.push_back(std::move(blk));
            }
        }
        chain.total_rows = total;
        out.partition_rows[p] = total;
    }
    return out;
}


PartitionedShuffleOutput shufflePassFromBlockStream(
    const BlockStream & input,
    const PayloadSchema & out_schema,
    const std::vector<size_t> & in_col_index,
    size_t parts,
    uint32_t shift,
    size_t threads)
{
    std::vector<ThreadShard> shards(threads);
    for (auto & sh : shards)
        sh.init(parts, out_schema);

    const size_t n_blocks = input.blocks.size();
    parallelRun(
        threads,
        [&](size_t tid)
        {
            const size_t start = (n_blocks * tid) / threads;
            const size_t end = (n_blocks * (tid + 1)) / threads;
            ScatterScratch scratch;
            for (size_t b = start; b < end; ++b)
            {
                const Block & blk = input.blocks[b];
                if (blk.rows == 0)
                    continue;
                scatterBatch(blk.view(), in_col_index, out_schema, shift, parts, shards[tid].chains.data(), scratch);
            }
        });

    return collectChains(std::move(shards), out_schema);
}


PartitionedShuffleOutput shufflePassFromChains(const PartitionedShuffleOutput & input, size_t sub_parts, uint32_t shift, size_t threads)
{
    const size_t in_parts = input.partitions;
    const size_t out_parts = in_parts * sub_parts;
    const size_t n_payload = input.schema.types.size();

    std::vector<ThreadShard> shards(threads);
    for (auto & sh : shards)
        sh.init(out_parts, input.schema);

    /// Identity column map: input is already in `input.schema` order.
    std::vector<size_t> identity(n_payload);
    for (size_t c = 0; c < n_payload; ++c)
        identity[c] = c;

    parallelRun(
        threads,
        [&](size_t tid)
        {
            ThreadShard & shard = shards[tid];
            ScatterScratch scratch;
            /// Each thread processes a disjoint set of input partitions;
            /// the sub-partition the thread emits encodes as
            /// `ip * sub_parts + s`.
            for (size_t ip = tid; ip < in_parts; ip += threads)
            {
                const size_t base_idx = ip * sub_parts;
                for (const OutBlock & ob : input.chains[ip].blocks)
                {
                    if (ob.rows == 0)
                        continue;
                    scatterBatch(ob.view(), identity, input.schema, shift, sub_parts, shard.chains.data() + base_idx, scratch);
                }
            }
        });

    return collectChains(std::move(shards), input.schema);
}

}


/// ============================================================
/// Public scatter primitives
/// ============================================================

[[nodiscard]] size_t initialOutBlockRows(const PayloadSchema & schema) noexcept
{
    size_t max_element_size = sizeof(uint64_t);
    for (auto t : schema.types)
        max_element_size = std::max(max_element_size, payloadTypeSize(t));
    return std::max<size_t>(1, TARGET_INITIAL_BUFFER_BYTES / max_element_size);
}


void PartitionOut::init(size_t initial_cap) noexcept
{
    blocks.clear();
    cur = nullptr;
    next_cap = initial_cap;
}


void PartitionOut::grow(const PayloadSchema & schema, size_t min_required)
{
    /// The first allocation for this chain is bumped up if the
    /// impending batch's per-partition load exceeds `next_cap`. This
    /// matters when the pipeline batch size is large relative to the
    /// partition fanout (e.g., pass-1 with P=4 and 10K-row batches,
    /// or a heavily-skewed bucket in a later pass with up-to-65K-row
    /// input OutBlocks).
    const size_t cap = std::max(next_cap, min_required);
    blocks.emplace_back();
    OutBlock & blk = blocks.back();
    initOutBlock(blk, schema, cap);
    cur = &blocks.back();
    next_cap = std::min<size_t>(cap * 2, MAX_OUT_BLOCK_ROWS);
}


void initPartitionOut(PartitionOut & out, const PayloadSchema & schema)
{
    out.init(initialOutBlockRows(schema));
}


size_t partitionRows(const PartitionOut & po) noexcept
{
    size_t s = 0;
    for (const auto & b : po.blocks)
        s += b.rows;
    return s;
}


void dropPartition(PartitionOut & po) noexcept
{
    po.blocks.clear();
    po.cur = nullptr;
    /// `next_cap` is preserved.
}


void scatterBatch(
    BlockView in,
    const std::vector<size_t> & in_col_index,
    const PayloadSchema & out_schema,
    uint32_t shift,
    size_t parts,
    PartitionOut * out_chains,
    ScatterScratch & scratch)
{
    const size_t rows = in.rows;
    if (rows == 0)
        return;

    const size_t K = out_schema.types.size();
    const uint64_t mask = parts - 1;

    /// Resize scratch buffers as needed (they grow over the life of
    /// the worker and never shrink).
    if (scratch.hashes.size() < rows)
        scratch.hashes.resize(rows);
    if (scratch.pids.size() < rows)
        scratch.pids.resize(rows);
    if (scratch.local_hist.size() < parts)
        scratch.local_hist.assign(parts, 0);
    else
        std::memset(scratch.local_hist.data(), 0, parts * sizeof(uint32_t));
    if (scratch.key_ptrs.size() < parts)
        scratch.key_ptrs.resize(parts);
    if (scratch.ptrs_flat.size() < K * parts)
        scratch.ptrs_flat.resize(K * parts);

    /// Column-element-size lookup; rebuilt per call to keep the call-
    /// signature scratch-free. Cheap relative to the row loops.
    const std::vector<size_t> in_sizes = elementSizes(out_schema);
    const std::vector<ScatterFn> scatters = scatterTable(out_schema);

    uint64_t * hashes = scratch.hashes.data();
    uint32_t * pids = scratch.pids.data();
    uint32_t * local_hist = scratch.local_hist.data();
    uint64_t ** key_ptrs = scratch.key_ptrs.data();
    std::byte ** ptrs_flat = scratch.ptrs_flat.data();

    /// Phase 1: SIMD hash over keys -> pids[].
    intHash64Batch(in.keys, rows, hashes);
    for (size_t j = 0; j < rows; ++j)
        pids[j] = static_cast<uint32_t>((hashes[j] >> shift) & mask);

    /// Phase 2: histogram.
    for (size_t j = 0; j < rows; ++j)
        ++local_hist[pids[j]];

    /// Phase 3: pre-grow + commit `filled` + live-pointer setup.
    for (size_t p = 0; p < parts; ++p)
    {
        if (local_hist[p] == 0)
            continue;
        PartitionOut & chain = out_chains[p];
        if (chain.cur == nullptr || chain.cur->rows + local_hist[p] > chain.cur->capacity)
            chain.grow(out_schema, local_hist[p]);
        OutBlock & ob = *chain.cur;
        const size_t base = ob.rows;
        ob.rows = base + local_hist[p];
        key_ptrs[p] = ob.keys.data() + base;
        for (size_t c = 0; c < K; ++c)
            ptrs_flat[c * parts + p] = ob.payloads[c].raw() + base * in_sizes[c];
    }

    /// Phase 4: column-first branch-free scatter. Keys first, then
    /// each payload column via the per-type scatter function.
    for (size_t j = 0; j < rows; ++j)
        *key_ptrs[pids[j]]++ = in.keys[j];
    for (size_t c = 0; c < K; ++c)
    {
        std::byte ** col_ptrs = ptrs_flat + c * parts;
        const std::byte * src = in.payloads[in_col_index[c]].raw();
        scatters[c](col_ptrs, pids, src, rows);
    }
}


Block outBlockToBlock(OutBlock && ob)
{
    Block b;
    b.rows = ob.rows;
    b.keys = std::move(ob.keys);
    b.keys.resize(ob.rows);
    b.payloads.resize(ob.payloads.size());
    for (size_t c = 0; c < ob.payloads.size(); ++c)
    {
        b.payloads[c].type = ob.payloads[c].type;
        b.payloads[c].data = std::move(ob.payloads[c].data);
        b.payloads[c].data.resize(ob.rows * payloadTypeSize(b.payloads[c].type));
    }
    return b;
}


void refineToLeaves(
    PartitionOut && input,
    const PayloadSchema & schema,
    const std::vector<uint8_t> & pass_bits,
    PartitionOut * leaves_out,
    ScatterScratch & scratch)
{
    if (pass_bits.empty())
        throw std::invalid_argument("refineToLeaves: empty pass_bits");

    const size_t total_passes = pass_bits.size();

    /// Single-pass refinement is structurally a no-op: pass-1 buckets
    /// ARE leaves (leaves_per_pass1 == 1). Move the input chain's
    /// `OutBlock`s directly into `leaves_out[0]`. Leaves_out's `cur` is
    /// left untouched so the next pass-1 scatter into that leaf range
    /// (if any) starts a fresh OutBlock; under BEP-1-pass the same
    /// bucket is repeatedly refined so each refinement just appends
    /// whatever the bucket has accumulated since the last drain.
    if (total_passes == 1)
    {
        for (auto & blk : input.blocks)
        {
            if (blk.rows > 0)
                leaves_out[0].blocks.push_back(std::move(blk));
        }
        input.blocks.clear();
        input.cur = nullptr;
        return;
    }

    /// Identity column map: refinement input and output share `schema`.
    std::vector<size_t> identity(schema.types.size());
    for (size_t c = 0; c < schema.types.size(); ++c)
        identity[c] = c;

    /// Two-pass refinement: scatter the input chain DIRECTLY into the
    /// destination leaf chains, appending into whichever OutBlocks the
    /// leaves already hold from earlier refinements of the same bucket.
    /// This is critical under BEP — a single pass-1 bucket is refined
    /// many times across the run (once each time the bucket grows
    /// largest), so packing successive refinements into the existing
    /// per-leaf OutBlocks (instead of allocating a fresh
    /// `leaves_per_p1`-wide set of OutBlocks per refinement that are
    /// then drained into `leaves_out`) is the difference between
    /// O(refinements × leaves_per_p1) OutBlock allocations and
    /// O(total_leaf_rows / OutBlock_capacity) — typically 10–30× less
    /// allocation traffic and dramatically denser cache lines.
    if (total_passes == 2)
    {
        const uint8_t bi = pass_bits[1];
        const size_t sub = size_t{1} << bi;
        const uint32_t shift = static_cast<uint32_t>(64u - pass_bits[0] - bi);
        for (const OutBlock & ob : input.blocks)
        {
            if (ob.rows == 0)
                continue;
            scatterBatch(ob.view(), identity, schema, shift, sub, leaves_out, scratch);
        }
        input.blocks.clear();
        input.cur = nullptr;
        return;
    }

    /// Three or more passes: cascade the intermediate passes through
    /// temporary chains (one set per pass) and run the final pass
    /// straight into `leaves_out`. The final-pass shortcut keeps the
    /// per-leaf chains dense across successive refinements; only the
    /// intermediate passes pay per-refinement allocation cost.
    std::vector<PartitionOut> cur;
    cur.emplace_back(std::move(input));
    uint8_t consumed = pass_bits[0];

    for (size_t pi = 1; pi + 1 < total_passes; ++pi)
    {
        const uint8_t bi = pass_bits[pi];
        const size_t sub = size_t{1} << bi;
        const uint32_t shift = static_cast<uint32_t>(64u - consumed - bi);

        std::vector<PartitionOut> next(cur.size() * sub);
        for (auto & po : next)
            initPartitionOut(po, schema);

        for (size_t i = 0; i < cur.size(); ++i)
        {
            const size_t base = i * sub;
            for (const OutBlock & ob : cur[i].blocks)
            {
                if (ob.rows == 0)
                    continue;
                scatterBatch(ob.view(), identity, schema, shift, sub, next.data() + base, scratch);
            }
            /// Free the consumed level's blocks immediately so peak
            /// refinement-time memory is ~2× the input chain rather
            /// than (passes + 1)×.
            cur[i].blocks.clear();
            cur[i].cur = nullptr;
        }

        cur = std::move(next);
        consumed = static_cast<uint8_t>(consumed + bi);
    }

    /// Final pass — scatter the deepest intermediate level into
    /// `leaves_out` directly.
    const uint8_t bf = pass_bits.back();
    const size_t sub_f = size_t{1} << bf;
    const uint32_t shift_f = static_cast<uint32_t>(64u - consumed - bf);
    for (size_t i = 0; i < cur.size(); ++i)
    {
        const size_t base = i * sub_f;
        for (const OutBlock & ob : cur[i].blocks)
        {
            if (ob.rows == 0)
                continue;
            scatterBatch(ob.view(), identity, schema, shift_f, sub_f, leaves_out + base, scratch);
        }
        cur[i].blocks.clear();
        cur[i].cur = nullptr;
    }
}


/// ============================================================
/// `radixShuffle` orchestrator (uses the public primitives above)
/// ============================================================

PartitionedShuffleOutput radixShuffle(
    const BlockStream & input,
    const PayloadSchema & out_schema,
    const std::vector<size_t> & in_col_index,
    const RadixConfig & cfg,
    size_t threads)
{
    if (cfg.pass_bits.empty())
        throw std::invalid_argument("radixShuffle: at least one pass required");
    const uint8_t total = cfg.totalBits();
    if (total == 0 || total > 32)
        throw std::invalid_argument("radixShuffle: total partition bits must be in [1, 32]");
    if (out_schema.types.size() != in_col_index.size())
        throw std::invalid_argument("radixShuffle: out_schema and in_col_index size mismatch");

    uint8_t consumed = 0;
    const uint8_t b0 = cfg.pass_bits[0];
    const uint32_t shift0 = static_cast<uint32_t>(64u - b0);
    PartitionedShuffleOutput result = shufflePassFromBlockStream(input, out_schema, in_col_index, size_t{1} << b0, shift0, threads);
    consumed = b0;

    for (size_t pass = 1; pass < cfg.pass_bits.size(); ++pass)
    {
        const uint8_t bi = cfg.pass_bits[pass];
        const uint32_t shift = static_cast<uint32_t>(64u - consumed - bi);
        result = shufflePassFromChains(result, size_t{1} << bi, shift, threads);
        consumed = static_cast<uint8_t>(consumed + bi);
    }

    return result;
}


PartitionedShuffleOutput radixShuffle(const BlockStream & input, const RadixConfig & cfg, size_t threads)
{
    std::vector<size_t> identity(input.schema.types.size());
    for (size_t c = 0; c < identity.size(); ++c)
        identity[c] = c;
    return radixShuffle(input, input.schema, identity, cfg, threads);
}

}
