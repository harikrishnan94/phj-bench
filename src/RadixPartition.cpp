#include "RadixPartition.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

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


/// Initial per-column buffer rows for a new `OutBlock`. Sized so that
/// the column with the largest element type yields a ~16 KiB buffer.
/// Smaller-element columns share the same row count and therefore
/// occupy proportionally less memory.
constexpr size_t TARGET_INITIAL_BUFFER_BYTES = 16 * 1024;
constexpr size_t MAX_OUT_BLOCK_ROWS = 65'536;

[[nodiscard]] size_t initialOutBlockRows(const PayloadSchema & schema) noexcept
{
    size_t max_element_size = sizeof(uint64_t);
    for (auto t : schema.types)
        max_element_size = std::max(max_element_size, payloadTypeSize(t));
    return std::max<size_t>(1, TARGET_INITIAL_BUFFER_BYTES / max_element_size);
}


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


/// Per-(thread, partition) growable output. The chain matches the
/// reference's structure (singly-linked block list); we keep it as a
/// vector here for cache-friendlier downstream traversal.
struct ThreadChain
{
    std::vector<OutBlock> blocks;
    OutBlock * cur = nullptr;
    size_t next_cap;

    void init(size_t initial_cap)
    {
        blocks.clear();
        cur = nullptr;
        next_cap = initial_cap;
    }

    /// Allocate a new OutBlock big enough to hold at least
    /// `min_required` rows. The doubling-capacity scheme (start at
    /// ~16 KiB per buffer, double each new block, cap at
    /// MAX_OUT_BLOCK_ROWS for the doubling progression) is preserved,
    /// but the FIRST allocation for this chain is bumped up if the
    /// impending batch's per-partition load exceeds `next_cap`. This
    /// matters when the pipeline batch size is large relative to the
    /// partition fanout (e.g., pass-1 with P=4 and 10K-row batches,
    /// or a heavily-skewed bucket in a later pass with up-to-65K-row
    /// input OutBlocks).
    void grow(const PayloadSchema & schema, size_t min_required)
    {
        size_t cap = std::max(next_cap, min_required);
        blocks.emplace_back();
        OutBlock & blk = blocks.back();
        initOutBlock(blk, schema, cap);
        cur = &blocks.back();
        next_cap = std::min<size_t>(cap * 2, MAX_OUT_BLOCK_ROWS);
    }
};


struct ThreadShard
{
    /// chains[partition]
    std::vector<ThreadChain> chains;

    void init(size_t partitions, const PayloadSchema & schema)
    {
        const size_t initial_cap = initialOutBlockRows(schema);
        chains.assign(partitions, {});
        for (auto & c : chains)
            c.init(initial_cap);
    }
};


/// One pass of the radix operator. Iterates the input blocks assigned
/// to thread `tid`, dispatches the 5-phase per-batch pipeline against
/// the thread's `ThreadShard`.
void runPassOverBlocks(
    const std::vector<const Block *> & input_blocks,
    const std::vector<size_t> & in_col_index,
    const PayloadSchema & out_schema,
    uint32_t shift,
    size_t parts,
    ThreadShard & shard)
{
    const size_t K = out_schema.types.size();
    const uint64_t mask = parts - 1;
    const std::vector<size_t> in_sizes = [&]
    {
        std::vector<size_t> v(K);
        for (size_t c = 0; c < K; ++c)
            v[c] = payloadTypeSize(out_schema.types[c]);
        return v;
    }();
    std::vector<ScatterFn> scatters(K);
    for (size_t c = 0; c < K; ++c)
        scatters[c] = scatterFor(out_schema.types[c]);

    /// Working arrays sized to the maximum batch (input block) we
    /// will see. Allocated once per pass per thread, not per batch.
    size_t max_rows = 0;
    for (const auto * blk : input_blocks)
        max_rows = std::max(max_rows, blk->rows);

    std::vector<uint64_t> hashes(max_rows);
    std::vector<uint32_t> pids(max_rows);
    std::vector<uint32_t> local_hist(parts, 0);
    /// Live key write pointers per partition.
    std::vector<uint64_t *> key_ptrs(parts, nullptr);
    /// Live column write pointers, flattened as `ptrs_flat[c * parts + p]`.
    PtrsFlat ptrs_flat(K * parts, nullptr);

    for (const Block * blk : input_blocks)
    {
        const size_t rows = blk->rows;
        if (rows == 0)
            continue;

        /// Phase 1: SIMD hash over keys -> pids[].
        intHash64Batch(blk->keyData(), rows, hashes.data());
        for (size_t j = 0; j < rows; ++j)
            pids[j] = static_cast<uint32_t>((hashes[j] >> shift) & mask);

        /// Phase 2: histogram.
        std::memset(local_hist.data(), 0, parts * sizeof(uint32_t));
        for (size_t j = 0; j < rows; ++j)
            ++local_hist[pids[j]];

        /// Phase 3: pre-grow + commit `filled` + live-pointer setup.
        for (size_t p = 0; p < parts; ++p)
        {
            if (local_hist[p] == 0)
                continue;
            auto & chain = shard.chains[p];
            if (chain.cur == nullptr || chain.cur->rows + local_hist[p] > chain.cur->capacity)
                chain.grow(out_schema, local_hist[p]);
            OutBlock & ob = *chain.cur;
            const size_t base = ob.rows;
            ob.rows = base + local_hist[p];
            key_ptrs[p] = ob.keys.data() + base;
            for (size_t c = 0; c < K; ++c)
                ptrs_flat[c * parts + p] = ob.payloads[c].raw() + base * in_sizes[c];
        }

        /// Phase 4: column-first branch-free scatter, live pointer is
        /// the position. Keys first, then each payload column via the
        /// per-type scatter function.
        const uint64_t * key_src = blk->keyData();
        for (size_t j = 0; j < rows; ++j)
            *key_ptrs[pids[j]]++ = key_src[j];
        for (size_t c = 0; c < K; ++c)
        {
            std::byte ** col_ptrs = ptrs_flat.data() + c * parts;
            const std::byte * src = blk->payloads[in_col_index[c]].raw();
            scatters[c](col_ptrs, pids.data(), src, rows);
        }
    }
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
            std::vector<const Block *> my_blocks;
            my_blocks.reserve(end - start);
            for (size_t b = start; b < end; ++b)
                my_blocks.push_back(&input.blocks[b]);
            runPassOverBlocks(my_blocks, in_col_index, out_schema, shift, parts, shards[tid]);
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

    /// Each thread processes a disjoint set of input partitions; the
    /// sub-partition the thread emits encodes as `ip * sub_parts + s`.
    parallelRun(
        threads,
        [&](size_t tid)
        {
            ThreadShard & shard = shards[tid];
            const size_t sub_mask = sub_parts - 1;
            for (size_t ip = tid; ip < in_parts; ip += threads)
            {
                /// 5-phase pipeline is inlined here over OutBlocks
                /// (each one is a batch) rather than wrapping them as
                /// Blocks; same algorithmic shape as the flat-stream
                /// pass, just sourcing from the previous pass's chain.
                const std::vector<size_t> in_sizes = [&]
                {
                    std::vector<size_t> v(n_payload);
                    for (size_t c = 0; c < n_payload; ++c)
                        v[c] = payloadTypeSize(input.schema.types[c]);
                    return v;
                }();
                std::vector<ScatterFn> scatters(n_payload);
                for (size_t c = 0; c < n_payload; ++c)
                    scatters[c] = scatterFor(input.schema.types[c]);

                /// Pre-size working arrays to the largest input OutBlock.
                size_t max_rows = 0;
                for (const OutBlock & ob : input.chains[ip].blocks)
                    max_rows = std::max(max_rows, ob.rows);

                std::vector<uint64_t> hashes(max_rows);
                std::vector<uint32_t> pids(max_rows);
                std::vector<uint32_t> local_hist(out_parts, 0);
                std::vector<uint64_t *> key_ptrs(out_parts, nullptr);
                PtrsFlat ptrs_flat(n_payload * out_parts, nullptr);

                /// Each emitted sub-partition's index is `ip * sub_parts + s`.
                const size_t base_idx = ip * sub_parts;

                for (const OutBlock & ob : input.chains[ip].blocks)
                {
                    const size_t rows = ob.rows;
                    if (rows == 0)
                        continue;

                    intHash64Batch(ob.keys.data(), rows, hashes.data());
                    for (size_t j = 0; j < rows; ++j)
                        pids[j] = static_cast<uint32_t>(base_idx + ((hashes[j] >> shift) & sub_mask));

                    std::memset(local_hist.data(), 0, out_parts * sizeof(uint32_t));
                    for (size_t j = 0; j < rows; ++j)
                        ++local_hist[pids[j]];

                    for (size_t p = 0; p < out_parts; ++p)
                    {
                        if (local_hist[p] == 0)
                            continue;
                        auto & chain = shard.chains[p];
                        if (chain.cur == nullptr || chain.cur->rows + local_hist[p] > chain.cur->capacity)
                            chain.grow(input.schema, local_hist[p]);
                        OutBlock & cur = *chain.cur;
                        const size_t base = cur.rows;
                        cur.rows = base + local_hist[p];
                        key_ptrs[p] = cur.keys.data() + base;
                        for (size_t c = 0; c < n_payload; ++c)
                            ptrs_flat[c * out_parts + p] = cur.payloads[c].raw() + base * in_sizes[c];
                    }

                    const uint64_t * key_src = ob.keys.data();
                    for (size_t j = 0; j < rows; ++j)
                        *key_ptrs[pids[j]]++ = key_src[j];
                    for (size_t c = 0; c < n_payload; ++c)
                    {
                        std::byte ** col_ptrs = ptrs_flat.data() + c * out_parts;
                        const std::byte * src = ob.payloads[c].raw();
                        scatters[c](col_ptrs, pids.data(), src, rows);
                    }
                }
            }
        });

    return collectChains(std::move(shards), input.schema);
}

}


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
