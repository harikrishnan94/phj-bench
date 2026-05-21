#pragma once

#include <cstdint>
#include <vector>

#include "ColumnStorage.h"
#include "Types.h"


namespace phj
{

/// Multi-pass radix configuration. `pass_bits` lists the number of hash bits
/// consumed at each pass; the sum is the total log2 of the partition count.
/// Single-pass is `pass_bits = {N}`; two-pass is `pass_bits = {N1, N2}`.
/// All bits come from the TOP of the 64-bit hash; the highest bits are
/// consumed first so the lower bits remain well-distributed for the
/// in-partition hashtable.
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


/// Output of the radix shuffle. Per `(partition, thread)` we have one
/// `ThreadPartitionBuffers` whose chains follow the spec layout: each
/// column owns its own append-only 16 KiB-per-buffer chain, append order
/// consistent across columns within a partition.
struct PartitionedShuffleOutput
{
    size_t partitions = 0;
    size_t threads = 0;
    PayloadSchema schema;
    /// data[partition][thread]
    std::vector<std::vector<ThreadPartitionBuffers>> data;
    std::vector<size_t> partition_rows;
};


/// Run a radix shuffle. Inputs:
///   - flat key column of `rows` keys
///   - `in_payloads` is the input side's full payload set (column-major)
///   - `out_schema` describes which payload types end up in the output
///   - `out_to_in_col` maps each output payload column to its input column
///     index in `in_payloads`. This makes the operator's column-subset
///     configurable: pass the build-side payloads to shuffle for the
///     build phase and the probe-side payloads for the probe phase.
///   - `cfg` lists the bits per pass.
///   - `threads` is the worker count (same as data-gen / build / probe).
PartitionedShuffleOutput radixShuffle(
    const uint64_t * keys,
    size_t rows,
    const std::vector<const PayloadColumn *> & in_payloads,
    const PayloadSchema & out_schema,
    const std::vector<size_t> & out_to_in_col,
    const RadixConfig & cfg,
    size_t threads);

}
