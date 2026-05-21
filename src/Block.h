#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Types.h"


namespace phj
{

/// A single column inside a Block. Type-erased: the column knows its
/// element type and stores `rows * elementSize()` bytes of contiguous
/// payload values. All payload columns in the pipeline are column-major.
struct Column
{
    PayloadType type = PayloadType::UInt8;
    std::vector<std::byte> data;

    [[nodiscard]] size_t elementSize() const noexcept { return payloadTypeSize(type); }
    [[nodiscard]] std::byte * raw() noexcept { return data.data(); }
    [[nodiscard]] const std::byte * raw() const noexcept { return data.data(); }
};


/// Non-owning view of a block: rows count plus pointers to a contiguous
/// key column and a contiguous range of payload columns. Used by the
/// probe operator to consume both Pipeline `Block`s (from the probe
/// input stream) and `OutBlock`s (from the radix shuffle output) under
/// a single interface, without forcing a copy of the latter into the
/// former.
struct BlockView
{
    size_t rows = 0;
    const uint64_t * keys = nullptr;
    const Column * payloads = nullptr;
    size_t n_payloads = 0;
};


/// A pipeline block: a uint64 key column plus N payload columns, all
/// column-major, all `rows` long. Blocks are the unit of work flowing
/// between operators. The data generator emits them, the radix
/// partition operator consumes them as batches and produces per-
/// partition `OutBlock` chains, and the build/probe operators consume
/// them one at a time.
struct Block
{
    size_t rows = 0;
    std::vector<uint64_t> keys;
    std::vector<Column> payloads;

    void init(const PayloadSchema & schema, size_t row_count)
    {
        rows = row_count;
        keys.assign(row_count, 0);
        payloads.assign(schema.types.size(), {});
        for (size_t c = 0; c < schema.types.size(); ++c)
        {
            payloads[c].type = schema.types[c];
            payloads[c].data.assign(row_count * payloadTypeSize(schema.types[c]), std::byte{});
        }
    }

    [[nodiscard]] const uint64_t * keyData() const noexcept { return keys.data(); }
    [[nodiscard]] uint64_t * keyData() noexcept { return keys.data(); }

    [[nodiscard]] BlockView view() const noexcept { return {rows, keys.data(), payloads.data(), payloads.size()}; }
};


/// A flat stream of blocks held in memory. Used as the materialised
/// output of the data generator and as the input to every operator that
/// is parameterised by an input stream. Different threads can read
/// non-overlapping slices `[blockStart, blockEnd)` of the same stream
/// concurrently without synchronisation.
struct BlockStream
{
    PayloadSchema schema;
    std::vector<Block> blocks;
    size_t total_rows = 0;
};

}
