#pragma once

#include <cstddef>
#include <vector>

#include "Block.h"
#include "Types.h"


namespace phj
{

/// Probe-side output: one composite block of matched rows. Holds the
/// join key column, the left (build-side) payload columns, and the
/// right (probe-side) payload columns, all column-major and all `rows`
/// long. Probe emits these block-by-block when the in-progress block
/// fills to ~10K rows.
struct OutputBlock
{
    size_t rows = 0;
    std::vector<uint64_t> keys;
    std::vector<Column> left;
    std::vector<Column> right;

    void init(const PayloadSchema & left_schema, const PayloadSchema & right_schema, size_t capacity)
    {
        rows = 0;
        keys.assign(capacity, 0);
        left.assign(left_schema.types.size(), {});
        for (size_t c = 0; c < left_schema.types.size(); ++c)
        {
            left[c].type = left_schema.types[c];
            left[c].data.assign(capacity * payloadTypeSize(left_schema.types[c]), std::byte{});
        }
        right.assign(right_schema.types.size(), {});
        for (size_t c = 0; c < right_schema.types.size(); ++c)
        {
            right[c].type = right_schema.types[c];
            right[c].data.assign(capacity * payloadTypeSize(right_schema.types[c]), std::byte{});
        }
    }
};


/// Per-worker append-only stream of probe-output blocks. The probe
/// operator maintains one in-progress `OutputBlock` of capacity
/// `PIPELINE_BLOCK_ROWS`, flushing it onto `blocks` whenever it fills.
/// At end-of-stream the in-progress block is flushed (truncated to its
/// actual filled row count).
struct OutputWorker
{
    std::vector<OutputBlock> blocks;

    [[nodiscard]] size_t totalRows() const noexcept
    {
        size_t s = 0;
        for (const auto & b : blocks)
            s += b.rows;
        return s;
    }
};


struct JoinOutput
{
    PayloadSchema left_schema;
    PayloadSchema right_schema;
    std::vector<OutputWorker> workers;

    [[nodiscard]] size_t totalRows() const noexcept
    {
        size_t s = 0;
        for (const auto & w : workers)
            s += w.totalRows();
        return s;
    }
};


struct PhaseTiming
{
    double wall_ms = 0.0;
    double ns_per_row = 0.0;
};

}
