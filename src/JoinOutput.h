#pragma once

#include <cstddef>
#include <vector>

#include "ColumnStorage.h"


namespace phj
{

/// Per-worker append-only column-major output buffers. Each matched row
/// pair produced by probe materialisation writes one entry to the keys
/// chain, one entry per left (build) payload chain, and one entry per
/// right (probe) payload chain.
struct OutputWorker
{
    KeyBufferChain keys;
    std::vector<PayloadBufferChain> left;
    std::vector<PayloadBufferChain> right;

    void init(const PayloadSchema & l, const PayloadSchema & r)
    {
        left.clear();
        right.clear();
        left.reserve(l.types.size());
        right.reserve(r.types.size());
        for (auto t : l.types)
            left.emplace_back(t);
        for (auto t : r.types)
            right.emplace_back(t);
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
            s += w.keys.rows;
        return s;
    }
};


struct PhaseTiming
{
    double wall_ms = 0.0;
    double ns_per_row = 0.0;
};

}
