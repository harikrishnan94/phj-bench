#pragma once

#include <cstddef>

#include "Block.h"
#include "JoinOutput.h"


namespace phj
{

struct ChjResult
{
    PhaseTiming build;
    PhaseTiming probe;
    double e2e_wall_ms = 0.0;
    size_t peak_mem_bytes = 0;
    JoinOutput output;
};


/// Concurrent hash join. A 256-way two-level hashtable is built in
/// parallel by `threads` workers, each owning a slice of build blocks.
/// The build store is shared across all sub-tables; concurrent block
/// appends serialise on the store's mutex. Probe materialises matched
/// pairs into per-worker output blocks (~10K rows each).
ChjResult runCHJ(const BlockStream & build, const BlockStream & probe, size_t threads);

}
