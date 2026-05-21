#pragma once

#include <cstddef>

#include "ColumnStorage.h"
#include "JoinOutput.h"


namespace phj
{

struct ChjResult
{
    PhaseTiming build;
    PhaseTiming probe;
    JoinOutput output;
};


/// Concurrent hash join. A 256-way two-level hashtable is built in parallel
/// by `threads` workers, each owning a slice of build rows. The build/probe
/// barrier is the std::thread join at the end of build. Probe materialises
/// matched pairs into per-worker column-major output chains.
ChjResult runCHJ(const ColumnSet & build_cs, const ColumnSet & probe_cs, size_t threads);

}
