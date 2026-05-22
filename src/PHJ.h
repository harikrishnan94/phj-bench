#pragma once

#include <cstddef>

#include "Block.h"
#include "JoinOutput.h"
#include "RadixPartition.h"


namespace phj
{

struct PhjResult
{
    PhaseTiming build_shuffle;
    PhaseTiming build;
    PhaseTiming probe_shuffle;
    PhaseTiming probe;
    double e2e_wall_ms = 0.0;
    size_t peak_mem_bytes = 0;
    JoinOutput output;
};


/// Radix-partitioned hash join with work-stealing per-partition
/// scheduling. `cfg` configures the partition fanout (and optional
/// multi-pass) for both the build and probe shuffles. Build and probe
/// run interleaved on each worker — claim a partition, build its HT
/// from the partition's `BlockStore`, probe it block-by-block,
/// materialise output, then claim the next partition.
PhjResult runPHJ(const BlockStream & build, const BlockStream & probe, const RadixConfig & cfg, size_t threads);

}
