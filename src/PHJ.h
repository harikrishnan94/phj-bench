#pragma once

#include <cstddef>

#include "ColumnStorage.h"
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
    JoinOutput output;
};


/// Radix-partitioned hash join with work-stealing per-partition scheduling.
/// `cfg` configures the partition fanout (and optional multi-pass) for both
/// the build and probe shuffles. Build and probe run interleaved on each
/// worker — claim a partition, build its HT, probe it, materialise output,
/// then claim the next partition. Per-worker per-phase ns accumulators
/// drive the phase-time aggregation described in the spec.
PhjResult runPHJ(const ColumnSet & build_cs, const ColumnSet & probe_cs, const RadixConfig & cfg, size_t threads);

}
