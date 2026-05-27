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


/// Concurrent hash join following ClickHouse's ConcurrentHashJoin design.
///
/// `min(threads, 256)` independent slots (rounded to a power-of-2) are
/// created, each with its own BlockStore and JoinHashTable. During build
/// every worker scatters its input blocks across slots by low hash bits;
/// a slot's mutex is held only while one sub-block is being inserted, so
/// workers building into different slots proceed in parallel. Probe is
/// fully lock-free: each probe block is scattered by the same hash bits
/// and each slot's table is queried independently.
ChjResult runCHJ(const BlockStream & build, const BlockStream & probe, size_t threads);

}
