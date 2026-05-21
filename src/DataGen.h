#pragma once

#include <cstdint>

#include "ColumnStorage.h"


namespace phj
{

/// Generate the build side: a uniform 64-bit key column plus payload columns
/// according to `schema`. Keys are `intHash64(seed_build ^ i)` for row i,
/// which gives a uniform distribution and essentially-zero duplicate
/// probability across normal benchmark sizes. Payload values are derived
/// from independent hash mixes per column and truncated to the column type
/// so the values are non-degenerate.
ColumnSet generateBuild(size_t rows, const PayloadSchema & schema, uint64_t seed, size_t threads);


/// Generate the probe side. Probe keys are drawn uniformly from the build
/// key set (`build_keys[intHash64(seed_probe ^ i) % build_rows]`). This
/// produces a 1.0 match rate by construction. Payload values are derived
/// from independent hash mixes per column.
ColumnSet
generateProbe(size_t rows, const PayloadSchema & schema, uint64_t seed, const uint64_t * build_keys, size_t build_rows, size_t threads);

}
