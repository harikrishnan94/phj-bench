#pragma once

#include <cstdint>

#include "Block.h"


namespace phj
{

/// Generate the build side as a stream of blocks of `PIPELINE_BLOCK_ROWS`
/// rows each (the final block may be short). Keys are
/// `intHash64(seed_build + i)` for row `i`, giving a uniform
/// distribution and essentially-zero duplicate probability across
/// normal benchmark sizes. Payload values are derived from independent
/// hash mixes per column and truncated to the column type.
///
/// `block_rows` selects the pipeline block size (defaults to
/// `PIPELINE_BLOCK_ROWS`).
BlockStream
generateBuild(size_t rows, const PayloadSchema & schema, uint64_t seed, size_t threads, size_t block_rows = PIPELINE_BLOCK_ROWS);


/// Generate the probe side as a stream of blocks of `block_rows` rows
/// each. Probe keys are drawn uniformly from the build key set,
/// producing a 1.0 match rate by construction. `build_stream` must be
/// the BlockStream returned by `generateBuild`.
BlockStream generateProbe(
    size_t rows,
    const PayloadSchema & schema,
    uint64_t seed,
    const BlockStream & build_stream,
    size_t threads,
    size_t block_rows = PIPELINE_BLOCK_ROWS);

}
