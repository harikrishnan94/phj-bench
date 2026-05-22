#pragma once

#include <cstddef>

#include "Block.h"
#include "JoinOutput.h"
#include "RadixPartition.h"


namespace phj
{

struct PhjBepResult
{
    /// Same five phases as PHJ, plus an `eviction_overhead` accumulator
    /// covering trigger checks, argmax across partitions, state-machine
    /// checks, `BUILDING`-skip / backoff, and buffer-chain teardown.
    PhaseTiming build_shuffle;
    PhaseTiming build;
    PhaseTiming probe_shuffle;
    PhaseTiming probe;
    PhaseTiming eviction_overhead;

    double e2e_wall_ms = 0.0;
    JoinOutput output;

    /// BEP-specific per-run metrics. Empty in PHJ rows; populated here.
    size_t bep_budget_mib = 0;
    size_t bep_evictions = 0;
    size_t bep_refinements = 0;
    /// Maximum per-worker probe-side buffered bytes observed during the run.
    size_t bep_peak_bytes = 0;
    size_t bep_build_skip_retries = 0;
};


/// Best-effort partitioning hash join (Zukowski et al., DaMoN 2006).
///
/// Build phase: identical to PHJ — `radixShuffle` runs at full leaf
/// depth via `cfg`. Per-leaf HT construction is deferred to probe time
/// (lazy build under a `NOT_BUILT` -> `BUILDING` -> `BUILT` state
/// machine, at most one construction per leaf across the whole run).
///
/// Probe phase, in three subphases:
///
///   1. Per-worker scatter + mid-stream eviction (parallelRun #1).
///      Each worker processes its slice of probe input. Each incoming
///      block is scattered through pass 1 only (top `pass_bits[0]`
///      bits) into per-worker unrefined chains. After each input block,
///      if total per-worker buffered bytes >= `M_bytes`, the eviction
///      loop fires: pick the largest buffered partition; if unrefined,
///      force-refine it through passes 2..N (cascading to leaves) and
///      re-select; if a leaf, `ensure_built` + `process_partition` +
///      drop. Eviction selection skips `BUILDING` leaves; if all
///      eligible candidates are mid-build, brief backoff and re-check.
///      At end of input each worker refines its remaining unrefined
///      pass-1 buckets into its own per-leaf chains, then exits.
///
///   2. Work-stealing drain (parallelRun #2). Each leaf is claimed by
///      exactly one worker via a shared `fetch_add`. The claiming
///      worker lazily cross-merges that leaf's per-worker chains into
///      a single chain (moves only), then — if mid-stream eviction
///      never built the leaf's HT — builds it now and probes the
///      merged chain end-to-end. This restores PHJ's single-owner-
///      per-partition build+probe semantics for everything that
///      survived mid-stream eviction: at large budgets where no leaf
///      is ever evicted mid-stream, the drain is structurally
///      equivalent to PHJ's post-shuffle build+probe loop.
///
/// `bep_budget_mib` is the per-worker memory budget in mebibytes; only
/// probe input buffer bytes (across both unrefined and leaf chains)
/// count toward it. With a budget large enough to hold each worker's
/// entire probe slice, the mid-stream eviction loop never fires and
/// the drain reduces to PHJ's build+probe loop modulo the cross-worker
/// merge (which is move-only, no data copies).
PhjBepResult
runPhjBep(const BlockStream & build, const BlockStream & probe, const RadixConfig & cfg, size_t threads, size_t bep_budget_mib);

}
