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
    /// checks, `BUILDING`-skip / backoff, cooperative-drain yields, and
    /// buffer-chain teardown.
    PhaseTiming build_shuffle;
    PhaseTiming build;
    PhaseTiming probe_shuffle;
    PhaseTiming probe;
    PhaseTiming eviction_overhead;

    double e2e_wall_ms = 0.0;
    size_t peak_mem_bytes = 0;
    JoinOutput output;

    /// BEP-specific per-run metrics. Empty in PHJ rows; populated here.
    size_t bep_budget_mib = 0;
    size_t bep_evictions = 0;
    size_t bep_refinements = 0;
    /// Maximum GLOBAL probe-side buffered bytes observed during the
    /// run. Equals `sum_workers(unrefined capacity) +
    /// total_published_leaf_capacity` at peak — directly comparable to
    /// `bep_budget_mib`. Reported as `bep_peak_mib` in the results.
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
///   1. Per-worker scatter + cooperative eviction (parallelRun #1).
///      Each worker processes its slice of probe input. Each incoming
///      block is scattered through pass 1 only (top `pass_bits[0]`
///      bits) into per-worker `unrefined` chains. After each input
///      block, two independent hysteresis triggers fire:
///
///        - per-worker unrefined trigger: if own unrefined capacity
///          crosses `1/4 * M_bytes`, force-refine the largest
///          unrefined pass-1 chain (through passes 2..N) into the
///          SHARED leaf chains, until own unrefined drops below
///          `1/8 * M_bytes`.
///        - global leaf trigger: if `published_leaf_capacity / threads`
///          crosses `3/4 * M_bytes`, the worker enters a cooperative
///          drain loop. Each iteration picks the globally largest
///          published leaf, CAS-claims it for HT construction if
///          NOT_BUILT, steals the entire shared chain under a per-leaf
///          mutex, builds + probes + drops. Drain continues until the
///          per-worker share drops below `5/8 * 3/4 * M_bytes`. All
///          workers observe the same global state, so all 16 evict in
///          parallel on distinct leaves naturally.
///
///      A worker that finishes its input slice waits at a cooperative
///      end-of-slice drain — it keeps draining global leaves until
///      every other worker has also exhausted its input slice — to
///      avoid letting fast workers idle at the parallelRun barrier
///      while slow workers absorb the residual eviction work alone.
///      Finally each worker refines its leftover unrefined pass-1
///      buckets and exits.
///
///   2. Work-stealing drain (parallelRun #2). Each leaf is claimed by
///      exactly one worker via a shared `fetch_add`. The claiming
///      worker steals the residual shared chain (no merge required —
///      pass 1 deposited blocks directly into the shared structure)
///      and — if mid-stream eviction never built the leaf's HT —
///      builds it now and probes end-to-end. At large budgets where
///      no leaf is ever evicted mid-stream, the drain is structurally
///      equivalent to PHJ's post-shuffle build+probe loop.
///
/// `bep_budget_mib` is the GLOBAL memory budget in mebibytes, shared
/// across all worker threads; only probe input buffer bytes (across
/// both unrefined and leaf chains) count toward it. Internally each
/// worker's view of the budget is `bep_budget_mib / threads` MiB, split
/// 1/4 unrefined / 3/4 leaf share as described above. With a budget
/// large enough to hold the entire probe stream, the mid-stream
/// eviction loop never fires and the drain reduces to PHJ's
/// build+probe loop.
PhjBepResult
runPhjBep(const BlockStream & build, const BlockStream & probe, const RadixConfig & cfg, size_t threads, size_t bep_budget_mib);

}
