# phj-bench

Standalone benchmark harness comparing three parallel hash-join strategies
on a shared CH-style primitive hashtable, so the difference attributable
to partitioning strategy is isolated.

- **CHJ** — concurrent hash join (ClickHouse-style): a 256-way two-level
  hashtable. The top 8 bits of the key hash select the sub-table; each
  sub-table has its own mutex and resizes independently. Parallel build,
  parallel probe, global barrier between the two.
- **PHJ** — radix-partitioned hash join: build and probe inputs are
  radix-partitioned into per-partition `OutBlock` chains. Post-shuffle,
  partitions form a work-stealing queue. Each worker claims a partition,
  builds its hashtable, probes it block-by-block, materialises matched
  output, then claims the next.
- **PHJ-BEP** — best-effort partitioning hash join (Zukowski, Héman,
  Boncz; *Architecture-Conscious Hashing*, DaMoN 2006, §3.1,
  Algorithm 1). Build side is identical to PHJ (radix-shuffled to full
  leaf depth via `--partitions` × `--passes`), but per-leaf HT
  construction is deferred. Probe is per-thread, single-pass over its
  slice of probe input. Each incoming block is scattered through pass 1
  only into per-worker unrefined chains. At every input-block boundary
  the worker checks whether its buffered probe bytes have reached the
  per-worker budget M (`--bep-budget-mib`, default 32). If so, the
  largest buffered partition is selected; unrefined targets are force-
  refined to leaves on the spot (passes 2..N); leaf targets are built
  (lazily, once per leaf across the whole run via a `NOT_BUILT` →
  `BUILDING` → `BUILT` CAS state machine), probed, and dropped. At end
  of input the remaining non-empty partitions are drained.

All three schemes use the same open-addressing `JoinHashTable`
primitive (cells of `(key, RowRefCell)`), the same `intHash64` mix
function, and the same `RowRefCell` row reference. Build-side payloads
live in a `BlockStore`; the hashtable carries only `(block_no, row_no)`.
PHJ-BEP reuses the multi-pass radix partition operator (in `scatter_pass1`
and `refine_to_leaves` modes), the per-leaf HT construction routine,
`process_partition`, the build-shuffle code path, the output sink, and
the thread pool — it adds only the per-thread BEP probe loop, the
per-thread byte accounting, the per-leaf HT state machine, the
structural unrefined/leaf distinction on the probe side, and the BEP-
specific timing/metrics layout.

## Pipeline architecture

Operators are wired as a block-pipelined stream. The unit of work is a
`Block` of ~10K rows (`PIPELINE_BLOCK_ROWS`). Each operator consumes
blocks from its input(s) and produces blocks to its output(s),
block-by-block, without materialising the entire input upfront.

- **Data generator** emits blocks of ~10K rows on both build and probe
  sides.
- **Radix partition operator** consumes input blocks and processes
  each one as one batch via the canonical 5-phase pipeline (SIMD hash
  → histogram → pre-grow with committed `filled` → column-first
  branch-free scatter via live pointers → next batch). The output is
  per-partition `OutBlock` chains whose per-column buffers start at
  ~16 KiB and double on each new block. Type-generic column scatter
  handles all of `uint8`/`uint16`/`uint32`/`uint64`/`uint128`.
- **Build operator** for each input block: appends the block to the
  partition's (PHJ) or the shared (CHJ) `BlockStore` (recording its
  `block_no`), SIMD-vectorises `intHash64` over the block's keys, and
  batch-inserts `(key, RowRefCell{block_no, row_no})` cells into the
  hashtable.
- **Probe operator** for each input block: SIMD-vectorises
  `intHash64` over the block's keys, batched HT lookup produces an
  array of head `RowRefCell`s, multi-match chains expand into flat
  `(probe_idx, build_ref)` arrays, and the column-major projection
  loop nest `for column { for matched_row }` gathers build-side values
  via `(block_no, row_no)` and probe-side values via the probe row
  index. Output blocks of ~10K rows are emitted as they fill.

`RowRefCell = (block_no, row_no)`, 8 bytes; HT cells are 16 bytes
(`key` + ref). The build-side block store keeps a parallel next-chain
matrix for multi-match traversal.

## Layout

| Header                                | Role                                                    |
|---------------------------------------|---------------------------------------------------------|
| `src/Types.h`                         | `PayloadType`, `UInt128`, `RowRefCell`, `PayloadSchema` |
| `src/Block.h`                         | `Block`, `BlockView`, `BlockStream`, `Column`           |
| `src/BlockStore.h`                    | Per-partition / shared block store + next-chain         |
| `src/Hash.h`                          | `intHash64` + SIMD batched `intHash64Batch` (AVX-512)   |
| `src/Timer.h`                         | `steady_clock` helpers                                  |
| `src/Threading.h`                     | `parallelRun(threads, fn)` spawn/join helper            |
| `src/HashTable.h`                     | Open-addressing primitive (Cell holds `RowRefCell`)     |
| `src/TwoLevelHashTable.h`             | 256-way wrapper with per-sub-table mutex                |
| `src/RadixPartition.{h,cpp}`          | Radix shuffle + reusable per-batch scatter primitives   |
| `src/JoinOps.h`                       | Vectorised build/probe + probe-side projection          |
| `src/JoinOutput.h`                    | `OutputBlock`, `OutputWorker`, `PhaseTiming`            |
| `src/DataGen.{h,cpp}`                 | Block-emitting parallel synthetic data generator        |
| `src/CHJ.{h,cpp}`                     | CHJ scheme: shared store + vectorised build/probe       |
| `src/PHJ.{h,cpp}`                     | PHJ scheme: per-partition store + work-stealing         |
| `src/PHJBep.{h,cpp}`                  | PHJ-BEP: bounded probe buffer + lazy per-leaf HT        |
| `src/Reference.{h,cpp}`               | `std::unordered_map`-based correctness reference        |
| `src/CLI.{h,cpp}`                     | Argument parser                                         |
| `src/Report.{h,cpp}`                  | Console table + CSV writer (with `e2e_wall_ms`)         |
| `src/Main.cpp`                        | Driver: generate once, loop reps, validate, report      |

## Build

No external dependencies beyond the C++ standard library and system
threads. C++23 is required. AVX-512 (F + DQ) is auto-detected at
configure time and used for `intHash64Batch`; the scalar fallback is
selected otherwise and auto-vectorises cleanly on most modern
compilers.

```
meson setup build
meson compile -C build
```

Binary: `build/src/phj-bench`.

## Run

```
build/src/phj-bench \
    --scheme all \
    --build-rows 10000000 \
    --probe-rows 10000000 \
    --build-payload-schema u32,u64 \
    --probe-payload-schema u32,u64,u128 \
    --threads 8 \
    --partitions 256 \
    --bep-budget-mib 32 \
    --reps 5 \
    --seed 42 \
    --csv runs.csv \
    --check
```

`--scheme` accepts `chj`, `phj`, `phj-bep`, or `all`. `--bep-budget-mib`
sets the per-worker probe-side buffer budget for PHJ-BEP in mebibytes
(default 32). See `build/src/phj-bench --help` for the full set of
options.

The console table is printed unconditionally. CSV is written only when
`--csv <path>` is provided; the file is appended to if it already
exists and the header is emitted only when the file is new.

## Measurement model

Each rep is an independent end-to-end execution: every transient state
— per-partition OutBlock chains, build-side block stores, hashtables,
probe output blocks — is rebuilt per rep. The generated `BlockStream`s
are not regenerated. Data generation and (optional) correctness
validation run once around the rep loop.

For each (scheme, rep) the harness records both wall ms and ns/row per
phase. The aggregation rules are:

| Phase / Scheme                      | Wall ms                                | ns/row                                       |
|-------------------------------------|----------------------------------------|----------------------------------------------|
| CHJ build                           | clock span (build barrier)             | sum per-thread phase ns / build_rows         |
| CHJ probe                           | clock span (probe completion)          | sum per-thread phase ns / probe_rows         |
| PHJ shuffles                        | clock span                             | wall ns × threads / rows                     |
| PHJ build                           | max per-worker accumulator             | sum per-worker phase ns / build_rows         |
| PHJ probe                           | max per-worker accumulator             | sum per-worker phase ns / probe_rows         |
| PHJ-BEP build-shuffle               | clock span                             | wall ns × threads / build_rows               |
| PHJ-BEP build                       | max per-worker accumulator             | sum per-worker phase ns / build_rows         |
| PHJ-BEP probe-shuffle               | max per-worker accumulator             | sum per-worker phase ns / probe_rows         |
| PHJ-BEP probe                       | max per-worker accumulator             | sum per-worker phase ns / probe_rows         |
| PHJ-BEP eviction-overhead           | max per-worker accumulator             | sum per-worker phase ns / probe_rows         |

For PHJ-BEP, the five per-worker accumulators cover:

- `build_shuffle_ns` — global, contiguous wall span of the full multi-
  pass build-shuffle (identical to PHJ; reported as a single accumulator
  shared across all workers).
- `build_ns` — lazy per-leaf HT construction during probe, charged to
  the worker that won the CAS race for that leaf.
- `probe_shuffle_ns` — initial pass-1 scatter of incoming probe blocks
  **plus** refinement scatter of selected partitions (passes 2..N).
- `probe_ns` — `process_partition`: HT lookup + column-major output
  projection.
- `eviction_overhead_ns` — trigger checks, argmax across partitions,
  state-machine loads/CAS, `BUILDING`-skip / backoff, and buffer-chain
  teardown.

PHJ-BEP additionally reports the following per-run metrics in the
console table footer and in the CSV (the columns are blank in the CHJ
and PHJ rows):

- `bep_budget_mib` — the per-worker buffer budget (CLI input).
- `bep_evictions` — total leaf evictions across all workers.
- `bep_refinements` — total forced refinements across all workers
  (BEP's only refinement trigger is the budget check).
- `bep_peak_buffered_rows` — max across workers and across the run of
  total buffered probe rows.
- `bep_build_skip_retries` — total `BUILDING`-skip events across all
  workers.

The harness additionally records a single end-to-end wall time per
scheme per rep:

- **CHJ `e2e_wall_ms`** spans from the start of build to the end of
  probe.
- **PHJ `e2e_wall_ms`** spans from the start of build-shuffle to the
  end of probe.
- **PHJ-BEP `e2e_wall_ms`** spans from the start of build-shuffle to
  the last probe output emitted.

PHJ build and probe interleave per partition under work-stealing; the
harness keeps per-thread per-phase accumulators inside the worker loop
so build and probe stay isolated despite interleaving. PHJ-BEP further
interleaves refinement, lazy HT construction, probing, and buffer
teardown on every worker; the five accumulators above isolate each
contribution.

The console table summarises median, min, and max across reps for each
metric (including `e2e_wall_ms`). The CSV emits one row per (scheme,
rep) with column groups for build/probe, build-shuffle/probe-shuffle,
eviction-overhead, `e2e_wall_ms`, and the BEP-only per-run metrics.

## Correctness

`--check` runs an in-process `std::unordered_map` reference join on
the same generated `BlockStream`s, serialises both the reference and
the actual output into a multiset of byte-encoded rows, sorts each,
and reports the first divergence. Validation is intended for sanity
checks at small row counts; the reference is single-threaded and
materialises the full result set.

## Style

Source files follow the ClickHouse `.clang-format` and `.clang-tidy`
configurations checked in at the repo root. Both files are copied
verbatim from `ClickHouse/ClickHouse@master`.

```
for f in src/*.h src/*.cpp; do clang-format-22 -i "$f"; done
for f in src/*.cpp; do clang-tidy-22 -p build "$f" --config-file=.clang-tidy; done
```

## Non-goals (v1)

NUMA awareness; multi-socket; software write-combining buffers
(SWWC paths from the radix-partition reference file are intentionally
not ported); swappable hash; VTune ITT; non-uniform key distributions;
non-inner joins; match rates other than 1.0; spill-to-disk; row-major
inline payload storage.
