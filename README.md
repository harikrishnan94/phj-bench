# phj-bench

Standalone benchmark harness comparing two parallel hash-join strategies on
a shared CH-style primitive hashtable, so the difference attributable to
partitioning strategy is isolated.

- **CHJ** — concurrent hash join (ClickHouse-style): a 256-way two-level
  hashtable. The top 8 bits of the key hash select the sub-table; each
  sub-table has its own mutex and resizes independently. Parallel build,
  parallel probe, global barrier between the two.
- **PHJ** — radix-partitioned hash join: build and probe inputs are
  radix-partitioned into per-partition column-major buffer chains.
  Post-shuffle, partitions form a work-stealing queue. Each worker claims a
  partition, builds its hashtable, probes it, materialises matched output,
  then claims the next.

Both schemes use the same open-addressing `JoinHashTable` primitive (cells
of `(key, RowRefCell)`), the same `intHash64` mix function, and the same
`RowRefCell` row reference. Payload columns live in externally-owned
column-major storage; the hashtable carries only row indices.

## Layout

| Header                                | Role                                              |
|---------------------------------------|---------------------------------------------------|
| `src/Types.h`                         | `PayloadType`, `UInt128`, `RowRefCell`            |
| `src/Hash.h`                          | `intHash64` integer mix                           |
| `src/Timer.h`                         | `steady_clock` helpers                            |
| `src/Threading.h`                     | `parallelRun(threads, fn)` spawn/join helper      |
| `src/ColumnStorage.h`                 | Flat `ColumnSet` + 16 KiB-buffer chains + cursors |
| `src/HashTable.h`                     | Open-addressing primitive, `reserve`, `insert`, `find` |
| `src/TwoLevelHashTable.h`             | 256-way wrapper with per-sub-table mutex          |
| `src/RadixPartition.{h,cpp}`          | Multi-pass radix shuffle (column-major chains)    |
| `src/DataGen.{h,cpp}`                 | Parallel synthetic build/probe generator          |
| `src/CHJ.{h,cpp}`                     | CHJ scheme + per-phase timing                     |
| `src/PHJ.{h,cpp}`                     | PHJ scheme with work-stealing partition queue     |
| `src/Reference.{h,cpp}`               | `std::unordered_map`-based correctness reference  |
| `src/CLI.{h,cpp}`                     | Argument parser                                   |
| `src/Report.{h,cpp}`                  | Console table + optional CSV writer               |
| `src/Main.cpp`                        | Driver: generate once, loop reps, validate, report|

## Build

No external dependencies beyond the C++ standard library and system
threads. C++23 is required.

```
meson setup build
meson compile -C build
```

Binary: `build/src/phj-bench`.

## Run

```
build/src/phj-bench \
    --scheme both \
    --build-rows 10000000 \
    --probe-rows 10000000 \
    --build-payload-schema u32,u64 \
    --probe-payload-schema u32,u64,u128 \
    --threads 8 \
    --partitions 256 \
    --reps 5 \
    --seed 42 \
    --csv runs.csv \
    --check
```

See `build/src/phj-bench --help` for the full set of options.

The console table is printed unconditionally. CSV is written only when
`--csv <path>` is provided; the file is appended to if it already exists
and the header is emitted only when the file is new.

## Measurement model

Each rep is an independent end-to-end execution: every transient state —
partition buffer chains, hashtables, output buffer chains — is rebuilt per
rep. The original generated input columns are not regenerated. Data
generation and (optional) correctness validation run once around the rep
loop.

For each (scheme, rep) the harness records both wall ms and ns/row per
phase. The aggregation rules are:

| Phase / Scheme | Wall ms                                | ns/row                                       |
|----------------|----------------------------------------|----------------------------------------------|
| CHJ build      | clock span (build barrier)             | sum per-thread phase ns / build_rows         |
| CHJ probe      | clock span (probe completion)          | sum per-thread phase ns / probe_rows         |
| PHJ shuffles   | clock span                             | wall ns × threads / rows                     |
| PHJ build      | max per-worker accumulator             | sum per-worker phase ns / build_rows         |
| PHJ probe      | max per-worker accumulator             | sum per-worker phase ns / probe_rows         |

PHJ build and probe interleave per partition under work-stealing; the
harness keeps per-thread per-phase accumulators inside the worker loop so
build and probe stay isolated despite interleaving.

The console table summarises median, min, and max across reps for each
metric. The CSV emits one row per (scheme, rep).

## Correctness

`--check` runs an in-process `std::unordered_map` reference join on the
same generated inputs, serialises both the reference and the actual output
into a multiset of byte-encoded rows, sorts each, and reports the first
divergence. Validation is intended for sanity checks at small row counts;
the reference is single-threaded and materialises the full result set.

## Style

Source files follow the ClickHouse `.clang-format` and `.clang-tidy`
configurations checked in at the repo root. Both files are copied verbatim
from `ClickHouse/ClickHouse@master`.

```
for f in src/*.h src/*.cpp; do clang-format-22 -i "$f"; done
for f in src/*.cpp; do clang-tidy-22 -p build "$f" --config-file=.clang-tidy; done
```

## Non-goals (v1)

NUMA awareness; multi-socket; software write-combining buffers; swappable
hash; VTune ITT; non-uniform key distributions; non-inner joins; match
rates other than 1.0; spill-to-disk; row-major inline payload storage.
