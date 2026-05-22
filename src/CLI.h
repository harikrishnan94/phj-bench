#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "RadixPartition.h"
#include "Types.h"


namespace phj
{

enum class SchemeChoice : uint8_t
{
    Default = 0,
    CHJ = 1,
    PHJ = 2,
    PhjPure = 3,
    All = 4,
};


struct Options
{
    SchemeChoice scheme = SchemeChoice::Default;
    size_t build_rows = 1'000'000;
    size_t probe_rows = 1'000'000;
    PayloadSchema build_schema;
    PayloadSchema probe_schema;
    size_t threads = 1;
    RadixConfig radix;
    size_t reps = 1;
    std::string csv_path;
    bool check = false;
    uint64_t seed = 42;
    /// Per-worker probe-side memory budget for the BEP-backed PHJ, in mebibytes.
    /// Only probe input buffer bytes (across unrefined and leaf chains)
    /// count toward this budget; checked at input-block boundaries.
    size_t bep_budget_mib = 512;
};


/// Parse the command line. On `--help`, prints usage on stdout and returns
/// `nullopt`. On any error, prints the message on stderr and returns
/// `nullopt` with `*ok = false`.
std::optional<Options> parseCli(int argc, char ** argv, bool * ok);


std::string schemaToString(const PayloadSchema & s);

}
