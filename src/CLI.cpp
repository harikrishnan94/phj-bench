#include "CLI.h"

#include <bit>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string_view>


namespace phj
{

namespace
{

const char * USAGE = R"(phj-bench: hash join benchmark (CHJ vs PHJ vs PHJ-BEP)

Usage:
  phj-bench [options]

Options:
  --scheme {chj|phj|phj-bep|all} Which scheme(s) to run (default: all).
  --build-rows N                 Number of build-side rows (default: 1000000).
  --probe-rows N                 Number of probe-side rows (default: 1000000).
  --build-payload-schema LIST    Comma-separated payload types, e.g. u32,u64,u128.
  --probe-payload-schema LIST    Comma-separated payload types, e.g. u32,u64,u128.
  --threads N                    Worker thread count (default: 1).
  --partitions N                 Total partitions, must be a power of 2 (default: 256).
  --passes K                     Number of radix passes (default: 1). Splits the
                                 partition bits evenly across passes.
  --partition-bits-per-pass LIST Comma-separated bits-per-pass list (overrides
                                 --partitions and --passes). Example: 6,4 ->
                                 1024 partitions in two passes.
  --bep-budget-mib N             PHJ-BEP per-worker probe buffer budget in MiB
                                 (default: 32). Only probe input buffer bytes
                                 (unrefined and leaf chains) count toward it.
  --reps N                       Repetitions per scheme (default: 1).
  --csv PATH                     Write per-rep CSV to this path (optional).
  --check                        Validate outputs against std::unordered_map
                                 reference. Default off.
  --seed N                       PRNG seed (default: 42).
  -h, --help                     Print this message and exit.

Types: u8, u16, u32, u64, u128.
)";


bool parseUint(std::string_view s, size_t & out)
{
    size_t v = 0;
    auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc{} || r.ptr != s.data() + s.size())
        return false;
    out = v;
    return true;
}


bool parseUint64(std::string_view s, uint64_t & out)
{
    uint64_t v = 0;
    auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc{} || r.ptr != s.data() + s.size())
        return false;
    out = v;
    return true;
}


bool parsePayloadType(std::string_view s, PayloadType & out)
{
    if (s == "u8")
        out = PayloadType::UInt8;
    else if (s == "u16")
        out = PayloadType::UInt16;
    else if (s == "u32")
        out = PayloadType::UInt32;
    else if (s == "u64")
        out = PayloadType::UInt64;
    else if (s == "u128")
        out = PayloadType::UInt128;
    else
        return false;
    return true;
}


bool parsePayloadSchema(std::string_view s, PayloadSchema & out)
{
    out.types.clear();
    if (s.empty())
        return true;
    size_t i = 0;
    while (i < s.size())
    {
        size_t j = s.find(',', i);
        if (j == std::string_view::npos)
            j = s.size();
        PayloadType t{};
        if (!parsePayloadType(s.substr(i, j - i), t))
            return false;
        out.types.push_back(t);
        i = j + 1;
    }
    return true;
}


bool parseBitsList(std::string_view s, std::vector<uint8_t> & out)
{
    out.clear();
    if (s.empty())
        return false;
    size_t i = 0;
    while (i < s.size())
    {
        size_t j = s.find(',', i);
        if (j == std::string_view::npos)
            j = s.size();
        size_t v = 0;
        if (!parseUint(s.substr(i, j - i), v))
            return false;
        if (v == 0 || v > 16)
            return false;
        out.push_back(static_cast<uint8_t>(v));
        i = j + 1;
    }
    return !out.empty();
}


bool needsValue(std::string_view key)
{
    return key != "--check" && key != "--help" && key != "-h";
}

}


std::optional<Options> parseCli(int argc, char ** argv, bool * ok)
{
    *ok = true;
    Options opts;
    /// Defaults: 256 partitions, single pass, 8 bits.
    opts.radix.pass_bits = {8};

    size_t partitions_override = 0;
    size_t passes_override = 0;
    std::vector<uint8_t> bits_override;

    for (int i = 1; i < argc; ++i)
    {
        std::string_view arg = argv[i];
        std::string_view key = arg;
        std::string_view value;

        const size_t eq = arg.find('=');
        if (eq != std::string_view::npos)
        {
            key = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        }
        else if (needsValue(arg))
        {
            if (i + 1 >= argc)
            {
                std::cerr << "missing value for " << key << "\n";
                *ok = false;
                return std::nullopt;
            }
            value = argv[++i];
        }

        if (key == "--help" || key == "-h")
        {
            std::cout << USAGE;
            return std::nullopt;
        }
        else if (key == "--scheme")
        {
            if (value == "chj")
            {
                opts.scheme = SchemeChoice::CHJ;
            }
            else if (value == "phj")
            {
                opts.scheme = SchemeChoice::PHJ;
            }
            else if (value == "phj-bep")
            {
                opts.scheme = SchemeChoice::PhjBep;
            }
            else if (value == "all")
            {
                opts.scheme = SchemeChoice::All;
            }
            else
            {
                std::cerr << "--scheme must be chj|phj|phj-bep|all\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--build-rows")
        {
            if (!parseUint(value, opts.build_rows))
            {
                std::cerr << "invalid --build-rows\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--probe-rows")
        {
            if (!parseUint(value, opts.probe_rows))
            {
                std::cerr << "invalid --probe-rows\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--build-payload-schema")
        {
            if (!parsePayloadSchema(value, opts.build_schema))
            {
                std::cerr << "invalid --build-payload-schema\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--probe-payload-schema")
        {
            if (!parsePayloadSchema(value, opts.probe_schema))
            {
                std::cerr << "invalid --probe-payload-schema\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--threads")
        {
            if (!parseUint(value, opts.threads) || opts.threads == 0)
            {
                std::cerr << "invalid --threads\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--partitions")
        {
            if (!parseUint(value, partitions_override) || partitions_override == 0
                || (partitions_override & (partitions_override - 1)) != 0)
            {
                std::cerr << "--partitions must be a positive power of 2\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--passes")
        {
            if (!parseUint(value, passes_override) || passes_override == 0)
            {
                std::cerr << "invalid --passes\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--partition-bits-per-pass")
        {
            if (!parseBitsList(value, bits_override))
            {
                std::cerr << "invalid --partition-bits-per-pass\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--reps")
        {
            if (!parseUint(value, opts.reps) || opts.reps == 0)
            {
                std::cerr << "invalid --reps\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--csv")
        {
            opts.csv_path = std::string(value);
        }
        else if (key == "--check")
        {
            opts.check = true;
        }
        else if (key == "--seed")
        {
            if (!parseUint64(value, opts.seed))
            {
                std::cerr << "invalid --seed\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else if (key == "--bep-budget-mib")
        {
            if (!parseUint(value, opts.bep_budget_mib))
            {
                std::cerr << "invalid --bep-budget-mib\n";
                *ok = false;
                return std::nullopt;
            }
        }
        else
        {
            std::cerr << "unknown option: " << key << "\n";
            *ok = false;
            return std::nullopt;
        }
    }

    /// Resolve radix configuration.
    if (!bits_override.empty())
    {
        opts.radix.pass_bits = bits_override;
    }
    else
    {
        const size_t parts = partitions_override == 0 ? size_t{256} : partitions_override;
        const size_t total_bits_sz = static_cast<size_t>(std::bit_width(parts)) - 1U;
        const auto total_bits = static_cast<uint8_t>(total_bits_sz);
        const size_t passes = passes_override == 0 ? size_t{1} : passes_override;
        if (passes > total_bits)
        {
            std::cerr << "passes cannot exceed log2(partitions)\n";
            *ok = false;
            return std::nullopt;
        }
        opts.radix.pass_bits.clear();
        opts.radix.pass_bits.reserve(passes);
        const uint8_t base = static_cast<uint8_t>(total_bits / passes);
        const uint8_t extra = static_cast<uint8_t>(total_bits % passes);
        for (size_t p = 0; p < passes; ++p)
        {
            opts.radix.pass_bits.push_back(static_cast<uint8_t>(base + (p < extra ? 1u : 0u)));
        }
    }

    if (opts.radix.totalBits() == 0 || opts.radix.totalBits() > 32)
    {
        std::cerr << "total partition bits must be in [1, 32]\n";
        *ok = false;
        return std::nullopt;
    }

    return opts;
}


std::string schemaToString(const PayloadSchema & s)
{
    std::ostringstream oss;
    for (size_t i = 0; i < s.types.size(); ++i)
    {
        if (i)
            oss << ',';
        oss << payloadTypeName(s.types[i]);
    }
    if (s.types.empty())
        oss << "<none>";
    return oss.str();
}


}
