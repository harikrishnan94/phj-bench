#include "Report.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>


namespace phj
{

namespace
{

struct Stats
{
    double med = 0.0;
    double min = 0.0;
    double max = 0.0;
    bool present = false;
};


Stats computeStats(std::vector<double> v)
{
    Stats s;
    if (v.empty())
        return s;
    s.present = true;
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.max = v.back();
    const size_t mid = v.size() / 2;
    s.med = (v.size() % 2 == 1) ? v[mid] : (v[mid - 1] + v[mid]) * 0.5;
    return s;
}


std::string formatDouble(double x, int prec = 3)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << x;
    return oss.str();
}


struct Row
{
    std::string phase;
    Stats wall_ms;
    Stats ns_per_row;
};


struct BepFooter
{
    size_t bep_budget_mib = 0;
    Stats evictions;
    Stats refinements;
    Stats peak_rows;
    Stats skip_retries;
    bool present = false;
};


void printSchemeTable(
    std::ostream & os, const std::string & scheme, const std::vector<Row> & rows, Stats e2e_ms, const BepFooter & bep = {})
{
    os << "\nScheme: " << scheme << "\n";

    constexpr size_t COL_WIDTH = 14;

    auto writeCell = [&](const std::string & s) { os << std::setw(static_cast<int>(COL_WIDTH)) << s; };
    auto writePhase = [&](const std::string & s) { os << std::setw(16) << std::left << s << std::right; };

    os << std::string(16 + COL_WIDTH * 6, '-') << "\n";
    writePhase("phase");
    writeCell("wall_ms_med");
    writeCell("wall_ms_min");
    writeCell("wall_ms_max");
    writeCell("ns/row_med");
    writeCell("ns/row_min");
    writeCell("ns/row_max");
    os << "\n";
    os << std::string(16 + COL_WIDTH * 6, '-') << "\n";

    for (const auto & r : rows)
    {
        if (!r.wall_ms.present)
            continue;
        writePhase(r.phase);
        writeCell(formatDouble(r.wall_ms.med));
        writeCell(formatDouble(r.wall_ms.min));
        writeCell(formatDouble(r.wall_ms.max));
        writeCell(formatDouble(r.ns_per_row.med, 2));
        writeCell(formatDouble(r.ns_per_row.min, 2));
        writeCell(formatDouble(r.ns_per_row.max, 2));
        os << "\n";
    }
    if (e2e_ms.present)
    {
        writePhase("e2e_wall_ms");
        writeCell(formatDouble(e2e_ms.med));
        writeCell(formatDouble(e2e_ms.min));
        writeCell(formatDouble(e2e_ms.max));
        writeCell("");
        writeCell("");
        writeCell("");
        os << "\n";
    }
    os << std::string(16 + COL_WIDTH * 6, '-') << "\n";

    if (bep.present)
    {
        /// BEP per-run metrics footer. `bep_budget_mib` is the input
        /// CLI value (constant across reps); the four counters are
        /// summarised med/min/max across reps (peak_rows is itself
        /// already a max across workers per rep).
        auto writeMetric = [&](const std::string & name, const Stats & s, int prec)
        {
            writePhase(name);
            writeCell(formatDouble(s.med, prec));
            writeCell(formatDouble(s.min, prec));
            writeCell(formatDouble(s.max, prec));
            writeCell("");
            writeCell("");
            writeCell("");
            os << "\n";
        };
        writePhase("bep_budget_mib");
        writeCell(formatDouble(static_cast<double>(bep.bep_budget_mib), 0));
        writeCell("");
        writeCell("");
        writeCell("");
        writeCell("");
        writeCell("");
        os << "\n";
        writeMetric("bep_evictions", bep.evictions, 0);
        writeMetric("bep_refinements", bep.refinements, 0);
        writeMetric("bep_peak_rows", bep.peak_rows, 0);
        writeMetric("bep_skip_retries", bep.skip_retries, 0);
        os << std::string(16 + COL_WIDTH * 6, '-') << "\n";
    }
}


std::vector<Row> chjRows(const std::vector<ChjResult> & reps, Stats & e2e_out)
{
    std::vector<double> build_ms;
    std::vector<double> build_ns;
    std::vector<double> probe_ms;
    std::vector<double> probe_ns;
    std::vector<double> e2e_ms;
    for (const auto & r : reps)
    {
        build_ms.push_back(r.build.wall_ms);
        build_ns.push_back(r.build.ns_per_row);
        probe_ms.push_back(r.probe.wall_ms);
        probe_ns.push_back(r.probe.ns_per_row);
        e2e_ms.push_back(r.e2e_wall_ms);
    }
    e2e_out = computeStats(e2e_ms);
    return {
        {"build", computeStats(build_ms), computeStats(build_ns)},
        {"probe", computeStats(probe_ms), computeStats(probe_ns)},
    };
}


std::vector<Row> phjRows(const std::vector<PhjResult> & reps, Stats & e2e_out)
{
    std::vector<double> bs_ms;
    std::vector<double> bs_ns;
    std::vector<double> b_ms;
    std::vector<double> b_ns;
    std::vector<double> ps_ms;
    std::vector<double> ps_ns;
    std::vector<double> p_ms;
    std::vector<double> p_ns;
    std::vector<double> e2e_ms;
    for (const auto & r : reps)
    {
        bs_ms.push_back(r.build_shuffle.wall_ms);
        bs_ns.push_back(r.build_shuffle.ns_per_row);
        b_ms.push_back(r.build.wall_ms);
        b_ns.push_back(r.build.ns_per_row);
        ps_ms.push_back(r.probe_shuffle.wall_ms);
        ps_ns.push_back(r.probe_shuffle.ns_per_row);
        p_ms.push_back(r.probe.wall_ms);
        p_ns.push_back(r.probe.ns_per_row);
        e2e_ms.push_back(r.e2e_wall_ms);
    }
    e2e_out = computeStats(e2e_ms);
    return {
        {"build-shuffle", computeStats(bs_ms), computeStats(bs_ns)},
        {"build", computeStats(b_ms), computeStats(b_ns)},
        {"probe-shuffle", computeStats(ps_ms), computeStats(ps_ns)},
        {"probe", computeStats(p_ms), computeStats(p_ns)},
    };
}


std::vector<Row> bepRows(const std::vector<PhjBepResult> & reps, Stats & e2e_out, BepFooter & footer)
{
    std::vector<double> bs_ms;
    std::vector<double> bs_ns;
    std::vector<double> b_ms;
    std::vector<double> b_ns;
    std::vector<double> ps_ms;
    std::vector<double> ps_ns;
    std::vector<double> p_ms;
    std::vector<double> p_ns;
    std::vector<double> e_ms;
    std::vector<double> e_ns;
    std::vector<double> e2e_ms;
    std::vector<double> evicts;
    std::vector<double> refines;
    std::vector<double> peaks;
    std::vector<double> skips;
    for (const auto & r : reps)
    {
        bs_ms.push_back(r.build_shuffle.wall_ms);
        bs_ns.push_back(r.build_shuffle.ns_per_row);
        b_ms.push_back(r.build.wall_ms);
        b_ns.push_back(r.build.ns_per_row);
        ps_ms.push_back(r.probe_shuffle.wall_ms);
        ps_ns.push_back(r.probe_shuffle.ns_per_row);
        p_ms.push_back(r.probe.wall_ms);
        p_ns.push_back(r.probe.ns_per_row);
        e_ms.push_back(r.eviction_overhead.wall_ms);
        e_ns.push_back(r.eviction_overhead.ns_per_row);
        e2e_ms.push_back(r.e2e_wall_ms);
        evicts.push_back(static_cast<double>(r.bep_evictions));
        refines.push_back(static_cast<double>(r.bep_refinements));
        peaks.push_back(static_cast<double>(r.bep_peak_buffered_rows));
        skips.push_back(static_cast<double>(r.bep_build_skip_retries));
    }
    e2e_out = computeStats(e2e_ms);
    footer.present = true;
    footer.bep_budget_mib = reps.empty() ? 0 : reps.front().bep_budget_mib;
    footer.evictions = computeStats(evicts);
    footer.refinements = computeStats(refines);
    footer.peak_rows = computeStats(peaks);
    footer.skip_retries = computeStats(skips);
    return {
        {"build-shuffle", computeStats(bs_ms), computeStats(bs_ns)},
        {"build", computeStats(b_ms), computeStats(b_ns)},
        {"probe-shuffle", computeStats(ps_ms), computeStats(ps_ns)},
        {"probe", computeStats(p_ms), computeStats(p_ns)},
        {"eviction-overhead", computeStats(e_ms), computeStats(e_ns)},
    };
}


std::string passesBitsString(const RadixConfig & cfg)
{
    std::ostringstream oss;
    for (size_t i = 0; i < cfg.pass_bits.size(); ++i)
    {
        if (i)
            oss << '+';
        oss << static_cast<int>(cfg.pass_bits[i]);
    }
    return oss.str();
}


bool fileExists(const std::string & path)
{
    std::ifstream f(path);
    return f.good();
}

}


void printSummary(
    std::ostream & os,
    const Options & opts,
    const std::vector<ChjResult> & chj_reps,
    const std::vector<PhjResult> & phj_reps,
    const std::vector<PhjBepResult> & bep_reps)
{
    os << "== Run configuration ==\n";
    os << "  threads:                " << opts.threads << "\n";
    os << "  build_rows:             " << opts.build_rows << "\n";
    os << "  probe_rows:             " << opts.probe_rows << "\n";
    os << "  build_payload_schema:   " << schemaToString(opts.build_schema) << "\n";
    os << "  probe_payload_schema:   " << schemaToString(opts.probe_schema) << "\n";
    os << "  partitions (PHJ):       " << opts.radix.partitions() << " (bits per pass: " << passesBitsString(opts.radix) << ")\n";
    os << "  bep_budget_mib:         " << opts.bep_budget_mib << "\n";
    os << "  reps:                   " << opts.reps << "\n";
    os << "  distribution:           uniform\n";
    os << "  match_rate:             1.000\n";
    os << "  seed:                   " << opts.seed << "\n";

    if (!chj_reps.empty())
    {
        Stats e2e;
        auto rows = chjRows(chj_reps, e2e);
        printSchemeTable(os, "CHJ", rows, e2e);
    }
    if (!phj_reps.empty())
    {
        Stats e2e;
        auto rows = phjRows(phj_reps, e2e);
        printSchemeTable(os, "PHJ-PURE", rows, e2e);
    }
    if (!bep_reps.empty())
    {
        Stats e2e;
        BepFooter footer;
        auto rows = bepRows(bep_reps, e2e, footer);
        printSchemeTable(os, "PHJ", rows, e2e, footer);
    }
}


void writeCsv(
    const std::string & path,
    const Options & opts,
    const std::vector<ChjResult> & chj_reps,
    const std::vector<PhjResult> & phj_reps,
    const std::vector<PhjBepResult> & bep_reps)
{
    const bool need_header = !fileExists(path);
    std::ofstream f(path, std::ios::out | std::ios::app);
    if (!f.good())
        return;
    if (need_header)
    {
        /// Columns in order:
        ///   - common metadata
        ///   - build/probe phases (shared across all three schemes)
        ///   - build_shuffle/probe_shuffle phases (PHJ-PURE + PHJ only;
        ///     blank for CHJ)
        ///   - eviction_overhead phase (PHJ only)
        ///   - e2e
        ///   - BEP-only per-run metrics (PHJ only)
        f << "scheme,rep,threads,build_rows,probe_rows,build_payload_schema,probe_payload_schema,partitions,distribution,match_rate,"
             "build_wall_ms,build_ns_per_row,probe_wall_ms,probe_ns_per_row,"
             "build_shuffle_wall_ms,build_shuffle_ns_per_row,probe_shuffle_wall_ms,probe_shuffle_ns_per_row,"
             "eviction_overhead_wall_ms,eviction_overhead_ns_per_row,"
             "e2e_wall_ms,"
             "bep_budget_mib,bep_evictions,bep_refinements,bep_peak_buffered_rows,bep_build_skip_retries\n";
    }

    const std::string bsch = schemaToString(opts.build_schema);
    const std::string psch = schemaToString(opts.probe_schema);
    auto writeCommon = [&](const char * scheme, size_t rep)
    {
        f << scheme << ',' << rep << ',' << opts.threads << ',' << opts.build_rows << ',' << opts.probe_rows << ",\"" << bsch << "\",\""
          << psch << "\"," << opts.radix.partitions() << ",uniform,1.0,";
    };

    f << std::fixed << std::setprecision(6);
    /// Column-blank helpers. `emit*` writes a value (or empty for
    /// blank columns) followed by the trailing column separator; the
    /// last column on each row uses `emitTail*` (no trailing comma)
    /// before the explicit `\n`.
    auto emit = [&](double v) { f << v << ','; };
    auto emitInt = [&](size_t v) { f << v << ','; };
    auto emitBlank = [&]() { f << ','; };
    auto emitTailInt = [&](size_t v) { f << v; };
    auto emitTailBlank = [&]() { /* nothing — the column ends EOL */ };

    for (size_t i = 0; i < chj_reps.size(); ++i)
    {
        const auto & r = chj_reps[i];
        writeCommon("CHJ", i);
        /// CHJ: build, probe filled; shuffles + eviction-overhead blank;
        /// e2e filled; BEP-only columns blank.
        emit(r.build.wall_ms);
        emit(r.build.ns_per_row);
        emit(r.probe.wall_ms);
        emit(r.probe.ns_per_row);
        emitBlank(); /* build_shuffle_wall_ms */
        emitBlank(); /* build_shuffle_ns_per_row */
        emitBlank(); /* probe_shuffle_wall_ms */
        emitBlank(); /* probe_shuffle_ns_per_row */
        emitBlank(); /* eviction_overhead_wall_ms */
        emitBlank(); /* eviction_overhead_ns_per_row */
        emit(r.e2e_wall_ms);
        emitBlank(); /* bep_budget_mib */
        emitBlank(); /* bep_evictions */
        emitBlank(); /* bep_refinements */
        emitBlank(); /* bep_peak_buffered_rows */
        emitTailBlank(); /* bep_build_skip_retries */
        f << '\n';
    }
    for (size_t i = 0; i < phj_reps.size(); ++i)
    {
        const auto & r = phj_reps[i];
        writeCommon("PHJ-PURE", i);
        emit(r.build.wall_ms);
        emit(r.build.ns_per_row);
        emit(r.probe.wall_ms);
        emit(r.probe.ns_per_row);
        emit(r.build_shuffle.wall_ms);
        emit(r.build_shuffle.ns_per_row);
        emit(r.probe_shuffle.wall_ms);
        emit(r.probe_shuffle.ns_per_row);
        emitBlank(); /* eviction_overhead_wall_ms */
        emitBlank(); /* eviction_overhead_ns_per_row */
        emit(r.e2e_wall_ms);
        emitBlank(); /* bep_budget_mib */
        emitBlank(); /* bep_evictions */
        emitBlank(); /* bep_refinements */
        emitBlank(); /* bep_peak_buffered_rows */
        emitTailBlank(); /* bep_build_skip_retries */
        f << '\n';
    }
    for (size_t i = 0; i < bep_reps.size(); ++i)
    {
        const auto & r = bep_reps[i];
        writeCommon("PHJ", i);
        emit(r.build.wall_ms);
        emit(r.build.ns_per_row);
        emit(r.probe.wall_ms);
        emit(r.probe.ns_per_row);
        emit(r.build_shuffle.wall_ms);
        emit(r.build_shuffle.ns_per_row);
        emit(r.probe_shuffle.wall_ms);
        emit(r.probe_shuffle.ns_per_row);
        emit(r.eviction_overhead.wall_ms);
        emit(r.eviction_overhead.ns_per_row);
        emit(r.e2e_wall_ms);
        emitInt(r.bep_budget_mib);
        emitInt(r.bep_evictions);
        emitInt(r.bep_refinements);
        emitInt(r.bep_peak_buffered_rows);
        emitTailInt(r.bep_build_skip_retries);
        f << '\n';
    }
}

}
