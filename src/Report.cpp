#include "Report.h"

#include <algorithm>
#include <array>
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


void printSchemeTable(std::ostream & os, const std::string & scheme, const std::vector<Row> & rows)
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
    os << std::string(16 + COL_WIDTH * 6, '-') << "\n";
}


std::vector<Row> chjRows(const std::vector<ChjResult> & reps)
{
    auto build_ms = std::vector<double>{};
    auto build_ns = std::vector<double>{};
    auto probe_ms = std::vector<double>{};
    auto probe_ns = std::vector<double>{};
    for (const auto & r : reps)
    {
        build_ms.push_back(r.build.wall_ms);
        build_ns.push_back(r.build.ns_per_row);
        probe_ms.push_back(r.probe.wall_ms);
        probe_ns.push_back(r.probe.ns_per_row);
    }
    return {
        {"build", computeStats(build_ms), computeStats(build_ns)},
        {"probe", computeStats(probe_ms), computeStats(probe_ns)},
    };
}


std::vector<Row> phjRows(const std::vector<PhjResult> & reps)
{
    std::vector<double> bs_ms;
    std::vector<double> bs_ns;
    std::vector<double> b_ms;
    std::vector<double> b_ns;
    std::vector<double> ps_ms;
    std::vector<double> ps_ns;
    std::vector<double> p_ms;
    std::vector<double> p_ns;
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
    }
    return {
        {"build-shuffle", computeStats(bs_ms), computeStats(bs_ns)},
        {"build", computeStats(b_ms), computeStats(b_ns)},
        {"probe-shuffle", computeStats(ps_ms), computeStats(ps_ns)},
        {"probe", computeStats(p_ms), computeStats(p_ns)},
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


void printSummary(std::ostream & os, const Options & opts, const std::vector<ChjResult> & chj_reps, const std::vector<PhjResult> & phj_reps)
{
    os << "== Run configuration ==\n";
    os << "  threads:                " << opts.threads << "\n";
    os << "  build_rows:             " << opts.build_rows << "\n";
    os << "  probe_rows:             " << opts.probe_rows << "\n";
    os << "  build_payload_schema:   " << schemaToString(opts.build_schema) << "\n";
    os << "  probe_payload_schema:   " << schemaToString(opts.probe_schema) << "\n";
    os << "  partitions (PHJ):       " << opts.radix.partitions() << " (bits per pass: " << passesBitsString(opts.radix) << ")\n";
    os << "  reps:                   " << opts.reps << "\n";
    os << "  distribution:           uniform\n";
    os << "  match_rate:             1.000\n";
    os << "  seed:                   " << opts.seed << "\n";

    if (!chj_reps.empty())
        printSchemeTable(os, "CHJ", chjRows(chj_reps));
    if (!phj_reps.empty())
        printSchemeTable(os, "PHJ", phjRows(phj_reps));
}


void writeCsv(
    const std::string & path, const Options & opts, const std::vector<ChjResult> & chj_reps, const std::vector<PhjResult> & phj_reps)
{
    const bool need_header = !fileExists(path);
    std::ofstream f(path, std::ios::out | std::ios::app);
    if (!f.good())
        return;
    if (need_header)
    {
        f << "scheme,rep,threads,build_rows,probe_rows,build_payload_schema,probe_payload_schema,partitions,distribution,match_rate,"
             "build_wall_ms,build_ns_per_row,probe_wall_ms,probe_ns_per_row,"
             "build_shuffle_wall_ms,build_shuffle_ns_per_row,probe_shuffle_wall_ms,probe_shuffle_ns_per_row\n";
    }

    const std::string bsch = schemaToString(opts.build_schema);
    const std::string psch = schemaToString(opts.probe_schema);
    auto writeCommon = [&](const char * scheme, size_t rep)
    {
        f << scheme << ',' << rep << ',' << opts.threads << ',' << opts.build_rows << ',' << opts.probe_rows << ",\"" << bsch << "\",\""
          << psch << "\"," << opts.radix.partitions() << ",uniform,1.0,";
    };

    f << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < chj_reps.size(); ++i)
    {
        const auto & r = chj_reps[i];
        writeCommon("CHJ", i);
        f << r.build.wall_ms << ',' << r.build.ns_per_row << ',' << r.probe.wall_ms << ',' << r.probe.ns_per_row << ',' << ',' << ',' << ','
          << '\n';
    }
    for (size_t i = 0; i < phj_reps.size(); ++i)
    {
        const auto & r = phj_reps[i];
        writeCommon("PHJ", i);
        f << r.build.wall_ms << ',' << r.build.ns_per_row << ',' << r.probe.wall_ms << ',' << r.probe.ns_per_row << ','
          << r.build_shuffle.wall_ms << ',' << r.build_shuffle.ns_per_row << ',' << r.probe_shuffle.wall_ms << ','
          << r.probe_shuffle.ns_per_row << '\n';
    }
}

}
