#include <cstdlib>
#include <iostream>
#include <vector>

#include "CHJ.h"
#include "CLI.h"
#include "DataGen.h"
#include "PHJ.h"
#include "PHJBep.h"
#include "Reference.h"
#include "Report.h"
#include "Timer.h"


int main(int argc, char ** argv)
{
    using namespace phj;

    bool ok = true;
    auto maybe = parseCli(argc, argv, &ok);
    if (!ok)
        return 2;
    if (!maybe.has_value())
        return 0;
    const Options & opts = *maybe;

    /// -------- Generate data once (outside the measured window) --------
    const TimePoint t_gen0 = now();
    BlockStream build = generateBuild(opts.build_rows, opts.build_schema, opts.seed, opts.threads);
    BlockStream probe = generateProbe(opts.probe_rows, opts.probe_schema, opts.seed, build, opts.threads);
    const TimePoint t_gen1 = now();
    const double gen_ms = toMillis(t_gen1 - t_gen0);

    std::cerr << "[info] data generation: " << gen_ms << " ms"
              << "  (build_rows=" << opts.build_rows << ", probe_rows=" << opts.probe_rows << ", threads=" << opts.threads
              << ", blocks(build)=" << build.blocks.size() << ", blocks(probe)=" << probe.blocks.size() << ")\n";

    /// -------- Optional reference --------
    std::vector<std::string> reference;
    if (opts.check)
    {
        const TimePoint t_ref0 = now();
        reference = referenceOutput(build, probe);
        const TimePoint t_ref1 = now();
        std::cerr << "[info] reference build: " << toMillis(t_ref1 - t_ref0) << " ms"
                  << "  (output_rows=" << reference.size() << ")\n";
    }

    /// -------- Run reps --------
    std::vector<ChjResult> chj_reps;
    std::vector<PhjResult> phj_reps;
    std::vector<PhjBepResult> bep_reps;

    const bool run_chj = opts.scheme == SchemeChoice::CHJ || opts.scheme == SchemeChoice::All || opts.scheme == SchemeChoice::Default;
    const bool run_phj_pure = opts.scheme == SchemeChoice::PhjPure || opts.scheme == SchemeChoice::All;
    const bool run_phj = opts.scheme == SchemeChoice::PHJ || opts.scheme == SchemeChoice::All || opts.scheme == SchemeChoice::Default;

    /// Only the first rep's `output` is consulted post-loop (by `--check`,
    /// see below). Drop every other rep's materialised output as soon as we
    /// have logged its row count, so the resident set stays bounded by
    /// (one output + one scheme's transient peak) regardless of `--reps`.
    /// At billion-row scales the retained outputs were the dominant cause
    /// of OOM kills across multi-rep runs.
    if (run_chj)
    {
        chj_reps.reserve(opts.reps);
        for (size_t r = 0; r < opts.reps; ++r)
        {
            ChjResult res = runCHJ(build, probe, opts.threads);
            std::cerr << "[info] CHJ rep " << r << ": build " << res.build.wall_ms << " ms, probe " << res.probe.wall_ms << " ms, e2e "
                      << res.e2e_wall_ms << " ms, out_rows " << res.output.totalRows() << "\n";
            if (!opts.check || !chj_reps.empty())
                res.output = JoinOutput{};
            chj_reps.push_back(std::move(res));
        }
    }

    if (run_phj_pure)
    {
        phj_reps.reserve(opts.reps);
        for (size_t r = 0; r < opts.reps; ++r)
        {
            PhjResult res = runPHJ(build, probe, opts.radix, opts.threads);
            std::cerr << "[info] PHJ-PURE rep " << r << ": shuffle " << res.build_shuffle.wall_ms << "/" << res.probe_shuffle.wall_ms
                      << " ms, build " << res.build.wall_ms << " ms, probe " << res.probe.wall_ms << " ms, e2e " << res.e2e_wall_ms
                      << " ms, out_rows " << res.output.totalRows() << "\n";
            if (!opts.check || !phj_reps.empty())
                res.output = JoinOutput{};
            phj_reps.push_back(std::move(res));
        }
    }

    if (run_phj)
    {
        bep_reps.reserve(opts.reps);
        for (size_t r = 0; r < opts.reps; ++r)
        {
            PhjBepResult res = runPhjBep(build, probe, opts.radix, opts.threads, opts.bep_budget_mib);
            std::cerr << "[info] PHJ rep " << r << ": shuffle(build) " << res.build_shuffle.wall_ms << " ms, build " << res.build.wall_ms
                      << " ms, shuffle(probe) " << res.probe_shuffle.wall_ms << " ms, probe " << res.probe.wall_ms << " ms, evict "
                      << res.eviction_overhead.wall_ms << " ms, e2e " << res.e2e_wall_ms << " ms, out_rows " << res.output.totalRows()
                      << ", budget " << res.bep_budget_mib << " MiB, evictions " << res.bep_evictions << ", refinements "
                      << res.bep_refinements << ", bep_peak_mib " << static_cast<double>(res.bep_peak_bytes) / (1024.0 * 1024.0)
                      << " MiB, skip-retries " << res.bep_build_skip_retries << "\n";
            if (!opts.check || !bep_reps.empty())
                res.output = JoinOutput{};
            bep_reps.push_back(std::move(res));
        }
    }

    /// -------- Optional validation --------
    if (opts.check)
    {
        int rc = 0;
        if (run_chj && !chj_reps.empty())
        {
            auto err = compareOutputs(reference, serializeOutput(chj_reps.front().output));
            if (!err.empty())
            {
                std::cerr << "[check] CHJ FAILED: " << err << "\n";
                rc = 1;
            }
            else
            {
                std::cerr << "[check] CHJ OK (" << reference.size() << " rows)\n";
            }
        }
        if (run_phj_pure && !phj_reps.empty())
        {
            auto err = compareOutputs(reference, serializeOutput(phj_reps.front().output));
            if (!err.empty())
            {
                std::cerr << "[check] PHJ-PURE FAILED: " << err << "\n";
                rc = 1;
            }
            else
            {
                std::cerr << "[check] PHJ-PURE OK (" << reference.size() << " rows)\n";
            }
        }
        if (run_phj && !bep_reps.empty())
        {
            auto err = compareOutputs(reference, serializeOutput(bep_reps.front().output));
            if (!err.empty())
            {
                std::cerr << "[check] PHJ FAILED: " << err << "\n";
                rc = 1;
            }
            else
            {
                std::cerr << "[check] PHJ OK (" << reference.size() << " rows)\n";
            }
        }
        if (rc != 0)
            return rc;
    }

    /// -------- Report --------
    printSummary(std::cout, opts, chj_reps, phj_reps, bep_reps);
    if (!opts.csv_path.empty())
        writeCsv(opts.csv_path, opts, chj_reps, phj_reps, bep_reps);

    return 0;
}
