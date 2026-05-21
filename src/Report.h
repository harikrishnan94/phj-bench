#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "CHJ.h"
#include "CLI.h"
#include "PHJ.h"
#include "PHJBep.h"


namespace phj
{

/// Pretty-print one run's results (per scheme: median/min/max across reps
/// for each phase metric). The PHJ-BEP table additionally contains an
/// `eviction-overhead` phase row and a footer with the BEP per-run
/// metrics (budget, evictions, refinements, peak buffered rows, build
/// skip retries).
void printSummary(
    std::ostream & os,
    const Options & opts,
    const std::vector<ChjResult> & chj_reps,
    const std::vector<PhjResult> & phj_reps,
    const std::vector<PhjBepResult> & bep_reps);


/// Write one CSV row per `(scheme, rep)`. Header is emitted iff the
/// file does not already exist. CHJ rows leave the PHJ-only phases and
/// BEP-only columns blank; PHJ rows leave the BEP-only columns blank.
void writeCsv(
    const std::string & path,
    const Options & opts,
    const std::vector<ChjResult> & chj_reps,
    const std::vector<PhjResult> & phj_reps,
    const std::vector<PhjBepResult> & bep_reps);

}
