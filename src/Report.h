#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "CHJ.h"
#include "CLI.h"
#include "PHJ.h"


namespace phj
{

/// Pretty-print one run's results (per scheme: median/min/max across reps
/// for each phase metric).
void printSummary(
    std::ostream & os, const Options & opts, const std::vector<ChjResult> & chj_reps, const std::vector<PhjResult> & phj_reps);


/// Write one CSV row per `(scheme, rep)`. Header is emitted iff the file
/// does not already exist and `write_header` is true.
void writeCsv(
    const std::string & path, const Options & opts, const std::vector<ChjResult> & chj_reps, const std::vector<PhjResult> & phj_reps);

}
