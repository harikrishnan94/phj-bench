#pragma once

#include <string>

#include "ColumnStorage.h"
#include "JoinOutput.h"


namespace phj
{

/// Compute the expected output of the inner join as a multiset of byte-
/// serialised `(key, left_payloads..., right_payloads...)` rows.
[[nodiscard]] std::vector<std::string> referenceOutput(const ColumnSet & build_cs, const ColumnSet & probe_cs);


/// Materialise the actual output as a sorted vector of byte-serialised rows
/// (same shape as `referenceOutput`).
[[nodiscard]] std::vector<std::string> serializeOutput(const JoinOutput & out);


/// Sort the two vectors and compare. Returns an empty string on success or
/// a description of the first mismatch. Both vectors must be deterministic
/// outputs that, when sorted, represent the same multiset.
[[nodiscard]] std::string compareOutputs(std::vector<std::string> expected, std::vector<std::string> actual);

}
