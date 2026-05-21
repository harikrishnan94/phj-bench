#pragma once

#include <string>
#include <vector>

#include "Block.h"
#include "JoinOutput.h"


namespace phj
{

/// Compute the expected output of the inner join as a multiset of
/// byte-serialised `(key, left_payloads..., right_payloads...)` rows.
[[nodiscard]] std::vector<std::string> referenceOutput(const BlockStream & build, const BlockStream & probe);


/// Materialise the actual output as a sorted vector of byte-serialised
/// rows (same shape as `referenceOutput`).
[[nodiscard]] std::vector<std::string> serializeOutput(const JoinOutput & out);


/// Sort the two vectors and compare. Returns an empty string on success
/// or a description of the first mismatch.
[[nodiscard]] std::string compareOutputs(std::vector<std::string> expected, std::vector<std::string> actual);

}
