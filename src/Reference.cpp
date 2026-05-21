#include "Reference.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <unordered_map>


namespace phj
{

namespace
{

void appendBytes(std::string & out, const void * src, size_t n)
{
    const auto * p = static_cast<const char *>(src);
    out.append(p, n);
}


std::string serializeRow(uint64_t key, const ColumnSet & left_cs, size_t left_row, const ColumnSet & right_cs, size_t right_row)
{
    std::string s;
    s.reserve(sizeof(key) + left_cs.schema.rowByteSize() + right_cs.schema.rowByteSize());
    appendBytes(s, &key, sizeof(key));
    for (size_t c = 0; c < left_cs.schema.types.size(); ++c)
    {
        const size_t sz = payloadTypeSize(left_cs.schema.types[c]);
        appendBytes(s, left_cs.payloads[c].data.get() + left_row * sz, sz);
    }
    for (size_t c = 0; c < right_cs.schema.types.size(); ++c)
    {
        const size_t sz = payloadTypeSize(right_cs.schema.types[c]);
        appendBytes(s, right_cs.payloads[c].data.get() + right_row * sz, sz);
    }
    return s;
}

}


std::vector<std::string> referenceOutput(const ColumnSet & build_cs, const ColumnSet & probe_cs)
{
    std::unordered_map<uint64_t, std::vector<uint32_t>> index;
    index.reserve(build_cs.rows * 2);
    for (size_t i = 0; i < build_cs.rows; ++i)
        index[build_cs.keyData()[i]].push_back(static_cast<uint32_t>(i));

    std::vector<std::string> rows;
    rows.reserve(probe_cs.rows);
    for (size_t j = 0; j < probe_cs.rows; ++j)
    {
        const uint64_t k = probe_cs.keyData()[j];
        auto it = index.find(k);
        if (it == index.end())
            continue;
        for (uint32_t r : it->second)
            rows.push_back(serializeRow(k, build_cs, r, probe_cs, j));
    }
    return rows;
}


std::vector<std::string> serializeOutput(const JoinOutput & out)
{
    std::vector<std::string> rows;
    rows.reserve(out.totalRows());
    const size_t row_size = sizeof(uint64_t) + out.left_schema.rowByteSize() + out.right_schema.rowByteSize();
    for (const auto & w : out.workers)
    {
        const size_t n_rows = w.keys.rows;
        KeyChainReader key_cur(w.keys);
        std::vector<PayloadChainReader> left_curs;
        left_curs.reserve(w.left.size());
        for (const auto & lc : w.left)
            left_curs.emplace_back(lc);
        std::vector<PayloadChainReader> right_curs;
        right_curs.reserve(w.right.size());
        for (const auto & rc : w.right)
            right_curs.emplace_back(rc);

        for (size_t i = 0; i < n_rows; ++i)
        {
            std::string row;
            row.reserve(row_size);
            const uint64_t k = key_cur.next();
            row.append(reinterpret_cast<const char *>(&k), sizeof(uint64_t));
            for (size_t c = 0; c < w.left.size(); ++c)
                row.append(reinterpret_cast<const char *>(left_curs[c].next()), w.left[c].element_size);
            for (size_t c = 0; c < w.right.size(); ++c)
                row.append(reinterpret_cast<const char *>(right_curs[c].next()), w.right[c].element_size);
            rows.push_back(std::move(row));
        }
    }
    return rows;
}


std::string compareOutputs(std::vector<std::string> expected, std::vector<std::string> actual)
{
    if (expected.size() != actual.size())
    {
        std::ostringstream oss;
        oss << "row count mismatch: expected " << expected.size() << ", actual " << actual.size();
        return oss.str();
    }
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    if (expected != actual)
    {
        size_t first_diff = 0;
        while (first_diff < expected.size() && expected[first_diff] == actual[first_diff])
            ++first_diff;
        std::ostringstream oss;
        oss << "row content mismatch at position " << first_diff;
        return oss.str();
    }
    return {};
}

}
