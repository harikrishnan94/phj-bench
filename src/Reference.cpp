#include "Reference.h"

#include <algorithm>
#include <cstdint>
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


std::string serializeRow(uint64_t key, const Block & left_blk, size_t left_row, const Block & right_blk, size_t right_row)
{
    std::string s;
    size_t row_size = sizeof(key);
    for (const auto & c : left_blk.payloads)
        row_size += c.elementSize();
    for (const auto & c : right_blk.payloads)
        row_size += c.elementSize();
    s.reserve(row_size);
    appendBytes(s, &key, sizeof(key));
    for (const auto & c : left_blk.payloads)
    {
        const size_t sz = c.elementSize();
        appendBytes(s, c.raw() + left_row * sz, sz);
    }
    for (const auto & c : right_blk.payloads)
    {
        const size_t sz = c.elementSize();
        appendBytes(s, c.raw() + right_row * sz, sz);
    }
    return s;
}

}


std::vector<std::string> referenceOutput(const BlockStream & build, const BlockStream & probe)
{
    /// Index `key -> [(block, row)]` over the build stream so probe
    /// lookups produce the same multi-match expansion as the schemes.
    std::unordered_map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>> index;
    index.reserve(build.total_rows * 2);
    for (size_t b = 0; b < build.blocks.size(); ++b)
    {
        const Block & blk = build.blocks[b];
        for (size_t r = 0; r < blk.rows; ++r)
            index[blk.keys[r]].push_back({static_cast<uint32_t>(b), static_cast<uint32_t>(r)});
    }

    std::vector<std::string> rows;
    rows.reserve(probe.total_rows);
    for (const Block & pblk : probe.blocks)
    {
        for (size_t pr = 0; pr < pblk.rows; ++pr)
        {
            const uint64_t k = pblk.keys[pr];
            auto it = index.find(k);
            if (it == index.end())
                continue;
            for (auto [bb, br] : it->second)
                rows.push_back(serializeRow(k, build.blocks[bb], br, pblk, pr));
        }
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
        for (const auto & b : w.blocks)
        {
            for (size_t i = 0; i < b.rows; ++i)
            {
                std::string row;
                row.reserve(row_size);
                const uint64_t k = b.keys[i];
                appendBytes(row, &k, sizeof(uint64_t));
                for (const auto & c : b.left)
                {
                    const size_t sz = payloadTypeSize(c.type);
                    appendBytes(row, c.raw() + i * sz, sz);
                }
                for (const auto & c : b.right)
                {
                    const size_t sz = payloadTypeSize(c.type);
                    appendBytes(row, c.raw() + i * sz, sz);
                }
                rows.push_back(std::move(row));
            }
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
