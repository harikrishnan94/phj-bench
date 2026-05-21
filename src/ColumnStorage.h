#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include "Types.h"


namespace phj
{

/// Flat type-erased payload column: `rows * payloadTypeSize(type)` bytes,
/// contiguous, owned. All payload columns are stored column-major.
struct PayloadColumn
{
    PayloadType type = PayloadType::UInt8;
    size_t rows = 0;
    std::unique_ptr<std::byte[]> data;

    static PayloadColumn allocate(PayloadType t, size_t r)
    {
        PayloadColumn c;
        c.type = t;
        c.rows = r;
        if (r > 0)
            c.data = std::make_unique<std::byte[]>(r * payloadTypeSize(t));
        return c;
    }

    [[nodiscard]] const std::byte * raw() const noexcept { return data.get(); }
};


struct PayloadSchema
{
    std::vector<PayloadType> types;

    [[nodiscard]] size_t rowByteSize() const noexcept
    {
        size_t s = 0;
        for (auto t : types)
            s += payloadTypeSize(t);
        return s;
    }

    [[nodiscard]] size_t size() const noexcept { return types.size(); }
};


/// A column-major set of rows: a uint64 key column plus N payload columns.
/// Owns all storage. Used for input data, for compacted per-partition build
/// columns, and for output column buffers.
struct ColumnSet
{
    size_t rows = 0;
    PayloadSchema schema;
    std::unique_ptr<uint64_t[]> keys;
    std::vector<PayloadColumn> payloads;

    static ColumnSet allocate(size_t rows_, const PayloadSchema & s)
    {
        ColumnSet cs;
        cs.rows = rows_;
        cs.schema = s;
        if (rows_ > 0)
            cs.keys = std::make_unique<uint64_t[]>(rows_);
        cs.payloads.reserve(s.types.size());
        for (auto t : s.types)
            cs.payloads.push_back(PayloadColumn::allocate(t, rows_));
        return cs;
    }

    [[nodiscard]] const uint64_t * keyData() const noexcept { return keys.get(); }
};


/// Single-key column chain: append-only chain of fixed 16 KiB buffers
/// holding `uint64_t` values. Each new buffer is exactly 16 KiB and holds
/// `ROWS_PER_BUFFER` entries; the chain grows by one buffer at a time when
/// it fills up.
struct KeyBufferChain
{
    static constexpr size_t BUFFER_BYTES = 16 * 1024;
    static constexpr size_t ROWS_PER_BUFFER = BUFFER_BYTES / sizeof(uint64_t);

    std::vector<std::unique_ptr<uint64_t[]>> buffers;
    size_t rows = 0;
    uint64_t * write_ptr = nullptr;
    uint64_t * write_end = nullptr;

    [[gnu::always_inline]] inline uint64_t * nextSlot()
    {
        if (write_ptr == write_end) [[unlikely]]
            grow();
        ++rows;
        return write_ptr++;
    }

    /// Random read access. Use after all writes are done.
    [[nodiscard, gnu::always_inline]] inline uint64_t valueAt(size_t row) const noexcept
    {
        return buffers[row / ROWS_PER_BUFFER].get()[row % ROWS_PER_BUFFER];
    }

    void clear() noexcept
    {
        buffers.clear();
        rows = 0;
        write_ptr = nullptr;
        write_end = nullptr;
    }

private:
    void grow()
    {
        buffers.emplace_back(std::make_unique<uint64_t[]>(ROWS_PER_BUFFER));
        write_ptr = buffers.back().get();
        write_end = write_ptr + ROWS_PER_BUFFER;
    }
};


/// Single payload column chain: type-erased fixed 16 KiB buffers. Knows
/// its element size at construction so different types yield different
/// rows-per-buffer counts.
struct PayloadBufferChain
{
    static constexpr size_t BUFFER_BYTES = 16 * 1024;

    PayloadType type = PayloadType::UInt8;
    size_t element_size = 1;
    size_t rows_per_buffer = BUFFER_BYTES;
    std::vector<std::unique_ptr<std::byte[]>> buffers;
    size_t rows = 0;
    std::byte * write_ptr = nullptr;
    std::byte * write_end = nullptr;

    PayloadBufferChain() = default;
    explicit PayloadBufferChain(PayloadType t)
        : type(t)
        , element_size(payloadTypeSize(t))
        , rows_per_buffer(BUFFER_BYTES / payloadTypeSize(t))
    {
    }

    [[gnu::always_inline]] inline std::byte * nextSlot()
    {
        if (write_ptr == write_end) [[unlikely]]
            grow();
        std::byte * slot = write_ptr;
        write_ptr += element_size;
        ++rows;
        return slot;
    }

    /// Random read access. Returns a pointer to the row's storage.
    [[nodiscard, gnu::always_inline]] inline const std::byte * pointerAt(size_t row) const noexcept
    {
        return buffers[row / rows_per_buffer].get() + (row % rows_per_buffer) * element_size;
    }

    void clear() noexcept
    {
        buffers.clear();
        rows = 0;
        write_ptr = nullptr;
        write_end = nullptr;
    }

private:
    void grow()
    {
        buffers.emplace_back(std::make_unique<std::byte[]>(BUFFER_BYTES));
        write_ptr = buffers.back().get();
        write_end = write_ptr + rows_per_buffer * element_size;
    }
};


/// Per-(thread, partition) buffer set produced by the radix-shuffle
/// operator: one key chain and one chain per payload column being shuffled.
/// Append order is consistent across columns so row n in `keys` corresponds
/// to row n in each payload chain; however because different payload types
/// hold different numbers of rows per 16 KiB buffer, the chunk layouts of
/// `keys` and the individual payload chains differ. Consumers iterate by
/// flat row index (via the cursors below) rather than by chunk.
struct ThreadPartitionBuffers
{
    KeyBufferChain keys;
    std::vector<PayloadBufferChain> payloads;
    size_t rows = 0;

    void init(const PayloadSchema & s)
    {
        payloads.clear();
        payloads.reserve(s.types.size());
        for (auto t : s.types)
            payloads.emplace_back(t);
    }
};


/// Sequential read cursor over a key chain. Each call to `next()` returns
/// the value at the next flat row index.
struct KeyChainReader
{
    const KeyBufferChain * chain = nullptr;
    const uint64_t * cur = nullptr;
    const uint64_t * end = nullptr;
    size_t next_buf = 0;

    explicit KeyChainReader(const KeyBufferChain & c)
        : chain(&c)
    {
        reload();
    }

    [[gnu::always_inline]] inline uint64_t next()
    {
        if (cur == end) [[unlikely]]
            reload();
        /// Caller iterates exactly `chain->rows` times and never beyond.
        if (cur == nullptr) [[unlikely]]
            __builtin_unreachable();
        return *cur++;
    }

private:
    void reload()
    {
        if (chain == nullptr || next_buf >= chain->buffers.size())
        {
            cur = nullptr;
            end = nullptr;
            return;
        }
        cur = chain->buffers[next_buf].get();
        const size_t consumed = next_buf * KeyBufferChain::ROWS_PER_BUFFER;
        const size_t remaining = chain->rows - consumed;
        const size_t take = remaining < KeyBufferChain::ROWS_PER_BUFFER ? remaining : KeyBufferChain::ROWS_PER_BUFFER;
        end = cur + take;
        ++next_buf;
    }
};


/// Sequential read cursor over a payload chain. Each call to `next()`
/// returns a pointer to the row at the next flat row index.
struct PayloadChainReader
{
    const PayloadBufferChain * chain = nullptr;
    const std::byte * cur = nullptr;
    const std::byte * end = nullptr;
    size_t next_buf = 0;

    explicit PayloadChainReader(const PayloadBufferChain & c)
        : chain(&c)
    {
        reload();
    }

    [[gnu::always_inline]] inline const std::byte * next()
    {
        if (cur == end) [[unlikely]]
            reload();
        if (cur == nullptr) [[unlikely]]
            __builtin_unreachable();
        const std::byte * p = cur;
        cur += chain->element_size;
        return p;
    }

private:
    void reload()
    {
        if (chain == nullptr || next_buf >= chain->buffers.size())
        {
            cur = nullptr;
            end = nullptr;
            return;
        }
        cur = chain->buffers[next_buf].get();
        const size_t consumed = next_buf * chain->rows_per_buffer;
        const size_t remaining = chain->rows - consumed;
        const size_t take = remaining < chain->rows_per_buffer ? remaining : chain->rows_per_buffer;
        end = cur + take * chain->element_size;
        ++next_buf;
    }
};

}
