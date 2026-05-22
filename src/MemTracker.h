#pragma once

#include <atomic>
#include <cstddef>
#include <memory_resource>


namespace phj
{

/// Polymorphic memory resource that records per-allocation byte counts
/// using lock-free atomics. All allocations are forwarded to `upstream`
/// (defaults to `new_delete_resource()`). Because the PMR contract
/// requires callers to pass the same `bytes` value to `do_deallocate`
/// as was returned from `do_allocate`, we can maintain an exact
/// live-byte counter and a running peak with no bookkeeping overhead.
class MemTracker : public std::pmr::memory_resource
{
public:
    explicit MemTracker(std::pmr::memory_resource * upstream = std::pmr::new_delete_resource()) noexcept
        : upstream_(upstream)
    {
    }

    /// Maximum number of live bytes observed at any single instant during
    /// this tracker's lifetime (safe to read from any thread after all
    /// concurrent allocators have finished).
    [[nodiscard]] size_t peakBytes() const noexcept { return peak_.load(std::memory_order_relaxed); }

    /// Bytes currently outstanding (allocated but not yet freed).
    [[nodiscard]] size_t currentBytes() const noexcept { return current_.load(std::memory_order_relaxed); }

private:
    std::pmr::memory_resource * upstream_;
    std::atomic<size_t> current_{0};
    std::atomic<size_t> peak_{0};

    void * do_allocate(size_t bytes, size_t align) override
    {
        void * p = upstream_->allocate(bytes, align);
        const size_t cur = current_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        size_t old_peak = peak_.load(std::memory_order_relaxed);
        while (cur > old_peak && !peak_.compare_exchange_weak(old_peak, cur, std::memory_order_relaxed))
        {
        }
        return p;
    }

    void do_deallocate(void * p, size_t bytes, size_t align) override
    {
        upstream_->deallocate(p, bytes, align);
        current_.fetch_sub(bytes, std::memory_order_relaxed);
    }

    bool do_is_equal(const std::pmr::memory_resource & other) const noexcept override { return this == &other; }
};

}
