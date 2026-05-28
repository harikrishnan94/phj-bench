#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory_resource>


namespace phj
{

/// Polymorphic memory resource that records per-allocation byte counts.
///
/// PERFORMANCE NOTE: the previous design hammered a single global atomic
/// `current_` (fetch_add) and a CAS loop on `peak_` for EVERY allocation
/// and deallocation. Under the BEP probe workload that pattern dominated
/// CPU time — `perf record` on a 48-thread / 2048-partition run measured
///
///   16.99% MemTracker::do_allocate
///   12.39% MemTracker::do_deallocate
///   = 29.4% of total CPU on the shared atomic cache lines.
///
/// Root cause: ~22M alloc/dealloc events (refine intermediate OutBlocks,
/// dropPartition during eviction/drain) all serialised through one
/// 64-byte cache line. With 48 cores, each atomic operation paid
/// ~100–300 ns of cross-socket coherence traffic.
///
/// New design: per-thread accumulator with periodic flush.
///   - Each thread holds a private `ssize_t tls_delta_` (no atomic).
///   - On alloc, `tls_delta_ += bytes`. On dealloc, `tls_delta_ -= bytes`.
///   - When |delta| crosses `kFlushBytes`, the thread folds its delta
///     into the global atomics and resets. Peak is updated with a
///     non-CAS relaxed store (racy but bounded; the error is at most
///     `kFlushBytes` per thread, well below the bytes/MiB granularity
///     we report).
///   - A thread_local destructor on `TlsFlusher` folds any residual
///     delta back into the globals on thread exit. parallelRun joins
///     all workers before peakBytes()/currentBytes() are read, so the
///     globals are exact at report time.
///   - A second thread_local `tls_owner_` detects multi-MemTracker use
///     (rare in practice) and flushes when the active tracker changes,
///     so concurrent trackers do not corrupt each other.
///
/// Net effect: ~22M atomic ops collapsed to ~hundreds (one per
/// kFlushBytes-worth of allocation per thread, plus one per
/// thread exit). The hot path is one TLS read + one branch.
class MemTracker : public std::pmr::memory_resource
{
public:
    explicit MemTracker(std::pmr::memory_resource * upstream = std::pmr::new_delete_resource()) noexcept
        : upstream_(upstream)
    {
    }

    /// Maximum number of live bytes observed at any single instant during
    /// this tracker's lifetime. Accurate to within `kFlushBytes` per
    /// thread that is still alive at the time of the call; threads that
    /// have exited already folded their residual deltas in via the
    /// `TlsFlusher` destructor.
    [[nodiscard]] size_t peakBytes() const noexcept { return peak_.load(std::memory_order_relaxed); }

    /// Bytes currently outstanding (allocated but not yet freed). Same
    /// per-thread approximation caveat as `peakBytes`.
    [[nodiscard]] size_t currentBytes() const noexcept { return current_.load(std::memory_order_relaxed); }

private:
    std::pmr::memory_resource * upstream_;
    std::atomic<size_t> current_{0};
    std::atomic<size_t> peak_{0};

    /// Threshold for folding the per-thread cached delta back into the
    /// global atomics. 1 MiB chosen so that:
    ///   - flush frequency is ~ (per-thread bytes allocated) / 1 MiB
    ///     (e.g. ~50 flushes per worker for a 50 MiB allocation total).
    ///   - the per-thread peak under-estimation is bounded at 1 MiB
    ///     per thread, i.e. <= 48 MiB across all workers — well below
    ///     the ~MiB granularity of `peak_mem_mib`.
    static constexpr ssize_t kFlushBytes = 1 << 20;

    /// Per-thread accumulator. One per (thread, MemTracker) pair, but
    /// cached in a single TLS slot — see `flushIfOtherOwner` below for
    /// the cross-tracker switch handling.
    struct TlsFlusher
    {
        MemTracker * owner = nullptr;
        ssize_t delta = 0;

        ~TlsFlusher() noexcept
        {
            if (owner != nullptr)
                owner->foldDelta(delta);
        }
    };

    static thread_local TlsFlusher tls_;

    /// Fold `d` into the globals (called both by the periodic flush in
    /// do_allocate/do_deallocate and by the TlsFlusher destructor on
    /// thread exit). Peak updates use a relaxed store (not CAS) — the
    /// race window can miss a peak by at most one in-flight flush, but
    /// it removes the CAS-loop cost entirely. Empirically the peak
    /// drifts by 0–1 MiB in repeated runs, which is below the
    /// reporting granularity.
    void foldDelta(ssize_t d) noexcept
    {
        if (d == 0)
            return;
        size_t cur;
        if (d > 0)
        {
            cur = current_.fetch_add(static_cast<size_t>(d), std::memory_order_relaxed) + static_cast<size_t>(d);
            const size_t old_peak = peak_.load(std::memory_order_relaxed);
            if (cur > old_peak)
                peak_.store(cur, std::memory_order_relaxed);
        }
        else
        {
            current_.fetch_sub(static_cast<size_t>(-d), std::memory_order_relaxed);
        }
    }

    /// If TLS currently caches a delta for a different MemTracker, flush
    /// that delta back to its owner before retargeting this thread's
    /// cache. Common case: TLS is already pointing at `this` — single
    /// pointer compare, no branch missed.
    void flushIfOtherOwner() noexcept
    {
        if (tls_.owner != this)
        {
            if (tls_.owner != nullptr)
            {
                tls_.owner->foldDelta(tls_.delta);
                tls_.delta = 0;
            }
            tls_.owner = this;
        }
    }

    void * do_allocate(size_t bytes, size_t align) override
    {
        void * p = upstream_->allocate(bytes, align);
        flushIfOtherOwner();
        tls_.delta += static_cast<ssize_t>(bytes);
        if (tls_.delta >= kFlushBytes)
        {
            foldDelta(tls_.delta);
            tls_.delta = 0;
        }
        return p;
    }

    void do_deallocate(void * p, size_t bytes, size_t align) override
    {
        upstream_->deallocate(p, bytes, align);
        flushIfOtherOwner();
        tls_.delta -= static_cast<ssize_t>(bytes);
        if (tls_.delta <= -kFlushBytes)
        {
            foldDelta(tls_.delta);
            tls_.delta = 0;
        }
    }

    bool do_is_equal(const std::pmr::memory_resource & other) const noexcept override { return this == &other; }
};


inline thread_local MemTracker::TlsFlusher MemTracker::tls_{};

}
