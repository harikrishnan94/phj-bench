#pragma once

#include <chrono>
#include <cstdint>


namespace phj
{

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = Clock::duration;

[[gnu::always_inline]] inline TimePoint now() noexcept
{
    return Clock::now();
}

inline uint64_t toNanos(Duration d) noexcept
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}

inline double toMillis(Duration d) noexcept
{
    return std::chrono::duration<double, std::milli>(d).count();
}

inline double nanosToMillis(uint64_t ns) noexcept
{
    return static_cast<double>(ns) * 1e-6;
}

}
