#pragma once

#include <cstddef>
#include <thread>
#include <vector>


namespace phj
{

/// Spawn `threads` joinable workers and invoke `fn(tid)` on each. Joins
/// before returning. The join provides an implicit barrier; callers that
/// need a barrier inside a phase can use std::barrier or call this twice.
template <class F>
void parallelRun(size_t threads, F && fn)
{
    if (threads <= 1)
    {
        fn(static_cast<size_t>(0));
        return;
    }
    std::vector<std::thread> tv;
    tv.reserve(threads);
    for (size_t t = 0; t < threads; ++t)
        tv.emplace_back([t, &fn] { fn(t); });
    for (auto & th : tv)
        th.join();
}

}
