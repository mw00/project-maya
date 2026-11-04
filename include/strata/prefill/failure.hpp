// include/strata/prefill/failure.hpp - a failure in the prompt path, and what kind it is.
//
// The prompt path allocates (the device buffers for a chunk, the MMQ scratch, the pinned staging) and launches
// hundreds of kernels, and none of them can return a value: they are called from the middle of a chunk's work.
// Until v1.1.27 such a failure ended the process - `std::exit(1)` in the prefill's own checks, `std::abort()` in
// the MMQ scratch pool - so an out-of-memory moment took the engine down and the client got a 503 instead of an
// answer.  A failure is now thrown instead, and the request that ran into it ends with an error; the engine lives
// and the next request runs.
//
// The recovery goal (docs/SETTINGS.md): never end the inference process for an allocation.  Where the failure
// happens before anything is computed (the layout of a chunk that does not fit), the request is read again with a
// smaller chunk; where it happens mid-chunk, the request ends with an error and the next one uses less memory.
#pragma once

#include <cctype>
#include <stdexcept>
#include <string>

namespace strata::prefill {

/// A failure in the prompt path (an allocation, a launch): the reason, so the caller can turn it into an error for
/// that request instead of ending the process.
class Failure : public std::runtime_error {
public:
    explicit Failure(const std::string& what) : std::runtime_error(what) {}
};

/// Throws a Failure with the reason (the prompt path's own check).
[[noreturn]] inline void fail(const std::string& what) { throw Failure(what); }

/// Whether the message is an allocation failure - the memory ran out ("out of memory", "does not fit", a failed
/// cudaMalloc).  Those are the ones a smaller chunk can help with; a kernel that failed for another reason cannot
/// be retried.
inline bool is_allocation_failure(const std::string& what) {
    // the phrases CUDA and this repository use for a memory failure: cudaErrorMemoryAllocation reads "out of
    // memory", a layout that does not fit reads "do not fit" / "does not fit", a failed cudaMalloc says so.
    static const char* kPhrases[] = {"out of memory",    "does not fit",  "do not fit",         "not enough memory",
                                     "not enough space", "cudamalloc",   "memoryallocation",  "insufficient memory",
                                     "allocation failed"};
    std::string low;
    low.reserve(what.size());
    for (char c : what) low.push_back((char) std::tolower((unsigned char) c));
    for (const char* p : kPhrases)
        if (low.find(p) != std::string::npos) return true;
    return false;
}

}  // namespace strata::prefill
