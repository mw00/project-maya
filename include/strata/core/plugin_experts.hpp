#pragma once
#include "strata/core/expert_executor.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace strata::core {
// Host-only adapter. External ordinals belong to the plugin's runtime, not CUDA.
// Caller serializes ownership updates and destroys only after its workers join.
class PluginExperts {
public:
    struct Layout { int gu=0, down=0; size_t bytes=0, down_offset=0; };
    static bool settings(std::string& path, int& device, size_t& budget, std::string& error);
    ~PluginExperts();
    bool open(const std::string& path, int device, size_t budget, int width, int ff,
              const std::vector<Layout>& layouts, const std::vector<int>& ranked, std::string& error);
    bool selected(int key) const;
    bool insert(int key, const void* blob, size_t bytes, std::string& error);
    bool replace(int out, int in, const void* blob, std::string& error);
    bool compute(int gu, int down, size_t down_offset, float clamp, int n, const int* keys,
                 const float* weights, const float* input, float* output, std::string& error);
    bool can_read(int key) const { return read_ && selected(key) && entries_[key].generation != 0; }
    bool read(int key, void* output, size_t bytes, std::string& error);
    double prefill_share() const { return prefill_share_; }
    bool can_rows(int key) const { return rows_ && selected(key) && entries_[key].generation != 0; }
    bool compute_rows(int n, int tokens, const int* keys, const int* row_tok,
                      const float* input, float clamp, float* output, std::string& error);
    bool close(std::string& error);
    size_t capacity() const { return capacity_; }
    size_t allocated() const { return allocated_; }
private:
    struct Group { Layout layout; uint32_t count=0; void* context=nullptr; };
    struct Entry { int group=-1; uint32_t slot=0; uint64_t generation=0; };
    maya_context_rows_fn rows_=nullptr;
    double prefill_share_=0;
    maya_context_read_fn read_=nullptr;
    std::atomic<uint64_t> reads_{0}, read_bytes_{0};
    void* library_=nullptr;
    const maya_context_api* api_=nullptr;
    std::vector<Group> groups_;
    std::vector<Entry> entries_;
    std::vector<Layout> layouts_;
    size_t capacity_=0, allocated_=0;
};
}
