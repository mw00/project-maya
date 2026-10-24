// src/metal/expert_pool.hpp - the routed experts of a GLM GGUF in a resident slot pool, filled from the SSD.
//
// Experts are grouped by format (gate/up/down types and shapes): each group owns three ggml tensors
// [ne0, ne1, n_slots] in memory the GPU can read, and a slot holds one (layer, expert) - the expert's three
// matrices at the same index, so a MoE graph runs mul_mat_id on the pool tensors with slot ids.  Slots are shared
// by every layer of a group and evicted least-recently-used; the slots of the layer being made resident are never
// victims, and the previous layers are done (the graph is cut at every router), so a slot can be rewritten safely.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
struct ggml_backend_device;

namespace maya::metal {

struct ReadJob {
    int fd = -1;
    uint64_t offset = 0;
    void* dst = nullptr;
    size_t bytes = 0;
};

// A fixed set of threads doing pread; read() splits the jobs into <= 1 MiB pieces and blocks until all are done.
class ParallelReader {
public:
    explicit ParallelReader(int threads);
    ~ParallelReader();
    ParallelReader(const ParallelReader&) = delete;
    ParallelReader& operator=(const ParallelReader&) = delete;
    bool read(const std::vector<ReadJob>& jobs, std::string& err);

private:
    struct State;
    State* s_;
};

struct ExpertMatrix {          // one of gate/up/down of one layer, as stored in the GGUF
    int fd = -1;
    uint64_t offset = 0;       // file offset of expert 0
    size_t expert_bytes = 0;   // nb[2]: one expert's slice
    int type = 0;              // ggml_type
    int64_t ne0 = 0, ne1 = 0;
};

class ExpertPool {
public:
    ExpertPool() = default;
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    // layers: per layer, gate/up/down (empty entries for dense layers).  budget: bytes for the slots in use now;
    // cap: bytes the pool may grow to (its address space is mapped up front, a slot takes memory once written).
    // forced_slots > 0 overrides both (per group).  dev == nullptr: plain host memory (the CPU backend).
    bool init(const std::vector<std::vector<ExpertMatrix>>& layers, int n_expert, int min_slots, size_t budget,
              size_t cap, int forced_slots, ggml_backend_device* dev, int io_threads, std::string& err);

    // take up to `bytes` more of the mapped slots into use (each group by its share); returns the bytes taken
    size_t grow(size_t bytes);
    size_t bytes_in_use() const;
    size_t bytes_cap() const;

    // the pool tensors of a layer's group (gate, up, down)
    ggml_tensor* tensor(int layer, int which) const;

    // fill the free slots with experts in this order (a ranked expert profile: most used first); the earlier an
    // expert's rank, the later LRU evicts it.  Pairs of layers without routed experts are skipped.
    bool warm(const std::vector<std::pair<int, int>>& ranked, std::string& err);

    // make experts[i] resident for `layer`; slots[i] receives its slot.  Returns false on a read error.
    bool ensure(int layer, const int32_t* experts, size_t n, int32_t* slots, std::string& err);

    // how often each (layer, expert) was picked (layer * n_expert + expert), for the learned profile
    const std::vector<uint32_t>& usage() const { return usage_; }
    void count(int layer, const int32_t* experts, size_t n);

    bool all_resident() const { return all_resident_; }
    // every routed expert has a slot in use (all resident, or a streaming pool grown that far: after each expert's
    // first use, nothing misses)
    bool holds_all() const { return all_resident_ || slots_total() >= (size_t) n_pairs_; }
    // all resident: expert e of `layer` sits in slot slot_base(layer) + e of its group
    int slot_base(int layer) const { return layers_[(size_t) layer].base; }
    size_t slots_total() const;
    size_t bytes() const { return bytes_; }
    uint64_t hits = 0, misses = 0, bytes_read = 0;
    double io_seconds = 0.0;

private:
    struct Group {
        int type[3] = {0, 0, 0};
        int64_t ne0[3] = {0, 0, 0}, ne1[3] = {0, 0, 0};
        size_t ebytes[3] = {0, 0, 0};
        int n_slots = 0;                 // the slots mapped (the tensors' ne[2])
        int active = 0;                  // the slots in use: eviction and placement stay below it
        ggml_tensor* t[3] = {nullptr, nullptr, nullptr};
        std::vector<int> owner;          // slot -> layer * n_expert + expert, -1 free
        std::vector<uint64_t> last_use;  // slot -> tick
        std::vector<uint32_t> stamp;     // slot -> ensure() call that pinned it
    };
    struct Layer {
        int group = -1;
        ExpertMatrix m[3];
        std::vector<int> slot_of;        // expert -> slot, -1 not resident
        int base = 0;                    // all resident: the layer's first slot
    };
    void pin_slots(Group& g, int from, int to);
    std::vector<Group> groups_;
    std::vector<Layer> layers_;
    ggml_context* ctx_ = nullptr;
    std::vector<ggml_backend_buffer*> bufs_;
    std::vector<std::pair<void*, size_t>> mem_;
    ParallelReader* reader_ = nullptr;
    int n_expert_ = 0;
    int n_pairs_ = 0;                    // (layer, expert) pairs with routed experts
    std::vector<uint32_t> usage_;
    uint64_t tick_ = 0;
    uint32_t call_ = 0;
    size_t bytes_ = 0;
    bool all_resident_ = false;
};

void* host_alloc(size_t bytes);           // page-aligned, zeroed; the GPU can wrap it without a copy
void host_free(void* p, size_t bytes);
// keep [p, p + bytes) in memory: unified memory the GPU reads must not be compressed or swapped out by another
// program's demand (a graph would then wait seconds on page-ins).  MAYA_METAL_NO_PIN=1 skips it.  False: refused
// (the wire limit, vm.user_wire_limit); the memory then stays pageable, said once.
bool host_pin(void* p, size_t bytes);

}  // namespace maya::metal
