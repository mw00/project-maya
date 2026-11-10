// Compile the patched server itself against an in-memory stateful model.
#include "serve.cpp"
#include <cassert>
#include <functional>

namespace maya::metal {
using Tokens = std::vector<int32_t>;
static Tokens state, snapshot;
static bool have_snapshot = false;
static int restores = 0;
#ifdef TEST_METAL_SLOTS
static int disk_saves = 0, disk_loads = 0;
#endif
static std::function<void(const Tokens&)> on_forward;
struct GlmMetal::Impl {};
GlmMetal::GlmMetal() : p_(new Impl) {}
GlmMetal::~GlmMetal() = default;
bool GlmMetal::forward(const Tokens& t, std::vector<float>& logits, std::string&) {
    state.insert(state.end(), t.begin(), t.end());
    logits.assign((size_t) geometry().n_vocab, 0);
    if (on_forward) on_forward(t);
    return true;
}
void GlmMetal::reset() { state.clear(); have_snapshot = false; }
void GlmMetal::snapshot_save() { snapshot = state; have_snapshot = true; }
bool GlmMetal::snapshot_restore() { if (!have_snapshot) return false; state = snapshot; ++restores; return true; }
int64_t GlmMetal::snapshot_position() const { return have_snapshot ? (int64_t) snapshot.size() : -1; }
#ifdef TEST_METAL_SLOTS
uint64_t GlmMetal::slot_snapshot_size() const { return have_snapshot ? snapshot.size() * sizeof(int32_t) : 0; }
bool GlmMetal::slot_snapshot_save(const std::string& path, uint64_t bytes) {
    if (bytes != slot_snapshot_size()) return false;
    FILE* f = std::fopen(path.c_str(), "wb"); if (!f) return false;
    bool ok = std::fwrite(snapshot.data(), 1, (size_t)bytes, f) == bytes;
    if (std::fclose(f)) ok = false;
    if (ok) ++disk_saves;
    return ok;
}
bool GlmMetal::slot_snapshot_load(const std::string& path, int64_t pos, uint64_t budget) {
    if (pos <= 0 || (uint64_t)pos > budget / sizeof(int32_t)) return false;
    FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return false;
    Tokens v((size_t)pos);
    bool ok = std::fread(v.data(), sizeof(int32_t), v.size(), f) == v.size() && std::fgetc(f) == EOF;
    if (std::fclose(f)) ok = false;
    if (!ok) { reset(); return false; }
    state = snapshot = std::move(v); have_snapshot = true; ++disk_loads; return true;
}
#endif
int64_t GlmMetal::position() const { return (int64_t) state.size(); }
const GlmGeometry& GlmMetal::geometry() const { static GlmGeometry g = [] { GlmGeometry x; x.n_vocab = 154855; x.n_embd = 1; return x; }(); return g; }
const GlmStats& GlmMetal::stats() const { static GlmStats s; return s; }
int64_t GlmMetal::max_context() const { return 131072; }
int GlmMetal::ubatch() const { return 2; }
bool GlmMetal::has_mtp() const { return false; }
bool GlmMetal::has_dflash() const { return false; }
int GlmMetal::dflash_block() const { return 0; }
bool GlmMetal::dflash_draft(int32_t, int, float, int, float, Tokens&, std::string&) { return false; }
size_t GlmMetal::grow_pool() { return 0; }
bool GlmMetal::pool_holds_all() const { return true; }
bool GlmMetal::save_profile(std::string&) const { return true; }
std::string GlmMetal::describe() const { return "fake"; }
void GlmMetal::set_image_rows(std::vector<int64_t>, std::vector<float>) {}
int32_t pick(const std::vector<float>&, const Sampling&, const Tokens&) { return 7; }
DraftLength::DraftLength(int n) : cap_(n), start_(0), fixed_(0), adapt_(false) {}
bool decode_spec(GlmMetal&, int32_t, int, const Drafter&, const std::function<void(int,int,double)>&, const DecodeHooks&, std::string&) { return false; }
bool decode_mtp(GlmMetal&, int32_t, DraftLength&, const DecodeHooks&, std::string&) { return false; }
}
int main() {
    using namespace maya::metal;
    constexpr int32_t A = 154828, T = 154841, U = 154827;
    assert(glm_snapshot_target({}) == 0);
    assert(glm_snapshot_target({7}) == 0);
    assert(glm_snapshot_target({7, A, T}) == 1);
    assert(glm_snapshot_target({7, A, 1}) == 2);
    #ifdef TEST_METAL_SLOTS
    setenv("MAYA_METAL_SLOTS", "0", 1);
    #endif
    GlmMetal model;
    Server server{model, {}, {}};
    Sampling sampling;
    Spec spec;
    Tokens first{1,2,3,A,T}, next{1,2,3,U,4,A,T};
    on_forward = [&](const Tokens& t) { if (t == Tokens{A,T}) server.in.stop.store(true); };
    server.run(first, 4, sampling, 0, -1, spec);
    assert(server.cached == Tokens({1,2,3}) && snapshot == server.cached);
    server.in.stop.store(false); on_forward = {};
    server.run(next, 1, sampling, 0, -1, spec);
    assert(restores == 1 && state == next && server.cached == Tokens({1,2,3,U,4}));
    on_forward = [&](const Tokens& t) { if (t == Tokens{7}) { server.in.pause.store(true); server.in.stop.store(true); } };
    server.run(next, 4, sampling, 0, -1, spec);
    Tokens paused = next; paused.push_back(7);
    assert(server.cached == paused && snapshot == paused);
    server.in.stop.store(false); server.in.pause.store(false); on_forward = {};
    Tokens follow = paused; follow.push_back(7); follow.push_back(8);
    int before = restores;
    server.run(follow, 1, sampling, 0, -1, spec);
    assert(restores == before + 1 && state == follow);
    server.in.pause_keep.store(false);
    on_forward = [&](const Tokens& t) { if (t == Tokens{7}) { server.in.pause_keep.store(true); server.in.pause.store(true); server.in.stop.store(true); } };
    server.run(next, 4, sampling, 0, -1, spec);
    assert(server.cached == Tokens({1,2,3,U,4}) && snapshot == server.cached);
    server.in.stop.store(false); server.in.pause.store(false); server.in.pause_keep.store(false); on_forward = {};
    before = restores;
    server.run(Tokens{1,2,3,U,4,U,5,A,T}, 1, sampling, 0, -1, spec);
    assert(restores == before + 1);
    before = restores;
    server.run(next, 1, sampling, 0, -1, spec); // a longer snapshot cannot serve a shorter target
    assert(restores == before && state == next);
    before = restores;
    server.run(Tokens{1,2,6,U,4,A,T}, 1, sampling, 0, -1, spec);
    assert(restores == before);
#ifdef TEST_METAL_SLOTS
    setenv("MAYA_METAL_SLOTS", "2", 1);
    setenv("MAYA_METAL_SLOT_MIN", "1", 1);
    setenv("MAYA_METAL_SLOT_DIR", "/Volumes/NVM/AI_Agent_Runtime/tmp/maya-test-slots", 1);
    model.reset();
    Server multi{model, {}, {}};
    Tokens alpha{10,11,12,A,T}, beta{20,21,22,A,T};
    multi.run(alpha, 1, sampling, 0, -1, spec);
    multi.run(beta, 1, sampling, 0, -1, spec);
    assert(disk_saves == 1);
    multi.run(alpha, 1, sampling, 0, -1, spec);
    assert(disk_loads == 1 && state == alpha && multi.cached == Tokens({10,11,12}));
#endif
}
