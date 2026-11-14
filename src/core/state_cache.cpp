#include "strata/core/state_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <deque>
#include <random>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace strata::core::statecache {
namespace {
struct Elapsed {   // adds the wall time of a scope to a counter
    double& total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Elapsed() { total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};
double number(const char* key, double fallback) {
    const char* value = std::getenv(key);
    if (!value) return fallback;
    char* end = nullptr;
    const double n = std::strtod(value, &end);
    return end != value && *end == 0 && std::isfinite(n) && n >= 0 ? n : fallback;
}
uint64_t capacity(const char* key, uint64_t fallback, double scale = 1) {
    const double value = number(key, double(fallback) / scale) * scale;
    // Conversion of an out-of-range floating value to an integer is undefined.
    if (value >= double(std::numeric_limits<uint64_t>::max())) return fallback;
    return uint64_t(value);
}
void put(Bytes& b, uint64_t n) {
    for (int i = 0; i < 8; ++i) b.push_back(uint8_t(n >> (i * 8)));
}
void put(Bytes& b, const std::string& s) {
    put(b, s.size()); b.insert(b.end(), s.begin(), s.end());
}
struct Reader {
    const Bytes& b;
    size_t at = 0;
    uint64_t n() {
        if (b.size() - at < 8) throw std::runtime_error("truncated state manifest");
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= uint64_t(b[at++]) << (i * 8);
        return v;
    }
    size_t count(size_t minimum_bytes = 1) {
        const uint64_t v = n();
        if (v > (b.size() - at) / minimum_bytes) throw std::runtime_error("invalid state manifest count");
        return size_t(v);
    }
    std::string s() {
        const size_t len = count();
        std::string v(reinterpret_cast<const char*>(b.data() + at), len); at += len; return v;
    }
};
class Files final : public Storage {
    std::filesystem::path root_;
    mutable std::mutex sizes_mutex_;
    std::mutex put_mutex_;
    std::map<std::string, uint64_t> sizes_;
#ifdef _WIN32
    HANDLE owner_ = INVALID_HANDLE_VALUE;
#else
    int owner_ = -1;
#endif
    bool valid(const std::string& key) const {
        return !key.empty() && key.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-.") == std::string::npos;
    }
public:
    explicit Files(std::string path) : root_(std::move(path)) {
        std::filesystem::create_directories(root_);
        const auto lock = root_ / "owner.lock";
#ifdef _WIN32
        owner_ = CreateFileW(lock.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (owner_ == INVALID_HANDLE_VALUE) throw std::runtime_error("state directory is owned by another engine");
#else
        owner_ = ::open(lock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (owner_ < 0 || flock(owner_, LOCK_EX | LOCK_NB) != 0) {
            if (owner_ >= 0) ::close(owner_);
            throw std::runtime_error("state directory is unavailable or owned by another engine");
        }
#endif
        std::error_code ec;
        for (const auto& f : std::filesystem::directory_iterator(root_, ec))
            if (f.is_regular_file(ec)) sizes_[f.path().filename().string()] = f.file_size(ec);
    }
    ~Files() override {
#ifdef _WIN32
        if (owner_ != INVALID_HANDLE_VALUE) CloseHandle(owner_);
#else
        if (owner_ >= 0) ::close(owner_);
#endif
    }
    bool put(const std::string& key, const Bytes& data, std::string& error) override {
        std::lock_guard serialization(put_mutex_);
        if (!valid(key)) { error = "invalid state storage key"; return false; }
        const auto path = root_ / key;
        const auto tmp = root_ / (key + ".tmp");
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        f.close();
        if (!f) {
            std::error_code cleanup; std::filesystem::remove(tmp, cleanup);
            error = "cannot write state object " + key; return false;
        }
        std::error_code ec;
#ifdef _WIN32
        if (!MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            ec = std::error_code(int(GetLastError()), std::system_category());
#else
        std::filesystem::rename(tmp, path, ec);
#endif
        if (ec) {
            error = ec.message();
            std::error_code cleanup; std::filesystem::remove(tmp, cleanup);
            return false;
        }
        { std::lock_guard lock(sizes_mutex_); sizes_[key] = data.size(); }
        return true;
    }
    bool get(const std::string& key, Bytes& data, std::string& error) override {
        if (!valid(key)) { error = "invalid state storage key"; return false; }
        std::ifstream f(root_ / key, std::ios::binary | std::ios::ate);
        const auto length = f.tellg();
        if (!f || length < 0 || uint64_t(length) > (64ull << 20)) { error = "missing or oversized state object " + key; return false; }
        try { data.resize(size_t(length)); }
        catch (const std::bad_alloc&) { error = "state object does not fit in RAM"; return false; }
        f.seekg(0);
        if (!f.read(reinterpret_cast<char*>(data.data()), length)) { error = "truncated state object " + key; return false; }
        return true;
    }
    bool erase(const std::string& key, std::string& error) override {
        if (!valid(key)) { error = "invalid state storage key"; return false; }
        std::error_code ec; std::filesystem::remove(root_ / key, ec);
        if (!ec) { std::lock_guard lock(sizes_mutex_); sizes_.erase(key); }
        if (ec) error = ec.message();
        return !ec;
    }
    std::vector<std::string> manifests(std::string& error) override {
        std::vector<std::string> result;
        std::error_code ec;
        for (const auto& f : std::filesystem::directory_iterator(root_, ec))
            if (f.path().extension() == ".state") result.push_back(f.path().filename().string());
        if (ec) error = ec.message();
        std::sort(result.begin(), result.end());
        return result;
    }
    void prune(const std::vector<std::string>& live, std::string& error) override {
        const std::set<std::string> keep(live.begin(), live.end());
        std::error_code ec;
        for (const auto& f : std::filesystem::directory_iterator(root_, ec)) {
            const auto name = f.path().filename().string();
            if ((name.starts_with("chunk-") || name.starts_with("unit-")) && !keep.count(name)) {
                std::filesystem::remove(f.path(), ec);
                if (!ec) { std::lock_guard lock(sizes_mutex_); sizes_.erase(name); }
            }
            if (ec) break;
        }
        if (ec) error = ec.message();
    }
    uint64_t local_bytes(const std::string& key, uint64_t) const override {
        std::lock_guard lock(sizes_mutex_);
        const auto it = sizes_.find(key); return it == sizes_.end() ? 0 : it->second;
    }
    double age_seconds(const std::string& key) const override {
        std::error_code ec;
        const auto written = std::filesystem::last_write_time(root_ / key, ec);
        return ec ? 0 : std::chrono::duration<double>(std::filesystem::file_time_type::clock::now()-written).count();
    }
    void touch(const std::string& key) override {
        std::error_code ec;
        std::filesystem::last_write_time(root_ / key, std::filesystem::file_time_type::clock::now(), ec);
    }
};
}

bool extends(const Prefix& a, const Prefix& b) {
    if (a.identity != b.identity || a.tokens.size() < b.tokens.size() ||
        !std::equal(b.tokens.begin(), b.tokens.end(), a.tokens.begin())) return false;
    std::vector<ExtraInput> extra;
    for (const auto& x : a.extra) if (x.position < b.tokens.size()) extra.push_back(x);
    return extra == b.extra;
}

double CostModel::seconds(uint64_t a, uint64_t b) const {
    if (b <= a) return 0;
    return double(b - a) * per_token + per_token_per_100k * double(b - a) * (double(b) + double(a)) / 200000.;
}

Config Config::environment(const std::string& dir, const std::string& family) {
    Config c;
    const auto key = [&](const char* name) { return "MAYA_" + family + "_STATE_" + name; };   // MAYA_GLM_STATE_RAM_GB
    {
        const auto old = "MAYA_" + family + "_";
        if (!std::getenv(key("RAM_GB").c_str())) {
            const auto alias = capacity((old+"SLOT_RAM_GB").c_str(), c.ram_bytes, 1073741824.);
            if (alias) c.ram_bytes = alias;
        }
        if (!std::getenv(key("DISK_GB").c_str())) c.disk_bytes = capacity((old+"SLOT_GB").c_str(), c.disk_bytes, 1073741824.);
        if (!std::getenv(key("MAX").c_str())) c.max_units = size_t(capacity((old+"SLOTS").c_str(), c.max_units));
        c.max_age_seconds = number((old+"SLOT_DAYS").c_str(), 7) * 86400.;
        bool aliases = false;
        for (const char* key : {"SLOT_RAM_GB", "SLOT_GB", "SLOT_DAYS", "SLOTS"})
            aliases = aliases || std::getenv((old+key).c_str()) != nullptr;
        if (aliases) {
            static std::once_flag notice;
            std::call_once(notice, [] { std::fprintf(stderr, "state cache: legacy slot budgets are translated; MAYA_<engine>_STATE_* settings take precedence\n"); });
        }
    }
    c.ram_bytes = capacity(key("RAM_GB").c_str(), c.ram_bytes, 1073741824.);
    c.disk_bytes = capacity(key("DISK_GB").c_str(), c.disk_bytes, 1073741824.);
    c.max_units = size_t(std::min<uint64_t>(capacity(key("MAX").c_str(), c.max_units), std::numeric_limits<size_t>::max()));
    c.idle_seconds = number(key("IDLE_SECONDS").c_str(), 60);
    c.max_age_seconds = number(key("DAYS").c_str(), c.max_age_seconds / 86400.) * 86400.;
    c.leaf_last = number(key("LEAF_LAST").c_str(), 1) != 0;
    c.anchor_tier = number(key("ANCHOR_TIER").c_str(), 1) != 0;
    c.write_through = number(key("WRITE_THROUGH").c_str(), 1) != 0;
    c.supersede_lag = size_t(capacity(key("SUPERSEDE_LAG").c_str(), 2));
    c.cost.per_token = number(key("COST_D").c_str(), 1);
    c.cost.per_token_per_100k = number(key("COST_E").c_str(), 0);
    c.calibrate = number(key("COST_CALIBRATE").c_str(), std::getenv(key("COST_D").c_str()) || std::getenv(key("COST_E").c_str()) ? 0 : 1) != 0;
    const char* path = std::getenv(key("DIR").c_str()); c.directory = path ? path : dir;
    return c;
}

std::shared_ptr<Storage> file_storage(const std::string& path) { return std::make_shared<Files>(path); }
RegionSource bytes_source(std::string name, Bytes bytes) {
    auto owned = std::make_shared<const Bytes>(std::move(bytes));
    RegionSource source;
    source.name = std::move(name); source.bytes = owned->size(); source.host = owned->data();
    source.read = [owned](uint64_t at, uint8_t* data, size_t n, std::string& error) {
        if (at > owned->size() || n > owned->size()-at) { error = "state sideband range invalid"; return false; }
        std::memcpy(data, owned->data()+at, n); return true;
    };
    return source;
}
RegionTarget check_region(std::string name, Bytes bytes) {
    auto expected = std::make_shared<const Bytes>(std::move(bytes));
    return {std::move(name), expected->size(), [expected](uint64_t at, const uint8_t* data, size_t n, std::string& error) {
        if (at > expected->size() || n > expected->size()-at || std::memcmp(data, expected->data()+at, n)) {
            error = "state sideband does not match native layout"; return false;
        }
        return true;
    }};
}

uint64_t fingerprint(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    const auto mix = [](uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };
    uint64_t sum = mix(bytes);
    const size_t words = bytes / 8;
    for (size_t i = 0; i < words; ++i) {
        uint64_t w; std::memcpy(&w, p + 8 * i, 8);
        sum += mix(w ^ (uint64_t(i) * 0x9e3779b97f4a7c15ull));
    }
    if (bytes % 8) {
        uint64_t w = 0; std::memcpy(&w, p + 8 * words, bytes % 8);
        sum += mix(w ^ (uint64_t(words) * 0x9e3779b97f4a7c15ull));
    }
    return sum;
}

struct Chunk {
    std::string key;   // its object name in storage, unique to this chunk
    uint64_t size = 0, print = 0;
    bool printed = false;   // `print` holds the bytes' fingerprint
    std::shared_ptr<const Bytes> data;
    bool stored = false;
    bool reserved_ram = false, reserved_disk = false;
    // Scratch of one bookkeeping pass (under the cache lock): the pass that last saw it, and what it counted.
    mutable uint64_t pass = 0;
    mutable uint32_t marks = 0;
};
struct Part { std::string name; std::vector<std::shared_ptr<Chunk>> chunks; };
struct StateCache::Unit {
    Prefix prefix;
    std::vector<Part> parts;
    std::string key;
    double stamp = 0, used_at = 0;
    uint64_t uses = 0;
    uint64_t min_prompt_tokens = 0;
    bool sampling_ready = false, anchor = false, frontier = false, ram = true, stored = false, pending = false, dead = false;
    std::set<Unit*> above, below;   // the held states this one extends, and those that extend it
};

struct StateCache::Impl {
    Config config;
    std::shared_ptr<Storage> storage;
    std::function<double()> clock;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, std::shared_ptr<Unit>> units;
    struct Job {
        std::shared_ptr<Unit> unit;   // write this unit's chunks and manifest
        std::string touch;            // or refresh this manifest's age
        bool payload_ready = false;
    };
    std::deque<Job> writes;
    size_t active_jobs = 0, active_updates = 0;
    Stats stats;
    double cost_xx = 0, cost_xy = 0, cost_yy = 0, cost_xt = 0, cost_yt = 0;
    bool stopping = false;
    std::thread worker;
    std::string salt;
    uint64_t named = 0;
    mutable uint64_t passes = 0;
    // Starts a bookkeeping pass; mark() then returns a chunk's marks from this pass (zero the first time).
    uint64_t begin_pass() const { return ++passes; }
    static uint32_t& mark(const Chunk& c, uint64_t pass) {
        if (c.pass != pass) { c.pass = pass; c.marks = 0; }
        return c.marks;
    }

    Impl(Config c, std::shared_ptr<Storage> s, std::function<double()> now)
        : config(std::move(c)), storage(std::move(s)), clock(std::move(now)) {
        if (!config.chunk_bytes || config.chunk_bytes > (64ull << 20)) throw std::invalid_argument("state cache chunk size must be between 1 and 64 MiB");
        stats.cost_per_token = config.cost.per_token;
        stats.cost_per_token_per_100k = config.cost.per_token_per_100k;
        if (!clock) clock = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        if (!storage && !config.directory.empty() && config.disk_bytes) storage = file_storage(config.directory);
        std::random_device random;
        char buf[24]; std::snprintf(buf, sizeof(buf), "%08x%08x", random(), random()); salt = buf;
        if (storage) {
            recover(); trim();
            std::vector<std::string> live;
            for (const auto& [key, unit] : units) {
                live.push_back(key);
                for (const auto& part : unit->parts) for (const auto& chunk : part.chunks) live.push_back(chunk->key);
            }
            std::string error;
            storage->prune(live, error);
            worker = std::thread([this] { run(); });
        }
    }
    ~Impl() {
        { std::lock_guard lock(mutex); stopping = true; cv.notify_all(); }
        if (worker.joinable()) worker.join();
    }
    void touch_later(const std::string& key) {
        if (writes.size() >= 65536) { ++stats.write_failures; return; }
        Job job; job.touch = key; writes.push_back(std::move(job)); cv.notify_all();
    }
    std::string chunk_name() {
        char buf[48]; std::snprintf(buf, sizeof(buf), "chunk-%s-%llx", salt.c_str(), (unsigned long long) ++named);
        return buf;
    }
    // Caller holds the lock; the chunk's bytes are immutable.
    uint64_t print_of(Chunk& c, const Bytes& bytes) {
        if (!c.printed) { c.print = fingerprint(bytes.data(), bytes.size()); c.printed = true; stats.hashed_bytes += bytes.size(); }
        return c.print;
    }
    // Reads one stored chunk and checks it. The caller does not hold the lock.
    std::shared_ptr<const Bytes> load(const Chunk& c, std::string& error) {
        Bytes b;
        if (!storage || !storage->get(c.key, b, error) || b.size() != c.size) {
            if (error.empty()) error = "missing state object " + c.key;
            return {};
        }
        const uint64_t print = fingerprint(b.data(), b.size());
        std::lock_guard lock(mutex);
        stats.hashed_bytes += b.size();
        if (!c.printed || print != c.print) { error = "state object checksum mismatch: " + c.key; return {}; }
        return std::make_shared<const Bytes>(std::move(b));
    }

    // The prefix graph: relations are found once, when a state is added, and removed with it.
    void link(Unit& u) {
        for (auto& [key, other] : units) {
            (void) key;
            if (other.get() == &u || other->prefix.tokens.size() == u.prefix.tokens.size()) continue;
            if (extends(u.prefix, other->prefix)) { u.above.insert(other.get()); other->below.insert(&u); }
            else if (extends(other->prefix, u.prefix)) { u.below.insert(other.get()); other->above.insert(&u); }
        }
    }
    void unlink(Unit& u) {
        for (auto* a : u.above) a->below.erase(&u);
        for (auto* d : u.below) d->above.erase(&u);
        u.above.clear(); u.below.clear();
    }
    void remove(std::map<std::string, std::shared_ptr<Unit>>::iterator it) { unlink(*it->second); units.erase(it); }
    // A state's chunks; it holds each one once (a reused chunk keeps its region and position).
    std::vector<const Chunk*> holders(const Unit& u) const {
        std::vector<const Chunk*> list;
        for (const auto& p : u.parts) for (const auto& c : p.chunks) list.push_back(c.get());
        return list;
    }
    void account() {
        stats.ram_bytes = stats.disk_bytes = stats.logical_bytes = 0;
        stats.unique_bytes = stats.shared_bytes = stats.pinned_units = 0;
        stats.pending_writes = 0;
        stats.pending_updates = active_updates;
        for (const auto& job : writes) stats.pending_updates += !job.unit;
        stats.units = units.size();
        const auto pass = begin_pass();
        for (const auto& [key, u] : units) {
            (void) key;
            stats.pending_writes += u->pending;
            stats.pinned_units += pinned(u);
            for (const auto& p : u->parts) for (const auto& c : p.chunks) {
                stats.logical_bytes += c->size;
                if (!mark(*c, pass)++) {
                    stats.unique_bytes += c->size;
                    if (c->data || c->reserved_ram) stats.ram_bytes += c->size;
                    const auto disk = c->stored && storage ? storage->local_bytes(c->key, c->size) : 0;
                    stats.disk_bytes += c->reserved_disk ? std::max(disk, c->size) : disk;
                }
            }
        }
        stats.shared_bytes = stats.logical_bytes - stats.unique_bytes;
    }
    // How many held states keep each chunk alive (a chunk's marks in the returned pass): in the RAM tier, the
    // states in RAM or pinned; on disk, all of them.
    uint64_t keepers(bool ram) const {
        const auto pass = begin_pass();
        for (const auto& [key, u] : units) {
            (void) key;
            if (ram && !u->ram && u.use_count() == 1 && !u->pending) continue;
            for (const auto& p : u->parts) for (const auto& c : p.chunks) ++mark(*c, pass);
        }
        return pass;
    }
    uint64_t freeable(const Unit& unit, bool ram, uint64_t pass) const {
        uint64_t bytes = 0;
        for (const auto& p : unit.parts) for (const auto& chunk : p.chunks) {
            const Chunk* c = chunk.get();
            if (c->pass == pass && c->marks > 1) continue;
            if (ram && (c->data || c->reserved_ram)) bytes += c->size;
            else if (!ram && c->reserved_disk) bytes += c->size;
            else if (!ram && c->stored && storage) bytes += storage->local_bytes(c->key, c->size);
        }
        return bytes;
    }
    double gain(const Unit& unit) const {
        if (unit.dead) return 0;
        size_t fallback = 0;
        for (const auto* u : unit.above) if (!u->dead) fallback = std::max(fallback, u->prefix.tokens.size());
        return config.cost.seconds(fallback, unit.prefix.tokens.size());
    }
    void retire_frontiers(const Unit& newest) {
        if (!newest.frontier) return;
        std::vector<Unit*> ancestors;
        for (auto* u : newest.above) if (u->frontier && !u->anchor && !u->dead) ancestors.push_back(u);
        std::sort(ancestors.begin(), ancestors.end(), [](const auto* a, const auto* b) {
            return a->prefix.tokens.size() > b->prefix.tokens.size();
        });
        for (size_t i = config.supersede_lag; i < ancestors.size(); ++i) {
            Unit& old = *ancestors[i];
            bool branch = false;
            for (const auto* other : old.below)
                if (!other->dead && other != &newest && !newest.above.count(const_cast<Unit*>(other)) &&
                    !newest.below.count(const_cast<Unit*>(other))) branch = true;
            if (!branch) old.dead = true;
        }
    }
    int tier(const Unit& unit) const {
        if (unit.dead) return 0;
        if (!config.leaf_last) return 1;
        size_t children = 0;
        std::set<int32_t> branches;
        for (const auto* u : unit.below)
            if (!u->dead && u->prefix.tokens.size() > unit.prefix.tokens.size()) {
                ++children;
                branches.insert(u->prefix.tokens[unit.prefix.tokens.size()]);
            }
        if ((config.anchor_tier && unit.anchor) || unit.uses > 1 || branches.size() > 1) return 2;
        if (children || (config.idle_seconds > 0 && clock() - unit.used_at >= config.idle_seconds)) return 1;
        return 2;
    }
    std::pair<int, double> score(const Unit& u, bool ram, uint64_t pass) const {
        const auto size = freeable(u, ram, pass);
        return {tier(u), size ? u.stamp + gain(u) / double(size) : std::numeric_limits<double>::infinity()};
    }
    bool pinned(const std::shared_ptr<Unit>& unit) const { return unit.use_count() > 1 || unit->pending; }

    // What names a state: its prefix and the flags a lookup depends on.
    static Bytes identity(const Unit& u) {
        Bytes b; put(b, std::string("MAYASTATE4")); put(b, u.prefix.identity);
        put(b, u.prefix.tokens.size()); for (int32_t t : u.prefix.tokens) put(b, uint32_t(t));
        put(b, u.prefix.extra.size());
        for (const auto& x : u.prefix.extra) { put(b, x.position); put(b, x.length); put(b, x.identity); }
        put(b, u.sampling_ready); put(b, u.anchor); put(b, u.frontier); put(b, u.min_prompt_tokens);
        return b;
    }
    static std::string unit_key(const Unit& u) {
        const Bytes b = identity(u);
        return "unit-" + content_digest(b.data(), b.size()) + ".state";
    }
    // Every chunk's fingerprint must be known.
    static Bytes manifest(const Unit& u) {
        Bytes b = identity(u); put(b, u.parts.size());
        for (const auto& p : u.parts) {
            put(b, p.name); put(b, p.chunks.size());
            for (const auto& c : p.chunks) { put(b, c->key); put(b, c->print); put(b, c->size); }
        }
        put(b, fingerprint(b.data(), b.size())); return b;
    }
    void recover() {
        std::string error;
        std::map<std::string, std::shared_ptr<Chunk>> known;
        for (const auto& key : storage->manifests(error)) {
            if (units.count(key)) continue;
            if (storage->age_seconds(key) > config.max_age_seconds) continue;
            Bytes b;
            if (!storage->get(key, b, error)) continue;
            try {
                Reader r{b};
                if (r.s() != "MAYASTATE4") continue;
                auto u = std::make_shared<Unit>(); u->key = key; u->prefix.identity = r.s();
                const size_t n = r.count(8);
                for (size_t i = 0; i < n; ++i) u->prefix.tokens.push_back(int32_t(r.n()));
                const size_t nx = r.count(24);
                for (size_t i = 0; i < nx; ++i) { ExtraInput x; x.position = r.n(); x.length = r.n(); x.identity = r.s(); u->prefix.extra.push_back(std::move(x)); }
                u->sampling_ready = r.n() != 0; u->anchor = r.n() != 0; u->frontier = r.n() != 0;
                u->min_prompt_tokens = r.n();
                const size_t np = r.count(16);
                for (size_t i = 0; i < np; ++i) {
                    Part p; p.name = r.s(); const size_t nc = r.count(24);
                    for (size_t j = 0; j < nc; ++j) {
                        const std::string ck = r.s(); const auto print = r.n(), size = r.n();
                        if (!size || size > config.chunk_bytes) throw std::runtime_error("invalid state chunk size");
                        auto& found = known[ck];
                        if (!found) {
                            found = std::make_shared<Chunk>(); found->key = ck; found->print = print; found->printed = true;
                            found->size = size; found->stored = true;
                        }
                        if (found->print != print || found->size != size) throw std::runtime_error("conflicting state chunk");
                        p.chunks.push_back(found);
                    }
                    u->parts.push_back(std::move(p));
                }
                const auto expected = fingerprint(b.data(), r.at);
                if (r.n() != expected || r.at != b.size() || unit_key(*u) != key) continue;
                u->ram = false; u->stored = true; u->used_at = clock(); units[key] = std::move(u);
                link(*units[key]);
            } catch (const std::exception&) { ++stats.read_failures; }
        }
        account();
    }

    void run() {
        for (;;) {
            Job job;
            { std::unique_lock lock(mutex); cv.wait(lock, [&] { return stopping || !writes.empty(); });
              if (writes.empty()) { if (stopping) return; else continue; }
              job = std::move(writes.front()); writes.pop_front(); ++active_jobs;
              active_updates += !job.unit; }
            if (!job.unit) {
                bool ok = true;
                try { storage->touch(job.touch); } catch (const std::exception&) { ok = false; }
                { std::lock_guard lock(mutex); --active_jobs; --active_updates;
                  if (!ok) ++stats.write_failures;
                  cv.notify_all(); }
                continue;
            }
            auto u = std::move(job.unit);
            std::string error;
            bool ok = true;
            try {
              if (!job.payload_ready) for (const auto& p : u->parts) for (const auto& c : p.chunks) {
                std::shared_ptr<const Bytes> data;
                { std::lock_guard lock(mutex);
                  if (c->stored) continue;
                  data = c->data;
                  if (data) print_of(*c, *data); }
                if (!data || !storage->put(c->key, *data, error)) { ok = false; break; }
                { std::lock_guard lock(mutex); c->stored = true; }
              }
              if (ok && !job.payload_ready) {
                  Bytes b;
                  { std::lock_guard lock(mutex); b = manifest(*u); }
                  ok = storage->put(u->key, b, error);
              }
            } catch (const std::exception&) { ok = false; }
            { std::lock_guard lock(mutex); u->pending = false; u->stored = ok; --active_jobs;
              if (!ok) ++stats.write_failures;
              u.reset();
              try { trim(); } catch (const std::exception&) { ++stats.write_failures; }
              cv.notify_all(); }
        }
    }
    void drop(const std::string& key) {
        auto it = units.find(key);
        if (it == units.end()) return;
        const auto owned = holders(*it->second);
        std::vector<std::shared_ptr<Chunk>> retained;
        for (const auto& part : it->second->parts) for (const auto& chunk : part.chunks) retained.push_back(chunk);
        const auto pass = begin_pass();   // marks: 1 another state holds it, 2 one of them in RAM or pinned
        for (const auto& [k, u] : units) {
            if (k == key) continue;
            const uint32_t held = u->ram || pinned(u) ? 3 : 1;
            for (const auto& p : u->parts) for (const auto& c : p.chunks) mark(*c, pass) |= held;
        }
        std::vector<std::string> erase;
        for (const auto* c : owned) if (!(mark(*c, pass) & 1) && c->stored) erase.push_back(c->key);
        remove(it); ++stats.evicted;
        for (const auto& chunk : retained) if (!(mark(*chunk, pass) & 2)) chunk->data.reset();
        if (storage) {
            try {
                std::string error;
                if (!storage->erase(key, error)) ++stats.write_failures;
                for (const auto& k : erase) if (!storage->erase(k, error)) ++stats.write_failures;
            } catch (const std::exception&) { ++stats.write_failures; }
        }
    }
    void trim() {
        // Dead units are reclaimed even when byte limits have not been reached.
        // Pending writes and reader leases keep both metadata and payload alive.
        for (auto it = units.begin(); it != units.end();) {
            if (it->second->dead && !pinned(it->second)) {
                const auto key = it++->first; drop(key);
            } else ++it;
        }
        account();
        for (;;) {
            const bool ram = stats.ram_bytes > config.ram_bytes;
            const bool disk = stats.disk_bytes > config.disk_bytes;
            const bool count = units.size() > config.max_units;
            if (!ram && !disk && !count) break;
            auto best = units.end(); std::pair<int, double> rank{3, std::numeric_limits<double>::infinity()};
            const auto kept = keepers(ram);   // (a pass: scores read its marks before any other pass)
            for (auto it = units.begin(); it != units.end(); ++it) {
                if (pinned(it->second) || (ram && !it->second->ram)) continue;
                const auto candidate = score(*it->second, ram, kept);
                if (candidate < rank) { rank = candidate; best = it; }
            }
            if (best == units.end()) break;
            if ((!config.leaf_last || rank.first == 2) && std::isfinite(rank.second))
                stats.watermark = std::max(stats.watermark, rank.second);
            if (ram && best->second->stored && !disk && !count) {
                best->second->ram = false;
                const auto pass = begin_pass();   // marks: held by a state in RAM or pinned
                for (const auto& [key, u] : units) {
                    (void) key;
                    if (u->ram || pinned(u)) for (const auto& p : u->parts) for (const auto& c : p.chunks) mark(*c, pass) = 1;
                }
                for (const auto& p : best->second->parts) for (const auto& c : p.chunks) if (!mark(*c, pass)) c->data.reset();
                ++stats.demoted;
            } else { const std::string key = best->first; drop(key); }
            account();
        }
    }
};

StateCache::StateCache(Config c, std::shared_ptr<Storage> s, std::function<double()> clock)
    : p_(std::make_unique<Impl>(std::move(c), std::move(s), std::move(clock))) {}
StateCache::~StateCache() = default;
uint64_t StateCache::Lease::tokens() const { return unit_ ? unit_->prefix.tokens.size() : 0; }
bool StateCache::Lease::sampling_ready() const { return unit_ && unit_->sampling_ready; }
const Prefix& StateCache::Lease::prefix() const { if (!unit_) throw std::logic_error("empty state lease"); return unit_->prefix; }

bool StateCache::publish(Capture capture, std::string& error) {
    error.clear();
    if (capture.prefix.tokens.empty() || capture.regions.empty()) { error = "state capture is incomplete"; return false; }
    std::vector<RegionSource> sources;
    for (auto& region : capture.regions) sources.push_back(bytes_source(std::move(region.name), std::move(region.bytes)));
    capture.regions.clear();
    return publish(std::move(capture), sources, error);
}

bool StateCache::publish(Capture metadata, const std::vector<RegionSource>& sources, std::string& error) {
    error.clear();
    if (metadata.prefix.tokens.empty() || sources.empty() || !metadata.regions.empty()) {
        error = "streaming capture metadata is incomplete"; return false;
    }
    std::unique_lock lock(p_->mutex);
    ++p_->stats.publishes;
    Elapsed elapsed{p_->stats.publish_seconds};
    if (!p_->config.max_units) { ++p_->stats.rejected; return false; }
    auto unit = std::make_shared<Unit>();
    unit->prefix = std::move(metadata.prefix); unit->anchor = metadata.anchor;
    unit->frontier = metadata.frontier; unit->sampling_ready = metadata.sampling_ready;
    unit->min_prompt_tokens = metadata.min_prompt_tokens;
    unit->stamp = p_->stats.watermark; unit->used_at = p_->clock();
    const std::string key = Impl::unit_key(*unit);
    if (const auto existing = p_->units.find(key); existing != p_->units.end()) {   // this prefix is held: no copy
        if (existing->second->dead) { ++p_->stats.rejected; return false; }
        existing->second->stamp = p_->stats.watermark; existing->second->used_at = p_->clock();
        if (p_->storage && existing->second->stored) p_->touch_later(key);
        return true;
    }
    // The nearest saved ancestor: its append-only rows are this state's rows below its end.
    const Unit* parent = nullptr;
    for (const auto& [other_key, u] : p_->units) {
        (void) other_key;
        if (!u->dead && u->prefix.tokens.size() < unit->prefix.tokens.size() &&
            (!parent || u->prefix.tokens.size() > parent->prefix.tokens.size()) && extends(unit->prefix, u->prefix))
            parent = u.get();
    }
    struct Read {
        std::shared_ptr<Chunk> chunk;
        size_t source;
        uint64_t offset;
        std::shared_ptr<const Bytes> captured;
    };
    std::vector<Read> reads;                         // new chunks, copied once the state is admitted
    std::vector<std::shared_ptr<Chunk>> members;     // every distinct chunk of the state
    std::vector<std::shared_ptr<Chunk>> reserved;
    bool complete = false;
    std::string admitted_key;
    struct Cleanup {
        Impl& cache;
        std::vector<std::shared_ptr<Chunk>>& chunks;
        std::string& key;
        bool& complete;
        ~Cleanup() {
            if (!complete && !key.empty()) {
                try { cache.drop(key); }
                catch (...) {
                    if (const auto it = cache.units.find(key); it != cache.units.end()) cache.remove(it);
                    ++cache.stats.write_failures;
                }
            }
            for (const auto& c : chunks) c->reserved_ram = c->reserved_disk = false;
            try { cache.account(); } catch (...) { ++cache.stats.write_failures; }
        }
    } cleanup{*p_, reserved, admitted_key, complete};
    try {
        const size_t step = p_->config.chunk_bytes;
        std::set<std::string> names;
        std::set<const Chunk*> seen;
        uint64_t unique = 0;
        for (size_t i = 0; i < sources.size(); ++i) {
            const auto& source = sources[i];
            if (source.name.empty() || !names.insert(source.name).second || !source.read) {
                error = "duplicate or invalid streaming state region"; return false;
            }
            const Part* before = nullptr;
            if (parent && source.rows)
                for (const auto& part : parent->parts) if (part.name == source.name) before = &part;
            std::vector<uint64_t> prints;
            if (source.fingerprints && source.bytes) {
                if (!source.fingerprints(step, prints, error)) return false;
                if (prints.size() != (source.bytes + step - 1) / step) { error = "state fingerprint count mismatch"; return false; }
            }
            Part part; part.name = source.name;
            for (uint64_t at = 0, k = 0; at < source.bytes; at += step, ++k) {
                const size_t count = size_t(std::min<uint64_t>(step, source.bytes - at));
                std::shared_ptr<Chunk> chunk;
                std::shared_ptr<const Bytes> captured;
                if (before && k < before->chunks.size() && count == step && before->chunks[k]->size == step &&
                    before->chunks[k]->data) {
                    const auto& old = before->chunks[k];
                    bool same;
                    if (!prints.empty()) same = p_->print_of(*old, *old->data) == prints[k];
                    else if (source.host) same = std::memcmp(source.host + at, old->data->data(), count) == 0;
                    else {
                        auto live = std::make_shared<Bytes>(count);
                        if (!source.read(at, live->data(), count, error)) return false;
                        p_->stats.read_bytes += count;
                        same = std::memcmp(live->data(), old->data->data(), count) == 0;
                        captured = std::move(live);
                    }
                    if (same) { chunk = old; p_->stats.reused_bytes += count; }
                }
                if (!chunk) {
                    chunk = std::make_shared<Chunk>(); chunk->key = p_->chunk_name(); chunk->size = count;
                    if (!prints.empty()) { chunk->print = prints[k]; chunk->printed = true; }
                    reads.push_back({chunk, i, at, std::move(captured)});
                }
                if (seen.insert(chunk.get()).second) {
                    if (count > UINT64_MAX - unique) { error = "state byte count overflow"; return false; }
                    unique += count; members.push_back(chunk);
                }
                part.chunks.push_back(std::move(chunk));
            }
            unit->parts.push_back(std::move(part));
        }
        if (!unique) { error = "state capture has no payload"; return false; }
        unit->ram = unique <= p_->config.ram_bytes;
        if (!unit->ram && (!p_->storage || !p_->config.disk_bytes)) { ++p_->stats.rejected; return false; }
        for (const auto& chunk : members) {
            chunk->reserved_ram = unit->ram && !chunk->data;
            chunk->reserved_disk = !unit->ram && p_->storage->local_bytes(chunk->key, chunk->size) != chunk->size;
            reserved.push_back(chunk);
        }
        // Admission before any copy: the policy sees the bytes this state would add.
        unit->key = key; admitted_key = key;
        Unit& added = *p_->units.emplace(key, std::move(unit)).first->second;   // (no local reference: a pinned unit could not be evicted)
        p_->link(added);
        p_->trim();
        const auto retained = p_->units.find(key);
        if (retained == p_->units.end()) { ++p_->stats.rejected; admitted_key.clear(); return false; }
        Unit& target = *retained->second;
        const auto fail = [&] { target.pending = false; return false; };
        target.pending = true;   // pin every member while its reserved bytes become real
        Bytes run;
        for (size_t r = 0; r < reads.size();) {
            // A run of adjacent new chunks of one region is one native read (at most 8 MiB), or a host view.
            size_t end = r + 1;
            uint64_t bytes = reads[r].chunk->size;
            if (!reads[r].captured)
                while (end < reads.size() && !reads[end].captured && reads[end].source == reads[r].source &&
                       reads[end].offset == reads[r].offset + bytes && bytes + reads[end].chunk->size <= (8ull << 20))
                    bytes += reads[end++].chunk->size;
            const auto& source = sources[reads[r].source];
            const uint8_t* from = source.host ? source.host + reads[r].offset : nullptr;
            if (!reads[r].captured && !from) {
                run.resize(size_t(bytes));
                if (!source.read(reads[r].offset, run.data(), run.size(), error)) { ++p_->stats.rejected; return fail(); }
                from = run.data();
            }
            if (!reads[r].captured) p_->stats.read_bytes += bytes;
            for (uint64_t at = 0; r < end; at += reads[r++].chunk->size) {
                auto& read = reads[r];
                auto data = read.captured ? std::move(read.captured)
                                          : std::make_shared<const Bytes>(from + at, from + at + read.chunk->size);
                if (target.ram) { read.chunk->data = std::move(data); continue; }
                p_->print_of(*read.chunk, *data);   // the state does not fit in RAM: its new bytes go to storage now
                if (!p_->storage->put(read.chunk->key, *data, error)) { ++p_->stats.write_failures; return fail(); }
                read.chunk->stored = true;
            }
        }
        if (!target.ram) {
            for (const auto& chunk : members) {   // reused chunks only RAM held so far
                if (chunk->stored && p_->storage->local_bytes(chunk->key, chunk->size) == chunk->size) continue;
                if (!chunk->data) {   // (admission evicted the RAM-only ancestor that held it)
                    error = "state chunk evicted during admission"; ++p_->stats.rejected; return fail();
                }
                p_->print_of(*chunk, *chunk->data);
                if (!p_->storage->put(chunk->key, *chunk->data, error)) { ++p_->stats.write_failures; return fail(); }
                chunk->stored = true;
            }
            if (!p_->storage->put(key, Impl::manifest(target), error)) { ++p_->stats.write_failures; return fail(); }
            target.stored = true;
        }
        for (const auto& chunk : members) chunk->reserved_ram = chunk->reserved_disk = false;
        target.pending = false;
        ++p_->stats.admitted;
        if (p_->storage && p_->config.disk_bytes && p_->config.write_through && !target.stored) {
            target.pending = true;   // the queued write pins it until the worker finishes
            Impl::Job job; job.unit = retained->second;
            p_->writes.push_back(std::move(job)); p_->cv.notify_all();
        }
        p_->retire_frontiers(target);
        complete = true; return true;
    } catch (const std::exception& e) {
        error = std::string("state capture failed: ") + e.what(); ++p_->stats.rejected; return false;
    }
}

StateCache::Lease StateCache::find_resume(const Prefix& prompt, bool need_sample) {
    std::lock_guard lock(p_->mutex);
    Lease found;
    for (const auto& [key, u] : p_->units) {
        (void) key;
        if (u->dead || u->prefix.tokens.size() < found.tokens() || prompt.tokens.size() < u->min_prompt_tokens || !extends(prompt, u->prefix) ||
            (need_sample && u->prefix.tokens.size() == prompt.tokens.size() && !u->sampling_ready)) continue;
        found.unit_ = u;
    }
    if (found) {
        found.unit_->stamp = p_->stats.watermark; found.unit_->used_at = p_->clock(); ++found.unit_->uses;
        ++p_->stats.hits;
        bool ram = true;
        for (const auto& part : found.unit_->parts) for (const auto& chunk : part.chunks) ram = ram && bool(chunk->data);
        if (ram) ++p_->stats.ram_hits; else ++p_->stats.disk_hits;
        if (p_->storage && found.unit_->stored) p_->touch_later(found.unit_->key);
    }
    else ++p_->stats.misses;
    p_->trim();
    return found;
}

bool StateCache::restore(const Lease& lease, std::vector<Region>& regions, std::string& error) {
    error.clear(); regions.clear();
    if (!lease) { error = "empty state lease"; return false; }
    { std::lock_guard lock(p_->mutex);
      if (lease.unit_->dead) { error = "state recovery unit was invalidated"; return false; } }
    uint64_t total = 0;
    for (const auto& part : lease.unit_->parts) for (const auto& chunk : part.chunks) {
        if (chunk->size > p_->config.ram_bytes - total) { error = "state restore exceeds host budget"; return false; }
        total += chunk->size;
    }
    try {
    std::vector<Region> loaded;
    std::map<Chunk*, std::shared_ptr<const Bytes>> fetched;
    for (const auto& part : lease.unit_->parts) {
        Region region; region.name = part.name;
        for (const auto& c : part.chunks) {
            std::shared_ptr<const Bytes> bytes;
            { std::lock_guard lock(p_->mutex); bytes = c->data; }
            if (!bytes) {
                const auto previous = fetched.find(c.get());
                if (previous != fetched.end()) bytes = previous->second;
                else if (!(bytes = p_->load(*c, error))) {
                    std::lock_guard lock(p_->mutex); ++p_->stats.read_failures; lease.unit_->dead = true;
                    return false;
                } else fetched[c.get()] = bytes;
            }
            region.bytes.insert(region.bytes.end(), bytes->begin(), bytes->end());
        }
        loaded.push_back(std::move(region));
    }
    {
        std::lock_guard lock(p_->mutex);
        if (lease.unit_->dead) { error = "state recovery unit was invalidated during load"; return false; }
        if (!fetched.empty()) {
            std::vector<Chunk*> promoted;
            for (const auto& [chunk, bytes] : fetched) if (!chunk->data) { chunk->data = bytes; promoted.push_back(chunk); }
            const bool old_ram = lease.unit_->ram;
            lease.unit_->ram = true;
            p_->trim();
            if (p_->stats.ram_bytes > p_->config.ram_bytes) {
                for (auto* chunk : promoted) chunk->data.reset();
                lease.unit_->ram = old_ram; p_->account();
            } else ++p_->stats.promoted;
        }
    }
    regions = std::move(loaded);
    return true;
    } catch (const std::exception& e) {
        error = std::string("state restore failed: ") + e.what();
        std::lock_guard lock(p_->mutex); ++p_->stats.read_failures;
        return false;
    }
}
bool StateCache::restore(const Lease& lease, const std::vector<RegionTarget>& targets, std::string& error) {
    error.clear();
    if (!lease) { error = "empty state lease"; return false; }
    struct Timed {
        Impl& cache;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        ~Timed() {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::lock_guard lock(cache.mutex); ++cache.stats.restores; cache.stats.restore_seconds += s;
        }
    } timed{*p_};
    { std::lock_guard lock(p_->mutex);
      if (lease.unit_->dead) { error = "state recovery unit was invalidated"; return false; } }
    try {
        std::map<std::string, const Part*> parts;
        for (const auto& part : lease.unit_->parts)
            if (!parts.emplace(part.name, &part).second) { error = "duplicate recovery region"; return false; }
        std::set<std::string> names;
        if (parts.size() != targets.size()) { error = "incomplete native recovery layout"; return false; }
        for (const auto& target : targets) {
            const auto found = parts.find(target.name);
            uint64_t size = 0;
            if (found == parts.end() || !target.write || !names.insert(target.name).second) {
                error = "invalid native recovery region"; return false;
            }
            for (const auto& chunk : found->second->chunks) {
                if (chunk->size > UINT64_MAX-size) { error = "recovery byte count overflow"; return false; }
                size += chunk->size;
            }
            if (size != target.bytes) { error = "native recovery size mismatch: " + target.name; return false; }
        }
        uint64_t unique_bytes = 0;
        std::set<const Chunk*> unique;
        bool cache_payload = true, promoted = false;
        for (const auto& part : lease.unit_->parts) for (const auto& chunk : part.chunks)
            if (unique.insert(chunk.get()).second) {
                if (chunk->size > p_->config.ram_bytes-unique_bytes) cache_payload = false;
                else unique_bytes += chunk->size;
            }
        for (const auto& target : targets) {
            uint64_t at = 0;
            const auto& chunks = parts.at(target.name)->chunks;
            for (size_t begin = 0; begin < chunks.size();) {
                size_t end = begin;
                uint64_t batch_bytes = 0;
                do { batch_bytes += chunks[end++]->size; }
                while (end < chunks.size() && chunks[end]->size <= (4ull << 20) - std::min<uint64_t>(batch_bytes, 4ull << 20));
                std::vector<std::shared_ptr<const Bytes>> data(end-begin);
                bool loaded = false;
                {
                    std::lock_guard lock(p_->mutex);
                    if (lease.unit_->dead) { error = "state invalidated during native load"; return false; }
                    for (size_t i = begin; i < end; ++i) data[i-begin] = chunks[i]->data;
                }
                for (size_t i = begin; i < end; ++i) {
                    if (data[i-begin]) continue;
                    if (!(data[i-begin] = p_->load(*chunks[i], error))) {
                        std::lock_guard lock(p_->mutex); ++p_->stats.read_failures; lease.unit_->dead = true;
                        return false;
                    }
                    loaded = true;
                }
                if (data.size() == 1) {
                    if (!target.write(at, data[0]->data(), data[0]->size(), error)) return false;
                } else {   // one native write a batch
                    Bytes joined; joined.reserve(size_t(batch_bytes));
                    for (const auto& bytes : data) joined.insert(joined.end(), bytes->begin(), bytes->end());
                    if (!target.write(at, joined.data(), joined.size(), error)) return false;
                }
                at += batch_bytes;
                // Small recovery units can return to RAM without retaining an
                // extra full-unit buffer. Account each verified batch before the
                // next read, and roll it back if active pins prevent admission.
                if (cache_payload && loaded) {
                    std::lock_guard lock(p_->mutex);
                    if (lease.unit_->dead) { error = "state invalidated during native load"; return false; }
                    std::vector<Chunk*> fresh;
                    for (size_t i = begin; i < end; ++i) if (!chunks[i]->data) {
                        chunks[i]->data = data[i-begin]; fresh.push_back(chunks[i].get());
                    }
                    if (!fresh.empty()) {
                        const bool before = lease.unit_->ram;
                        lease.unit_->ram = true; p_->trim();
                        if (p_->stats.ram_bytes > p_->config.ram_bytes) {
                            for (auto* chunk : fresh) chunk->data.reset();
                            lease.unit_->ram = before; p_->account();
                        } else promoted = true;
                    }
                }
                begin = end;
            }
        }
        if (promoted) { std::lock_guard lock(p_->mutex); ++p_->stats.promoted; }
        return true;
    } catch (const std::exception& e) {
        error = std::string("native state load failed: ") + e.what();
        std::lock_guard lock(p_->mutex); ++p_->stats.read_failures; return false;
    }
}

bool StateCache::read_region(const Lease& lease, const std::string& name, Bytes& bytes, std::string& error, uint64_t limit) {
    bytes.clear(); error.clear();
    if (!lease) { error = "empty state lease"; return false; }
    { std::lock_guard lock(p_->mutex);
      if (lease.unit_->dead) { error = "invalidated state lease"; return false; } }
    const Part* found = nullptr;
    for (const auto& part : lease.unit_->parts) if (part.name == name) {
        if (found) { error = "duplicate recovery region"; return false; }
        found = &part;
    }
    if (!found) { error = "recovery sideband missing: " + name; return false; }
    uint64_t size = 0;
    for (const auto& chunk : found->chunks) {
        if (chunk->size > limit-size) { error = "recovery sideband exceeds bound"; return false; }
        size += chunk->size;
    }
    try {
        Bytes result; result.reserve(size_t(size));
        for (const auto& chunk : found->chunks) {
            std::shared_ptr<const Bytes> data;
            { std::lock_guard lock(p_->mutex); data = chunk->data; }
            if (!data && !(data = p_->load(*chunk, error))) return false;
            result.insert(result.end(), data->begin(), data->end());
        }
        bytes = std::move(result); return true;
    } catch (const std::exception& e) { error = std::string("recovery sideband failed: ") + e.what(); return false; }
}

void StateCache::restored(Lease& lease) {
    std::lock_guard lock(p_->mutex);
    if (lease && !lease.restored_ && !lease.unit_->dead) {
        p_->stats.restored_tokens += lease.tokens(); lease.restored_ = true;
    }
}
void StateCache::release(Lease& lease) {
    std::lock_guard lock(p_->mutex);
    if (lease.unit_) { lease.unit_->stamp = p_->stats.watermark; lease.unit_->used_at = p_->clock(); lease.unit_.reset(); }
    p_->trim();
}
void StateCache::supersede(const Prefix& newest, size_t keep) {
    std::lock_guard lock(p_->mutex);
    std::vector<Unit*> ancestors;
    for (auto& [key, u] : p_->units) {
        (void) key;
        if (!u->anchor && u->prefix.tokens.size() < newest.tokens.size() && extends(newest, u->prefix)) ancestors.push_back(u.get());
    }
    std::sort(ancestors.begin(), ancestors.end(), [](const Unit* a, const Unit* b) { return a->prefix.tokens.size() > b->prefix.tokens.size(); });
    for (size_t i = keep; i < ancestors.size(); ++i) if (p_->tier(*ancestors[i]) < 2) ancestors[i]->dead = true;
    p_->trim();
}
void StateCache::flush() {
    std::unique_lock lock(p_->mutex);
    p_->cv.wait(lock, [&] { return p_->writes.empty() && !p_->active_jobs; });
}
void StateCache::invalidate(const Prefix& failed) {
    std::lock_guard lock(p_->mutex);
    for (auto& [key, u] : p_->units) {
        (void) key;
        if (extends(failed, u->prefix) || extends(u->prefix, failed)) u->dead = true;
    }
    p_->trim();
}
Stats StateCache::stats() const { std::lock_guard lock(p_->mutex); p_->account(); return p_->stats; }
std::string StateCache::report() const {
    try {
    const auto s = stats();
    std::ostringstream out;
    out << std::setprecision(12)
        << "hits=" << s.hits << " misses=" << s.misses << " restored_tokens=" << s.restored_tokens
        << " units=" << s.units << " ram_bytes=" << s.ram_bytes << " disk_bytes=" << s.disk_bytes
        << " logical_bytes=" << s.logical_bytes << " admitted=" << s.admitted << " rejected=" << s.rejected
        << " evicted=" << s.evicted << " promoted=" << s.promoted << " demoted=" << s.demoted
        << " read_failures=" << s.read_failures << " write_failures=" << s.write_failures
        << " watermark=" << s.watermark << " cost_d=" << s.cost_per_token << " cost_e=" << s.cost_per_token_per_100k
        << " calibration_samples=" << s.calibration_samples << " pending_writes=" << s.pending_writes
        << " pending_updates=" << s.pending_updates;
    out << " publishes=" << s.publishes << " publish_ms=" << s.publish_seconds * 1000 << " read_bytes=" << s.read_bytes
        << " reused_bytes=" << s.reused_bytes << " hashed_bytes=" << s.hashed_bytes << " restores=" << s.restores
        << " restore_ms=" << s.restore_seconds * 1000;
    out << " unique_bytes=" << s.unique_bytes << " shared_bytes=" << s.shared_bytes << " pinned_units=" << s.pinned_units
        << " ram_hits=" << s.ram_hits << " disk_hits=" << s.disk_hits;
    return out.str();
    } catch (const std::bad_alloc&) { return {}; }
}

void StateCache::observe_prefill(uint64_t from, uint64_t to, double seconds) {
    if (to <= from || !std::isfinite(seconds) || seconds <= 0) return;
    std::lock_guard lock(p_->mutex);
    if (!p_->config.calibrate || p_->stats.calibration_samples >= 16) return;
    const double x = double(to - from), y = x * (double(to) + double(from)) / 200000.;
    p_->cost_xx += x * x; p_->cost_xy += x * y; p_->cost_yy += y * y;
    p_->cost_xt += x * seconds; p_->cost_yt += y * seconds;
    ++p_->stats.calibration_samples;
    double d = p_->cost_xt / p_->cost_xx, e = 0;
    const double determinant = p_->cost_xx * p_->cost_yy - p_->cost_xy * p_->cost_xy;
    if (determinant > 1e-10 * p_->cost_xx * p_->cost_yy) {
        d = (p_->cost_xt * p_->cost_yy - p_->cost_yt * p_->cost_xy) / determinant;
        e = (p_->cost_yt * p_->cost_xx - p_->cost_xt * p_->cost_xy) / determinant;
        if (e < 0) { e = 0; d = p_->cost_xt / p_->cost_xx; }
        if (d < 0) { d = 0; e = p_->cost_yt / p_->cost_yy; }
    }
    if (!std::isfinite(d) || !std::isfinite(e) || d + e <= 0) return;
    p_->config.cost = {d, e};
    p_->stats.cost_per_token = d; p_->stats.cost_per_token_per_100k = e;
    // During initial calibration, clock stamps and gains must use the same units.
    // Freeze the fitted model after sixteen observations; ordinary aging follows.
    p_->stats.watermark = 0;
    for (auto& [key, unit] : p_->units) { (void) key; unit->stamp = 0; }
}

ExportPlan plan_export(uint64_t start, uint64_t end, const std::vector<uint64_t>& marks, uint64_t alignment, bool supported) {
    ExportPlan plan{std::max(start, end), 0};
    std::vector<uint64_t> inner;
    for (auto p : marks) if (p > start && p < end) inner.push_back(p);
    std::sort(inner.begin(), inner.end()); inner.erase(std::unique(inner.begin(), inner.end()), inner.end());
    if (inner.empty()) return plan;
    if (!supported || !alignment || (inner[0] - start) % alignment) return {inner[0], 0};
    return {inner.size() > 1 ? inner[1] : end, inner[0]};
}
bool Commit::exact(uint64_t& endpoint) const {
    if (!completed || kept != accepted || accepted > processed || start > std::numeric_limits<uint64_t>::max() - accepted) return false;
    endpoint = start + accepted; return true;
}

}  // namespace strata::core::statecache
