// maya-ds4's incremental GEN line (src/ds4/ds4_keep.hpp, DS4_GEN_KEEP): the hash is the server's
// (hashlib.blake2b(digest_size=16) over the ids as little-endian int32 - the vectors below are Python's), the running
// hash is the hash of the state's own first tokens whatever happened to the state between two requests (it grew, was
// cut, went back and was written again, another conversation took the model), and keep_ids gives the ids the whole
// line would have carried or refuses.  No GPU, no model: a state here is its tokens.
// `ds4_keep_test bench` also times the engine's side of a prompt line both ways at 220K tokens: the whole line through
// a pipe in 4 KB reads, the reader's and the queue's copies and the parse (the serve loop's Lines reader, parse_gen's
// token copy and parse_ids, as in ds4_main.cpp), against the line with keep= (the same path over the new ids, then
// the hash check and the kept start copied from the state).  `ds4_keep_test bench <whole> <keep>` times two lines from
// files instead (a real prompt's, as the server writes them both ways).
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_keep.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <string>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using maya::ds4::Blake2b128;
using maya::ds4::keep_ids;
using maya::ds4::KeepHash;

namespace {

std::vector<int32_t> seq(size_t n) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (int32_t) ((i * 2654435761ull + 12345ull) % 200003ull);
    return v;
}

// the hash of v[0 .. n) from nothing, a byte at a time or in odd pieces (the reference for the running one)
std::string plain(const std::vector<int32_t>& v, size_t n, size_t piece = 0) {
    std::vector<uint8_t> b(4 * n);
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 4; ++k) b[4 * i + (size_t) k] = (uint8_t) ((uint32_t) v[i] >> (8 * k));
    Blake2b128 h;
    if (piece == 0) h.update(b.data(), b.size());
    else
        for (size_t at = 0; at < b.size(); at += piece) h.update(b.data() + at, std::min(piece, b.size() - at));
    return h.hex();
}

void vectors() {
    const struct { size_t n; const char* hex; } py[] = {   // hashlib.blake2b(array('i', ids).tobytes(), digest_size=16)
        {0, "cae66941d9efbd404e4d88758ea67670"},     {1, "dbdd6d6fbffd502a24e84e44d1156c24"},
        {31, "8b718e81b8a17bc52db0fb19a94918ba"},    {32, "448c539b6b4afd2d92b696b633cc4096"},
        {33, "51b1f9d881a5344cc85bab694faa3927"},    {63, "add31323c0fb2e2d0421fe061d37f56d"},
        {64, "64258d4bfb69b2a1f743cba83e54807d"},    {65, "170b9bf9c498b6c07cd65c53030a4c4a"},
        {1000, "24c30154f98a7581b30ec7aa1917843f"},  {220000, "a93958be56e8ab9bd77ffac9d267316c"}};
    for (const auto& c : py) {
        const std::vector<int32_t> v = seq(c.n);
        assert(plain(v, c.n) == c.hex);
        for (size_t piece : {1, 3, 127, 128, 129, 1000}) assert(plain(v, c.n, piece) == c.hex);
        KeepHash kh;
        assert(kh.hex(v.data(), c.n) == c.hex);
        assert(kh.hex(v.data(), c.n) == c.hex);   // (asked again: the running state was not finished by the first)
    }
}

// a state that lives as the engine's does, the running hash asked at random places: always the plain hash
void running() {
    std::mt19937 rng(20261121);
    for (int round = 0; round < 60; ++round) {
        std::vector<int32_t> st;
        KeepHash kh;
        for (int step = 0; step < 80; ++step) {
            const int op = (int) (rng() % 10);
            if (op < 5) {   // the conversation grows (a prompt's new tokens, an answer)
                const size_t add = rng() % 700;
                for (size_t i = 0; i < add; ++i) st.push_back((int32_t) (rng() % 200000));
            } else if (op < 7 && !st.empty()) {   // cut back (a restore point, an unwind, a STOP mid-read)
                st.resize(rng() % (st.size() + 1));
            } else if (op == 7 && !st.empty()) {   // one token of the kept part is another (an edit, the same length)
                st[rng() % st.size()] ^= 1 + (int32_t) (rng() % 7);
            } else if (op == 8) {   // another conversation takes the model
                st.assign(rng() % 3000, 0);
                for (int32_t& x : st) x = (int32_t) (rng() % 200000);
            }
            const size_t n = st.empty() ? 0 : rng() % (st.size() + 1);
            assert(kh.hex(st.data(), n) == plain(st, n));
            assert(kh.seen.size() == n);
        }
    }
}

void keep() {
    const std::vector<int32_t> st = seq(5000);
    const std::vector<int> tail = {7, 8, 9};
    KeepHash kh;
    // taken: the state's first n and the line's ids, whatever the state holds past n
    for (size_t n : {1, 4096, 4999, 5000, 300}) {
        std::vector<int> ids = tail;
        assert(keep_ids(kh, st, (long long) n, plain(st, n), ids));
        assert(ids.size() == n + 3);
        for (size_t i = 0; i < n; ++i) assert(ids[i] == st[i]);
        assert(ids[n] == 7 && ids[n + 2] == 9);
    }
    // refused, the ids untouched: more than the state holds (a fresh engine, a read that was stopped), nothing kept,
    // no ids, a hash that is another prompt's, of another count, cut short, or not a hash
    const auto refused = [&](const std::vector<int32_t>& s, long long n, const std::string& h) {
        std::vector<int> ids = tail;
        const bool ok = keep_ids(kh, s, n, h, ids);
        assert(ids == tail);
        return !ok;
    };
    assert(refused(st, 5001, plain(st, 5000)));
    assert(refused({}, 10, plain(st, 10)));
    assert(refused(st, 0, plain(st, 0)));
    assert(refused(st, -5, plain(st, 5)));
    assert(refused(st, 100, plain(st, 101)));
    assert(refused(st, 100, plain(st, 100).substr(0, 31)));
    assert(refused(st, 100, ""));
    assert(refused(st, 100, std::string(32, 'z')));
    std::vector<int32_t> other = st;
    other[50] += 1;   // (the same count, one token another: the server's prompt is not the state's)
    assert(refused(other, 100, plain(st, 100)));
    assert(refused(other, 5000, plain(st, 5000)));
    {
        std::vector<int> none;
        assert(!keep_ids(kh, st, 100, plain(st, 100), none));
    }
    {   // ... and still taken where the two agree, after all the refusals
        std::vector<int> ids = tail;
        assert(keep_ids(kh, other, 50, plain(st, 50), ids) && ids.size() == 53);
    }
}

#if !defined(_WIN32)
// ds4_main.cpp's parse_ids and its Lines reader's getline (4 KB reads into a growing buffer), as they are there
std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int) x);
        p = e;
        while (*p == ',' || *p == ' ') ++p;
    }
    return v;
}
bool getline_fd(int fd, std::string& buf, std::string& out) {
    char chunk[4096];
    size_t scanned = 0;
    for (;;) {
        const size_t nlpos = buf.find('\n', scanned);
        if (nlpos != std::string::npos) {
            out.assign(buf, 0, nlpos);
            buf.erase(0, nlpos + 1);
            return true;
        }
        scanned = buf.size();
        const ssize_t n = ::read(fd, chunk, sizeof chunk);
        if (n <= 0) return false;
        buf.append(chunk, (size_t) n);
    }
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// one request's line from the pipe to r.ids: the read, the queue's two copies, the ids token's copy, the parse, and
// with keep= the check and the kept start
double one(const std::string& line, const std::vector<int32_t>* st, KeepHash* kh, long long keep, const std::string& h,
           std::vector<int>& ids) {
    int fd[2];
    assert(::pipe(fd) == 0);
    std::thread w([&] {
        size_t o = 0;
        while (o < line.size()) o += (size_t) ::write(fd[1], line.data() + o, line.size() - o);
    });
    const double t0 = now_ms();
    std::string buf, l, got;
    assert(getline_fd(fd[0], buf, l));
    std::deque<std::string> q;
    q.push_back(l);
    got = q.front();
    q.pop_front();
    const size_t sp = got.rfind(' ');
    const std::string tok(got, sp + 1);
    ids = parse_ids(tok);
    if (st != nullptr) assert(keep_ids(*kh, *st, keep, h, ids));
    const double t1 = now_ms();
    w.join();
    ::close(fd[0]), ::close(fd[1]);
    return t1 - t0;
}
std::string slurp(const char* path) {
    std::string s;
    std::FILE* f = std::fopen(path, "rb");
    assert(f != nullptr);
    char b[65536];
    for (size_t k; (k = std::fread(b, 1, sizeof b, f)) > 0;) s.append(b, k);
    std::fclose(f);
    return s;
}
void bench(const char* whole_path, const char* keep_path) {
    size_t n = 220000, add = 160;   // (a turn of the live bench: ~160 new tokens on a 220K conversation)
    std::vector<int32_t> prompt = seq(n + add);
    std::string whole_file, keep_file;
    if (whole_path != nullptr) {   // the server's two lines for one prompt: n is the keep= line's
        whole_file = slurp(whole_path), keep_file = slurp(keep_path);
        const std::vector<int> v = parse_ids(whole_file.substr(whole_file.rfind(' ') + 1));
        prompt.assign(v.begin(), v.end());
        n = (size_t) std::atoll(keep_file.c_str() + keep_file.find("keep=") + 5);
        add = prompt.size() - n;
    }
    const std::vector<int32_t> last(prompt.begin(), prompt.begin() + (long) n);
    std::vector<int32_t> st = last;       // the state: the last prompt and an answer after it
    for (int i = 0; i < 300; ++i) st.push_back(i);
    const auto text = [&](size_t from) {
        std::string s;
        for (size_t i = from; i < prompt.size(); ++i) s += (i > from ? "," : "") + std::to_string(prompt[i]);
        return s;
    };
    KeepHash kh;
    const double tc0 = now_ms();
    const std::string hx = kh.hex(st.data(), n - 2000);   // (the turn before: the running hash stands 2000 tokens back)
    const double tc1 = now_ms();
    (void) hx;
    const std::string whole = whole_path != nullptr ? whole_file : "GEN 24 temperature=1.0 " + text(0) + "\n";
    const std::string delta = whole_path != nullptr ? keep_file :
        "GEN 24 temperature=1.0 keep=" + std::to_string(n) + " h=" + plain(last, n) + " " + text(n) + "\n";
    if (whole_path != nullptr)   // (the server's hash is the engine's)
        assert(delta.find(" h=" + plain(last, n) + " ") != std::string::npos);
    std::vector<double> a, b;
    std::vector<int> ia, ib;
    for (int rep = 0; rep < 21; ++rep) {
        a.push_back(one(whole, nullptr, nullptr, 0, "", ia));
        KeepHash k2 = kh;   // (every repeat from the same running state: 2000 tokens to hash)
        b.push_back(one(delta, &st, &k2, (long long) n, plain(last, n), ib));
        assert(ia == ib && ia.size() == n + add);
    }
    std::sort(a.begin(), a.end()), std::sort(b.begin(), b.end());
    std::printf("engine side of one prompt line, %zu tokens, %zu new (median of 21; min .. max):\n", n + add, add);
    std::printf("  whole line (%.2f MB): %.2f ms (%.2f .. %.2f)\n", (double) whole.size() / 1e6, a[10], a.front(), a.back());
    std::printf("  keep= line (%zu bytes): %.3f ms (%.3f .. %.3f)\n", delta.size(), b[10], b.front(), b.back());
    std::printf("  the hash of %zu tokens from nothing (a first keep=, or after the state went back): %.2f ms\n",
                n - 2000, tc1 - tc0);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    vectors();
    running();
    keep();
    std::printf("ds4_keep_test: ok\n");
#if !defined(_WIN32)
    if (argc > 1 && std::string(argv[1]) == "bench") bench(argc > 3 ? argv[2] : nullptr, argc > 3 ? argv[3] : nullptr);
#else
    (void) argc, (void) argv;
#endif
    return 0;
}
