"""Several conversations at once on maya-ds4 (--parallel): the engine's command line, a conversation's state the whole
of it, and the mailbox that carries several conversations' windows.  No GPU and no model needed.

The state: a conversation is set aside by swapping every piece of its state - the position, the tokens, the
checkpoints, the flag slot_save set, the penalties, the pictures, and the device pointers every layer's ring and
compressor group and every source's rows live behind - with those of another's (Impl::lane_swap, extracted from
src/ds4/ds4_model.cu and compiled against small host-only structs).  A field the swap forgets is a conversation mixed
with another's: a wrong prompt, a wrong answer, a slot loaded from what is not there - and nothing says so.  The check
swaps two lanes a few times over and compares every piece of both with what it started as.

The mailbox (Impl::Mail, the same way): the slots a conversation's window lives in, and the order the service takes
them in.  Three conversations' windows of DeepSeek V4.1's 40 layers are issued ahead - what multi_issue does when it
hands three conversations to the split before taking any - published a stage at a time, and taken as published: every
request must come back with its own layer, and a later conversation's first layer before an earlier one's last.  The
shape this replaced (one ring of kRing, numbered in one order, taken in that order) is driven the same way beside it,
and must fail: 120 requests through 64 slots is 56 shared, a request meets another's layer, and the later
conversation's first layer never comes before the earlier one's last.

    python -m unittest tools.test_ds4_parallel
"""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/ds4/ds4_model.cu"
KERNELS = ROOT / "src/ds4/ds4_kernels.cuh"
EXE = Path(os.environ.get("MAYA_DS4_EXE", str(ROOT / "build/maya-ds4")))


def cut(source: str, start: str) -> str:
    """The method or struct from `start` to its closing brace."""
    i = source.index(start)
    end = source.index("\n    };", i) + len("\n    };") if start.strip().startswith("struct") else \
        source.index("\n    }", i) + len("\n    }")
    return source[i:end]


def engine(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(EXE), *args], capture_output=True, text=True, timeout=120)


@unittest.skipUnless(EXE.exists(), "the engine is not built (build/maya-ds4, or MAYA_DS4_EXE)")
class Cli(unittest.TestCase):
    """--parallel N (1-8), its refusals, and that 1 is the engine as before - said before a 180 GB model loads."""

    def test_the_bounds_are_one_to_eight(self):
        for n in ("0", "9", "-1"):
            r = engine("--parallel", n)
            self.assertEqual(r.returncode, 2, n)
            self.assertIn("one to eight conversations at once", r.stderr, n)

    def test_the_serial_path_has_no_lanes(self):
        r = engine("--parallel", "2", "--threads", "off")
        self.assertEqual(r.returncode, 2)
        self.assertIn("needs --threads on", r.stderr)

    def test_tensor_parallel_refuses_it_naming_the_follow_up(self):
        r = engine("--parallel", "2", "--threads", "on", "--tp", "2")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--tp 2", r.stderr)
        self.assertIn("#115", r.stderr)   # (the batched step a --tp user wants, as the PR says)

    def test_parallel_2_with_threads_and_no_tensor_parallel_goes_on_to_its_pack(self):
        # (past the refusals: what is missing is the pack, asked for as the usage says - not a --parallel complaint)
        r = engine("--parallel", "2", "--threads", "on", "--tp", "1")
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("--parallel 2", r.stderr)
        self.assertIn("--pack", r.stderr)

    def test_the_usage_lists_it(self):
        self.assertIn("--parallel N", engine("--help").stderr)


@unittest.skipUnless(shutil.which("c++"), "a host C++ compiler is needed")
class ServePath(unittest.TestCase):
    """--parallel 1 is the single path: READY without multi=, untagged STOP, pictures and the engine-end slot."""

    def test_parallel_1_takes_the_single_path(self):
        source = (ROOT / "src/ds4/ds4_main.cpp").read_text()
        m = re.search(r"static bool serve_multi\(int parallel\) \{ return parallel > 1; \}", source)
        self.assertIsNotNone(m, "serve's multi/single choice is not factored")
        for line in ("const bool multi = serve_multi(parallel);", "in.tagged = multi;",
                     "if (!multi) slot_put(\" as the engine ended\");"):
            self.assertIn(line, source)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "path.cpp"
            src.write_text("#include <cstdio>\n" + m.group(0) + "\n"
                           "int main() {\n"
                           "    if (serve_multi(1)) { std::puts(\"parallel 1 is multi\"); return 1; }\n"
                           "    if (!serve_multi(2)) { std::puts(\"parallel 2 is not multi\"); return 2; }\n"
                           "    std::puts(\"ok: 1 is the single path, 2 is the multi path\");\n"
                           "    return 0;\n"
                           "}\n")
            exe = Path(d) / "path"
            r = subprocess.run(["c++", "-std=c++17", "-O1", str(src), "-o", str(exe)], capture_output=True, text=True,
                               timeout=180)
            self.assertEqual(r.returncode, 0, r.stderr[:4000])
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("1 is the single path", r.stdout)


FIXTURE = r'''
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

typedef void* cudaEvent_t;              // (the lane holds its window's event: opaque here)
namespace strata::kernels { struct SamplerParams { float temperature = 0.0f; }; }
namespace strata::ds4 {
struct Image { int64_t start = 0; };

struct Layer {                          // (what lane_swap touches of one)
    uint8_t* win = nullptr;
    float *c_kv = nullptr, *c_score = nullptr;
    std::vector<Layer> mates;
};

struct Src {                            // (a source's rows on each device)
    std::vector<uint8_t*> ckv, ik;
};

struct Impl {
    static constexpr int kWin = 16, kPenCap = 4096;
    int64_t pos = 0;
    std::vector<int32_t> toks, ctoks;
    struct Check {
        int64_t pos = -1;
        std::vector<std::vector<uint8_t>> win;
        std::vector<std::vector<float>> comp;
    };
    Check ck[3];
    bool slot_held = false;
    std::vector<Image> imgs;
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    std::vector<Layer> L;
    std::vector<Src> srcs;
    struct Dev { int id = 0; };
    std::vector<Dev> devs;
    struct { int eg_layers = 0, eg_ngram = 0, eg_heads = 0, eg_hd = 0; } pk;   // (eg_row_bytes')
    size_t eg_cap = 16;
    int32_t* h_pen_all = nullptr;
    int32_t* h_pen = nullptr;
    uint8_t *h_eg = nullptr, *h_eg_l = nullptr;
    int cur_seq = 0;                    // (the lanes themselves come with the engine's own text, below)
    Layer& member(Layer& L0, int mi) { return mi == 0 ? L0 : L0.mates[(size_t) mi - 1]; }

    // ---- from src/ds4/ds4_model.cu ----
%LANE%
%LANES%
%SWAP%
%EG%
%BIND%
};
}  // namespace strata::ds4
'''

MAIN = r'''
#include <cstdio>

using namespace strata::ds4;

static bool same(const Impl::Lane& a, const Impl::Lane& b) {
    if (a.pos != b.pos || a.toks != b.toks || a.ctoks != b.ctoks || a.slot_held != b.slot_held) return false;
    if (a.pen_n != b.pen_n || a.pen_repeat != b.pen_repeat || a.pen_freq != b.pen_freq || a.pen_present != b.pen_present)
        return false;
    if (a.imgs.size() != b.imgs.size()) return false;
    for (size_t i = 0; i < a.imgs.size(); ++i)
        if (a.imgs[i].start != b.imgs[i].start) return false;
    for (int k = 0; k < 3; ++k)
        if (a.ck[k].pos != b.ck[k].pos || a.ck[k].win != b.ck[k].win || a.ck[k].comp != b.ck[k].comp) return false;
    if (a.win.size() != b.win.size() || a.c_kv.size() != b.c_kv.size() || a.c_score.size() != b.c_score.size())
        return false;
    for (size_t i = 0; i < a.win.size(); ++i)
        if (a.win[i] != b.win[i] || a.c_kv[i] != b.c_kv[i] || a.c_score[i] != b.c_score[i]) return false;
    if (a.ckv.size() != b.ckv.size() || a.ik.size() != b.ik.size()) return false;
    for (size_t i = 0; i < a.ckv.size(); ++i)
        if (a.ckv[i] != b.ckv[i] || a.ik[i] != b.ik[i]) return false;
    return true;
}

// the model's bound conversation is, in every piece of it, the one a lane's slot holds
static bool bound(const Impl& m, const Impl::Lane& s) {
    if (m.pos != s.pos || m.toks != s.toks || m.ctoks != s.ctoks || m.slot_held != s.slot_held) return false;
    if (m.pen_n != s.pen_n || m.pen_repeat != s.pen_repeat || m.pen_freq != s.pen_freq || m.pen_present != s.pen_present)
        return false;
    if (m.imgs.size() != s.imgs.size()) return false;
    for (size_t i = 0; i < m.imgs.size(); ++i)
        if (m.imgs[i].start != s.imgs[i].start) return false;
    for (int k = 0; k < 3; ++k)
        if (m.ck[k].pos != s.ck[k].pos || m.ck[k].win != s.ck[k].win || m.ck[k].comp != s.ck[k].comp) return false;
    for (size_t i = 0; i < s.win.size(); ++i) {
        if (m.L[i].win != s.win[i][0] || m.L[i].c_kv != s.c_kv[i][0] || m.L[i].c_score != s.c_score[i][0]) return false;
        for (size_t j = 0; j < m.L[i].mates.size(); ++j) {
            if (m.L[i].mates[j].win != s.win[i][j + 1]) return false;
            if (m.L[i].mates[j].c_kv != s.c_kv[i][j + 1]) return false;
            if (m.L[i].mates[j].c_score != s.c_score[i][j + 1]) return false;
        }
    }
    for (size_t i = 0; i < s.ckv.size(); ++i)
        for (size_t d = 0; d < m.srcs[i].ckv.size(); ++d)
            if (m.srcs[i].ckv[d] != s.ckv[i][d] || m.srcs[i].ik[d] != s.ik[i][d]) return false;
    return true;
}

int main() {
    // two layers, the second with a tensor parallel member; two sources, each with rows on two devices
    Impl m;
    m.L.resize(2);
    m.L[1].mates.resize(1);
    m.srcs.resize(2);
    for (Src& s : m.srcs) {
        s.ckv.assign(2, nullptr);
        s.ik.assign(2, nullptr);
    }
    m.devs.resize(2);
    // the model's own conversation (lane 0's), first
    m.pos = 100;
    m.toks = {1, 2, 3};
    m.ctoks = {4, 5, 6};
    m.slot_held = false;
    m.imgs.push_back(Image{7});
    m.pen_n = 64;
    m.pen_repeat = 1.1f;
    m.pen_freq = 0.2f;
    m.pen_present = 0.3f;
    m.ck[0].pos = 99;
    m.ck[0].win.push_back({1, 2, 3});
    m.ck[0].comp.push_back({4.0f, 5.0f});
    m.ck[1].pos = 98;
    m.ck[2].pos = 97;
    for (size_t i = 0; i < m.L.size(); ++i) {
        m.L[i].win = (uint8_t*) (0x1000 + i);
        m.L[i].c_kv = (float*) (0x2000 + i);
        m.L[i].c_score = (float*) (0x3000 + i);
        for (size_t j = 0; j < m.L[i].mates.size(); ++j) {
            m.L[i].mates[j].win = (uint8_t*) (0x4000 + i);
            m.L[i].mates[j].c_kv = (float*) (0x5000 + i);
            m.L[i].mates[j].c_score = (float*) (0x6000 + i);
        }
    }
    for (size_t i = 0; i < m.srcs.size(); ++i)
        for (size_t d = 0; d < 2; ++d) {
            m.srcs[i].ckv[d] = (uint8_t*) (0x7000 + i * 16 + d);
            m.srcs[i].ik[d] = (uint8_t*) (0x8000 + i * 16 + d);
        }
    // three conversations: the model's (0) and two set aside, each with its own everything
    m.lanes.assign(3, Impl::Lane{});
    for (Impl::Lane& a : m.lanes) {     // (the shape every lane's pointers hold: the model's own too)
        a.win.assign(m.L.size(), {});
        a.c_kv.assign(m.L.size(), {});
        a.c_score.assign(m.L.size(), {});
        for (size_t i = 0; i < m.L.size(); ++i) {
            a.win[i].assign(1 + m.L[i].mates.size(), nullptr);
            a.c_kv[i].assign(1 + m.L[i].mates.size(), nullptr);
            a.c_score[i].assign(1 + m.L[i].mates.size(), nullptr);
        }
        a.ckv.assign(m.srcs.size(), {});
        a.ik.assign(m.srcs.size(), {});
        for (size_t i = 0; i < m.srcs.size(); ++i) {
            a.ckv[i].assign(2, nullptr);
            a.ik[i].assign(2, nullptr);
        }
    }
    for (int s = 1; s < 3; ++s) {
        Impl::Lane& a = m.lanes[(size_t) s];
        a.pos = 1000 * s;
        a.toks = {(int32_t) s, (int32_t) s + 1};
        a.ctoks = {(int32_t) s + 2};
        a.slot_held = s == 2;
        // (every piece of a conversation differs between lanes: a swap that forgets one shows)
        a.pen_n = 10 * s;
        a.pen_repeat = 2.0f + (float) s;
        a.pen_freq = 0.5f * (float) s;
        a.pen_present = 0.25f * (float) s;
        for (int k = 0; k < 3; ++k) {
            a.ck[k].pos = -7 - s - 100 * k;
            a.ck[k].win.push_back({(uint8_t) (s * 16 + k)});
            a.ck[k].comp.push_back({(float) (s * 16 + k)});
        }
        a.imgs.push_back(Image{100 + s});
        for (size_t i = 0; i < m.L.size(); ++i)
            for (size_t j = 0; j < a.win[i].size(); ++j) {
                a.win[i][j] = (uint8_t*) (0x10000 * (size_t) s + 0x1000 * i + j);
                a.c_kv[i][j] = (float*) (0x10000 * (size_t) s + 0x2000 * i + j);
                a.c_score[i][j] = (float*) (0x10000 * (size_t) s + 0x3000 * i + j);
            }
        for (size_t i = 0; i < m.srcs.size(); ++i)
            for (size_t d = 0; d < 2; ++d) {
                a.ckv[i][d] = (uint8_t*) (0x10000 * (size_t) s + 0x7000 + i * 16 + d);
                a.ik[i][d] = (uint8_t*) (0x10000 * (size_t) s + 0x8000 + i * 16 + d);
            }
    }
    // the three conversations as they stand (the model's own out into lane 0's place, copied, and back: a swap, twice)
    const Impl::Lane L0 = m.lanes[0], L1 = m.lanes[1], L2 = m.lanes[2];
    Impl::Lane M;
    m.lane_swap(m.lanes[0]);
    M = m.lanes[0];
    m.lane_swap(m.lanes[0]);
    if (!bound(m, M)) return 1;
    if (!same(m.lanes[0], L0)) return 2;
    // 1 -> 2 -> 0, three times: a bind puts the model's conversation in its own lane's slot and the one bound in
    for (int round = 0; round < 3; ++round) {
        m.seq_bind(1);                                       // (0 -> 1: lane 0 takes the model's, lane 1's comes in)
        if (!bound(m, L1)) return 10 + round;
        if (!same(m.lanes[0], M)) return 20 + round;
        if (!same(m.lanes[1], L0) || !same(m.lanes[2], L2)) return 30 + round;
        m.seq_bind(2);                                       // (1 -> 2: lane 1 takes its own again)
        if (!bound(m, L2)) return 40 + round;
        if (!same(m.lanes[0], M) || !same(m.lanes[1], L1)) return 50 + round;
        if (!same(m.lanes[2], L0)) return 60 + round;
        m.seq_bind(0);                                       // (2 -> 0: the model's own, whole, back)
        if (!bound(m, M)) return 70 + round;
        if (!same(m.lanes[0], L0) || !same(m.lanes[1], L1) || !same(m.lanes[2], L2)) return 80 + round;
        // (the pointer behind every piece of it, not just the scalars: the rings, the compressor groups, the rows)
        if (m.L[0].win != (uint8_t*) 0x1000 || m.L[0].c_kv != (float*) 0x2000) return 90 + round;
        if (m.L[1].mates[0].win != (uint8_t*) 0x4001 || m.L[1].mates[0].c_score != (float*) 0x6001) return 100 + round;
        if (m.srcs[0].ckv[1] != (uint8_t*) 0x7001 || m.srcs[1].ik[0] != (uint8_t*) 0x8010) return 110 + round;
        if (m.ck[0].pos != 99 || m.ck[0].win.size() != 1 || m.ck[1].pos != 98) return 120 + round;
        if (m.imgs.size() != 1 || m.imgs[0].start != 7 || m.pen_repeat != 1.1f) return 130 + round;
        m.seq_bind(1);
        if (!bound(m, L1) || m.L[1].mates[0].win != (uint8_t*) 0x11001) return 140 + round;   // (lane 1's own)
        m.seq_bind(0);
        if (!bound(m, M)) return 150 + round;
    }
    std::printf("ok: three conversations, every field swapped whole\\n");
    return 0;
}
'''



@unittest.skipUnless(shutil.which("c++"), "a host C++ compiler is needed")
class LaneState(unittest.TestCase):
    def test_a_conversation_swaps_whole(self):
        source = SOURCE.read_text()
        body = FIXTURE
        for name, text in (("%LANE%", cut(source, "    struct Lane {")),
                           ("%LANES%", "    std::vector<Lane> lanes;\n"),
                           ("%SWAP%", cut(source, "    void lane_swap(Lane& a) {")),
                           ("%EG%", cut(source, "    size_t eg_row_bytes() const {")),
                           ("%BIND%", cut(source, "    void seq_bind(int s) {"))):
            body = body.replace(name, text)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "lane.cpp"
            src.write_text(body + MAIN)
            exe = Path(d) / "lane"
            r = subprocess.run(["c++", "-std=c++17", "-O1", str(src), "-o", str(exe)], capture_output=True, text=True,
                               timeout=180)
            self.assertEqual(r.returncode, 0, r.stderr[:4000])
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("every field swapped whole", r.stdout)


MAIL_FIXTURE = r"""
#include <cstdio>
#include <vector>

namespace strata::ds4 {
constexpr int kRing = %RING%;   // (ds4_kernels.cuh's: the one conversation's ring)
struct Req { unsigned seq = 0; };   // (a mailbox slot: the protocol reads its seq, which the device writes last)

struct Impl {
%MAIL%
};
}  // namespace strata::ds4
"""


MAIL_MAIN = r"""
#include <cstdio>
#include <vector>

using namespace strata::ds4;

static constexpr int kLanes = 3, kLayers = 40;   // three conversations, a window of DeepSeek V4.1's 40 layers each

// a planned request: the slot it lives in, and its seq (the number its device publishes into it)
struct At {
    int slot = -1;
    unsigned seq = 0;
};

struct Report {
    int served = 0, mismatch = 0, collisions = 0;   // (requests taken; taken with another request's layer; slots shared)
    int first = -1, last = -1;                      // (the take a later conversation's layer 0, an earlier one's 39 came at)
};

// ---- the engine's mailbox (Mail), driven as plan_window, a device's route_publish and serve_lane drive it
struct Engine {
    Impl::Mail mail;
    std::vector<unsigned> lane_seq;
    std::vector<Req> reqs;
    std::vector<int> layer_of;   // (the plan's: the layer a request at a slot is for)
    std::vector<int> pub_lane, pub_layer;   // (the test's: what the device published at a slot - the routing there)

    Engine() : lane_seq(kLanes, 1), reqs((size_t) kLanes * kLayers), layer_of((size_t) kLanes * kLayers, -1),
               pub_lane((size_t) kLanes * kLayers, -1), pub_layer((size_t) kLanes * kLayers, -1) {
        mail.places = kLanes;
        mail.ring = kLayers;
        mail.next.assign(kLanes, 1);
    }
    int slots() const { return mail.slots(); }
    At plan(int lane, int layer) {          // (plan_window: its request's number, its slot, its layer for the service)
        const unsigned seq = lane_seq[(size_t) lane]++;
        const int at = mail.slot(lane, seq);
        layer_of[(size_t) at] = layer;
        return At{at, seq};
    }
    void publish(const At& a, int lane, int layer) {   // (route_publish: the routing, then T, then seq)
        reqs[(size_t) a.slot].seq = a.seq;
        pub_lane[(size_t) a.slot] = lane;
        pub_layer[(size_t) a.slot] = layer;
    }
    bool take(int& at, int& layer) {        // (serve_lane: Mail::take, the layer layer_of holds for the slot)
        int place = 0;
        unsigned seq = 0;
        const auto published = [this](int slot, unsigned q) { return reqs[(size_t) slot].seq == q; };
        if (!mail.take(published, place, at, seq)) return false;
        layer = layer_of[(size_t) at];
        return true;
    }
};

// ---- the engine as it was: one ring of kRing for every conversation's requests, numbered in one order (seq_issued),
// and served in that order - serve_lane's `next`: reqs[next % kRing], next = 1, 2, ...
struct OneRing {
    std::vector<Req> reqs;
    std::vector<int> layer_of, pub_lane, pub_layer;
    unsigned issued = 1, next = 1;

    OneRing() : reqs(kRing), layer_of(kRing, -1), pub_lane(kRing, -1), pub_layer(kRing, -1) {}
    int slots() const { return kRing; }
    At plan(int lane, int layer) {
        (void) lane;   // (the old plan knew no conversations: one number for every request, one ring)
        const unsigned seq = issued++;
        const int at = (int) (seq % kRing);
        layer_of[(size_t) at] = layer;
        return At{at, seq};
    }
    void publish(const At& a, int lane, int layer) {
        reqs[(size_t) a.slot].seq = a.seq;
        pub_lane[(size_t) a.slot] = lane;
        pub_layer[(size_t) a.slot] = layer;
    }
    bool take(int& at, int& layer) {
        at = (int) (next % kRing);
        if (reqs[(size_t) at].seq != next) return false;
        layer = layer_of[(size_t) at];
        ++next;
        return true;
    }
};

// the conversations' windows issued ahead (multi_issue, three times, before any is taken), the devices publishing a
// 40-layer split's requests a stage at a time (every conversation's layer 0, then every one's layer 1, ... - so a later
// conversation's first layer is published long before an earlier one's last), and the service taking whatever is
// published after each, as often as it can
template <class Ring>
static Report drive(Ring& r) {
    Report rep;
    std::vector<At> at((size_t) (kLanes * kLayers));
    std::vector<char> used((size_t) r.slots(), 0);
    for (int lane = 0; lane < kLanes; ++lane)
        for (int layer = 0; layer < kLayers; ++layer) {
            at[(size_t) (lane * kLayers + layer)] = r.plan(lane, layer);
            if (used[(size_t) at[(size_t) (lane * kLayers + layer)].slot]++) ++rep.collisions;
        }
    for (int k = 0; k < kLanes * kLayers; ++k) {
        const int lane = k % kLanes, layer = k / kLanes;
        r.publish(at[(size_t) (lane * kLayers + layer)], lane, layer);
        int sat = -1, sl = -1;
        while (r.take(sat, sl)) {
            if (sl != r.pub_layer[(size_t) sat]) ++rep.mismatch;
            if (r.pub_lane[(size_t) sat] == 1 && r.pub_layer[(size_t) sat] == 0 && rep.first < 0) rep.first = rep.served;
            if (r.pub_lane[(size_t) sat] == 0 && r.pub_layer[(size_t) sat] == kLayers - 1 && rep.last < 0)
                rep.last = rep.served;
            ++rep.served;
        }
    }
    return rep;
}

int main() {
    // ---- the engine's mailbox: three conversations x 40 layers issued ahead, a stage at a time published, taken as
    // published - every request with its own layer, and a later conversation's first layer before an earlier one's last
    Engine e;
    const Report a = drive(e);
    std::printf("mailbox: %d served of %d, %d with another request's layer, %d slots shared, lane 1's layer 0 taken "
                "at %d, lane 0's layer %d at %d\n", a.served, kLanes * kLayers, a.mismatch, a.collisions, a.first,
                kLayers - 1, a.last);
    if (a.collisions != 0) return 1;    // (a request of a window issued ahead shares a slot with another's)
    if (a.served != kLanes * kLayers) return 2;
    if (a.mismatch != 0) return 3;      // (a request taken with another's layer: a wrong token, no crash)
    if (a.first < 0 || a.last < 0 || a.first >= a.last) return 4;   // (the order the devices published in)
    // ---- the engine as it was: the same three windows through one ring of kRing, taken in the order they were issued
    OneRing o;
    const Report b = drive(o);
    std::printf("the old ring: %d served of %d, %d with another request's layer, %d slots shared, lane 1's layer 0 "
                "taken at %d, lane 0's layer %d at %d\n", b.served, kLanes * kLayers, b.mismatch, b.collisions, b.first,
                kLayers - 1, b.last);
    // (the review's arithmetic: 120 requests of three windows through 64 slots - 56 shared; a request meets another's
    // layer; and a later conversation's first layer does not come before an earlier one's last)
    if (b.collisions == 0) return 10;
    if (b.mismatch == 0) return 11;
    if (b.first >= 0 && b.last >= 0 && b.first < b.last) return 12;
    std::printf("ok: three conversations' windows held apart, and taken in the order the devices published them\n");
    return 0;
}
"""


class LaneBuffers(unittest.TestCase):
    def test_every_writer_and_reader_takes_the_same_lane(self):
        source = SOURCE.read_text()
        lines = source.splitlines()
        # the stage writer (layer()) and the stage reader (run_dev()) are the same expression; the old bug wrote
        # sr.x_stage while the lane reader waited on x_stage_l.  A lane use must take [lane] and a lane -1 use the
        # one conversation's buffer.
        for name, count in (("x_stage_l", 2), ("x_flag_l", 2), ("hand_l", 2), ("hand_flag_l", 2)):
            uses = [l.strip() for l in lines if name in l and "lane < 0 ?" in l]
            self.assertEqual(len(uses), count, name)
            for l in uses:
                self.assertIn("(size_t) lane", l)
        stage = [l.strip() for l in lines if "lane < 0 ? sr.x_stage : sr.x_stage_l" in l]
        self.assertEqual(len(stage), 2)
        self.assertEqual(stage[0], stage[1])
        flag = [l.strip() for l in lines if "lane < 0 ? sr.x_flag : sr.x_flag_l" in l]
        self.assertEqual(len(flag), 2)
        self.assertEqual(flag[0], flag[1])


@unittest.skipUnless(shutil.which("c++"), "a host C++ compiler is needed")
class Mailbox(unittest.TestCase):
    def test_the_mailbox_holds_every_conversation_of_a_window(self):
        ring = re.search(r"constexpr int kRing = (\d+);", KERNELS.read_text()).group(1)
        body = MAIL_FIXTURE.replace("%RING%", ring).replace("%MAIL%", cut(SOURCE.read_text(), "    struct Mail {"))
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "mail.cpp"
            src.write_text(body + MAIL_MAIN)
            exe = Path(d) / "mail"
            r = subprocess.run(["c++", "-std=c++17", "-O1", str(src), "-o", str(exe)], capture_output=True, text=True,
                               timeout=180)
            self.assertEqual(r.returncode, 0, r.stderr[:4000])
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("three conversations' windows held apart", r.stdout)


LOOKAHEAD_FIXTURE = r"""
#include <cstdint>
#include <cstdio>
#include <vector>

namespace strata::ds4 {
constexpr int kRing = 64;   // (ds4_kernels.cuh's: the one conversation's ring)
struct Impl {
%MAIL%
%TAG%
};
}  // namespace strata::ds4
"""

LOOKAHEAD_MAIN = r"""
#include <cstdio>

using namespace strata::ds4;

int main() {
    // one conversation in a two-conversation mailbox, its window of 40 layers; three windows
    const int ring = 40, j = 10;
    Impl::Mail mail;
    mail.ring = ring;
    mail.places = 2;
    const auto seq = [&](int w) -> unsigned { return 1u + (unsigned) (w * ring + j); };
    const uint64_t t0 = Impl::ahead_tag(0, seq(0));
    const uint64_t t1 = Impl::ahead_tag(0, seq(1));
    const uint64_t t2 = Impl::ahead_tag(0, seq(2));
    if (t0 == t1 || t1 == t2) { std::printf("flags reused across windows\n"); return 1; }
    if (mail.slot(0, seq(0)) != mail.slot(0, seq(1))) {
        std::printf("the slot tag was not reused here\n"); return 2;
    }
    if (Impl::ahead_tag(0, seq(0)) != seq(0)) { std::printf("a tag is not the request seq\n"); return 3; }
    std::printf("ok: layer %d window 1 tag %llu, window 2 tag %llu (slot %d for both)\n", j,
                (unsigned long long) t0, (unsigned long long) t1, mail.slot(0, seq(0)));
    return 0;
}
"""


@unittest.skipUnless(shutil.which("c++"), "a host C++ compiler is needed")
class LookaheadTag(unittest.TestCase):
    def test_a_window_does_not_keep_the_previous_window_s_flags(self):
        source = SOURCE.read_text()
        tag = cut(source, "    static uint64_t ahead_tag(int place, unsigned seq) {")
        body = LOOKAHEAD_FIXTURE.replace("%MAIL%", cut(source, "    struct Mail {")).replace("%TAG%", tag)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "ahead.cpp"
            src.write_text(body + LOOKAHEAD_MAIN)
            exe = Path(d) / "ahead"
            r = subprocess.run(["c++", "-std=c++17", "-O1", str(src), "-o", str(exe)], capture_output=True, text=True,
                               timeout=180)
            self.assertEqual(r.returncode, 0, r.stderr[:4000])
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("window 1 tag", r.stdout)
            self.assertIn("window 2 tag", r.stdout)


if __name__ == "__main__":
    unittest.main()
