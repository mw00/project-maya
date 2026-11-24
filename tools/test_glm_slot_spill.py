#!/usr/bin/env python3
"""Host-only regression for the GLM two-tier slot block.

The C++ fixture is generated from the exact Slot/lambda block in generate.cpp.
Only the model save/load surface is replaced with deterministic byte buffers.
No model, GPU, network, server, or WSL is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "src/program/generate.cpp"
START = "    static constexpr uint64_t kNoImages"
END = "    // a request's DFlash window"


def extract(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    a, b = text.find(START), text.find(END)
    if a < 0 or b < 0 or b <= a:
        raise RuntimeError("could not locate the Slot/lambda block in generate.cpp")
    return text[a:b]


def fixture(block: str, cli_tail: str) -> str:
    # This deliberately supplies only the surrounding function locals and fake
    # byte serialization; the policy lambdas themselves remain source-extracted.
    return r'''#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#define MAYA_VERSION "host-fixture"
static void set_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  if (*value) setenv(name, value, 1); else unsetenv(name);
#endif
}
namespace maya { namespace core {
inline bool is_system_prefix(const std::vector<int32_t>&, int) { return false; }
namespace glmpc { template<class T> bool engine_on(const T&) { return false; } }
}}
struct Options { std::string glm_pack; int max_context = 4096; bool serve = false; bool ablate_default = false;
  std::vector<int32_t> tokens; int max_new = 1; };
struct FakeModel {
  bool fast() const { return true; } bool has_mtp() const { return false; }
  bool ablating() const { return false; } bool ablate_loaded() const { return false; }
  struct G { int n_embd = 1; }; G geometry() const { return {}; }
  uint64_t slot_save(std::vector<uint8_t>& out, std::string&) { out.assign(8, 0x5a); return out.size(); }
  uint64_t slot_save(const std::string& p, std::string&) { std::ofstream f(p, std::ios::binary); const std::string b(8, 'Z'); f.write(b.data(), 8); return f ? 8 : 0; }
  bool slot_load(const std::vector<uint8_t>& b, int64_t, std::string&) { return b == std::vector<uint8_t>(8, 0x5a); }
  bool slot_load(const std::string& p, int64_t, std::string&) { std::ifstream f(p, std::ios::binary); std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {}); return b == std::vector<uint8_t>(8, 0x5a); }
};
static void fail(const char* m) { std::fprintf(stderr, "FAIL: %s\n", m); std::exit(2); }
int main(int argc, char** argv) {
  if (argc != 3) return 64; const std::string which = argv[1]; const std::filesystem::path root = argv[2];
  std::error_code clean_ec; if (which != "reindex" && which != "mixed-reindex" && which != "cli-reindex") std::filesystem::remove_all(root, clean_ec); std::filesystem::create_directories(root / "pack");
  for (const char* n: {"index.txt", "dense.bin", "native_experts.txt"}) std::ofstream(root / "pack" / n).put('x');
  const bool off = which == "off-disk" || which == "off-ram" || which == "off-unset";
  set_env("MAYA_GLM_NO_REUSE", ""); set_env("MAYA_GLM_SLOT_KEEP", "0"); set_env("MAYA_GLM_SLOT_DAYS", "7");
  set_env("MAYA_GLM_SLOTS", off ? "2" : "8"); set_env("MAYA_GLM_SLOT_MIN", "1"); set_env("MAYA_GLM_SLOT_SPILL", which == "off-unset" ? "" : off ? "0" : "1");
  set_env("MAYA_GLM_SLOT_RAM_GB", which == "off-disk" || which == "off-unset" || which == "failed-put" ? "0" : which == "off-ram" ? "0.000001" : "0.00000001");
  const bool tight = which == "lru" || which == "budget" || which == "ram-pressure" || which == "reindex" || which == "mixed-reindex";
  set_env("MAYA_GLM_SLOT_GB", which == "zero-budget" || which == "failed-put" ? "0" : tight ? "0.00000001" : "0.00000003");
  set_env("MAYA_GLM_SLOT_DIR", (root / "slots").string().c_str());
  if (which == "default-budgets") {
    set_env("MAYA_GLM_SLOT_RAM_GB", ""); set_env("MAYA_GLM_SLOT_GB", "");
  }
  Options o; o.glm_pack = (root / "pack").string(); FakeModel model;
  std::vector<int32_t> snap_tokens; uint64_t snap_img_hash = 1469598103934665603ull;
  std::vector<int> cur_img_pos; std::vector<float> cur_img_rows;
  const auto img_hash = [&](size_t) { return uint64_t{1469598103934665603ull}; };
  const auto abl_req = [&] { return false; };
  {
''' + block + r'''
  slot_clock = 2; // hand-installed slots use 1/2; a fixture hit is strictly newer.
  auto ram = [] { return std::make_shared<std::vector<uint8_t>>(8, 0x5a); };
  if (which == "lru") {
    // Disk H was recently taken; cold RAM C spills. The extracted trim must evict C, not H.
    const auto hot = (root / "slots" / "hot.bin").string(); std::ofstream(hot, std::ios::binary).write("12345678", 8);
    slots.push_back({{1}, hot, 8, 20}); slots.push_back({{2}, "", 8, 1}); slots.back().data = ram();
    if (!slot_write_file(1)) fail("cold RAM state was not handled");
    if (slots.size()!=1 || slots[0].tokens != std::vector<int32_t>{1}) fail("disk LRU evicted recently-hit state");
  } else if (which == "budget" || which == "zero-budget") {
    if (which == "budget") {
      const auto hot = (root / "slots" / "hot.bin").string(); std::ofstream(hot, std::ios::binary).write("ZZZZZZZZ", 8);
      slots.push_back({{1}, hot, 8, 20});
    }
    const uint64_t bytes = which == "budget" ? 11 : 8;
    slots.push_back({{3}, "", bytes, 1}); slots.back().data = std::make_shared<std::vector<uint8_t>>(bytes, 0x5a);
    const size_t n = slots.size();
    if (slot_write_file(n - 1)) fail("oversized/zero-budget state was written");
    if (slots.size() != n || !slots.back().data) fail("rejected spill changed cache state");
    if (which == "budget" && !std::filesystem::exists(root / "slots" / "hot.bin")) fail("oversized state evicted existing file");
  } else if (which == "default-budgets") {
    if (!slot_spill || slot_ram_budget != (uint64_t{8} << 30) || slot_disk_budget != (uint64_t{16} << 30))
      fail("opt-in fallback budgets changed");
    snap_tokens = {1}; slot_put("");
    if (slots.size() != 1 || !slots[0].data || !snap_in_slot) fail("default budgets did not retain RAM state");
  } else if (which == "ram-pressure" || which == "off-disk" || which == "off-ram" || which == "off-unset") {
    for (int id = 1; id <= 3; ++id) {
      snap_tokens = {id}; snap_in_slot = false; slot_put("");
      if (which == "ram-pressure" && id == 2 && slot_take({1,9}) != 1) fail("pressure fixture could not touch disk slot");
    }
    if (slots.size() != 2 || !snap_in_slot) fail("pressure/legacy cap did not retain expected snapshot");
    if (which == "ram-pressure") {
      if (slots[0].tokens != std::vector<int32_t>{1} || slots[0].data || slots[1].tokens != std::vector<int32_t>{3} || !slots[1].data)
        fail("RAM pressure did not retain hot disk slot and newest RAM slot");
    } else {
      if (slots[0].tokens != std::vector<int32_t>{2} || slots[1].tokens != std::vector<int32_t>{3}) fail("legacy count cap changed");
      for (const auto& s : slots) if ((bool)s.data != (which == "off-ram")) fail("legacy storage mode changed");
    }
  } else if (which == "failed-put") {
    snap_tokens = {3}; slot_put("");
    if (snap_in_slot || !slots.empty()) fail("discarded snapshot was marked retained");
  } else if (which == "flush") {
    // Deliberately no explicit flush: the source's function-exit guard must persist this state.
    // C is older and H is explicitly hit later. Files are deliberately flushed in
    // the opposite order, so restart must recover last-use time rather than write order.
    slots.push_back({{4,5}, "", 8, 1}); slots.back().data = ram();
    slots.push_back({{7,8}, "", 8, 2}); slots.back().data = ram();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (slot_take({4,5,9}) != 2) fail("could not mark hot RAM slot as used");
  } else if (which == "reindex") {
    std::string load_error;
    if (slots.size() != 1 || slots[0].tokens != std::vector<int32_t>{4,5} || slots[0].data ||
        !model.slot_load(slots[0].path, 2, load_error))
      fail("restart did not reindex/load the most recently used bytes");
  } else if (which == "mixed-flush") {
    // H is already on disk and hit; C stays older in RAM until shutdown writes it later.
    slots.push_back({{6,6}, "", 8, 1}); slots.back().data = ram();
    slots.push_back({{8,8}, "", 8, 2}); slots.back().data = ram();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (!slot_write_file(1) || slot_take({8,8,9}) != 2) fail("could not create/touch hot disk slot");
  } else if (which == "mixed-reindex") {
    std::string load_error;
    if (slots.size() != 1 || slots[0].tokens != std::vector<int32_t>{8,8} || slots[0].data ||
        !model.slot_load(slots[0].path, 2, load_error))
      fail("restart did not retain hot disk slot over later-flushed RAM slot");
  } else if (which == "cli") {
    const auto run_request = [&](const auto&, int, int) { snap_tokens = {9,9}; snap_in_slot = false; return 0; };
    const int sp = 0;
''' + cli_tail + r'''
  } else if (which == "cli-reindex") {
    if (slots.size() != 1 || slot_take({9,9,10}) != 2) fail("CLI success did not persist its active snapshot");
  } else if (which == "write-failure") {
    const auto bad = root / "not-a-directory"; std::ofstream(bad).put('x'); slot_dir = bad;
    slots.push_back({{6}, "", 8, 1}); slots.back().data = ram();
    if (slot_write_file(0) || !slots[0].data) fail("write failure did not preserve RAM state");
  } else if (which == "miss") {
    if (slot_take({9}) != 0) fail("no-candidate miss loaded a slot");
    slots.push_back({{1,2}, "", 8, 1}); slots.back().data=ram();
    if (slot_take({1,2}) != 0 || slot_take({1}) != 0 || slot_take({7,2,3}) != 0 || slot_take({1,8,3}) != 0) fail("miss predicate accepted an invalid prefix");
    slots[0].img = 7; if (slot_take({1,2,3}) != 0) fail("image mismatch loaded a slot");
  } else return 64;
  }
  if (which == "flush") {
    size_t bins = 0; for (const auto& e : std::filesystem::directory_iterator(root / "slots")) if (e.path().extension()==".bin") ++bins;
    if (bins != 2) fail("normal function exit did not flush RAM slots");
  }
  if (which == "off-ram" && std::filesystem::exists(root / "slots")) fail("legacy RAM-only mode wrote a slot directory");
  std::puts("PASS"); return 0;
}
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--receipt", type=Path, help="write compile output and every scenario result as JSON")
    ap.add_argument("--cxx", default=os.environ.get("CXX", "cl" if os.name == "nt" else "c++"),
                    help="C++20 compiler executable; on Windows run in an MSVC developer shell")
    args = ap.parse_args()
    source = args.source.resolve()
    source_text = source.read_text(encoding="utf-8")
    block = extract(source)
    cli_start = source_text.index("        const int rc = run_request(o.tokens, o.max_new, sp);")
    cli_end = source_text.index("        return rc;", cli_start) + len("        return rc;")
    cli_tail = source_text[cli_start:cli_end]
    receipt = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "Extracted actual host slot logic; fake model serialization; no GPU/model",
               "platform": sys.platform, "compiler": args.cxx,
               "cases": []}
    def save_receipt():
        if args.receipt:
            args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="glm-slot-spill-") as td:
        td = Path(td); cpp, exe = td / "fixture.cpp", td / ("fixture.exe" if os.name == "nt" else "fixture")
        cpp.write_text(fixture(block, cli_tail), encoding="utf-8")
        cxx = shutil.which(args.cxx)
        if not cxx:
            ap.error(f"C++ compiler not found: {args.cxx}; set --cxx or CXX")
        if Path(cxx).stem.lower() in ("cl", "clang-cl"):
            cmd = [cxx, "/nologo", "/std:c++20", "/EHsc", "/W3", f"/Fo{td / 'fixture.obj'}", str(cpp), f"/Fe:{exe}"]
        else:
            cmd = [cxx, "-std=c++20", "-O0", "-Wall", "-Wextra", str(cpp), "-o", str(exe)]
        build = subprocess.run(cmd, text=True, capture_output=True, cwd=td, timeout=60)
        receipt["compile"] = {"exit_code": build.returncode, "stdout": build.stdout, "stderr": build.stderr}
        print(build.stdout, end=""); print(build.stderr, end="", file=sys.stderr)
        if build.returncode:
            save_receipt()
            return build.returncode
        failed = []
        for case in ("lru", "budget", "zero-budget", "ram-pressure", "failed-put", "flush", "reindex",
                     "mixed-flush", "mixed-reindex", "cli", "cli-reindex", "write-failure", "miss", "off-disk", "off-ram",
                     "off-unset", "default-budgets"):
            case_root = td / ({"reindex": "flush", "mixed-reindex": "mixed-flush", "cli-reindex": "cli"}.get(case, case))
            # The executable removes its setup root; never hand it a path outside this fresh temporary tree.
            if os.path.commonpath((str(td.resolve()), str(case_root.resolve()))) != str(td.resolve()):
                raise RuntimeError(f"unsafe fixture root: {case_root}")
            p = subprocess.run([str(exe), case, str(case_root)], text=True, capture_output=True, timeout=10)
            print(f"[{case}] rc={p.returncode}\n{p.stdout}{p.stderr}", end="")
            error = ""
            if case == "miss":
                reasons = [line.removeprefix("glm slots: ") for line in p.stderr.splitlines()
                           if line.startswith("glm slots: miss:")]
                expected = ["miss: no candidate", "miss: prompt is not longer than the saved token prefix",
                            "miss: prompt is not longer than the saved token prefix", "miss: diverged at token 0",
                            "miss: diverged at token 1", "miss: token prefix matches but images differ"]
                if reasons != expected:
                    error = f"wrong runtime miss reasons: {reasons!r}"
                    print(error, file=sys.stderr)
            receipt["cases"].append({"name": case, "exit_code": p.returncode, "stdout": p.stdout,
                                     "stderr": p.stderr, "assertion_error": error, "passed": not p.returncode and not error})
            if p.returncode or error: failed.append(case)
        receipt["passed"] = len(receipt["cases"]) - len(failed)
        receipt["failed"] = failed
        save_receipt()
        print(f"RESULT {receipt['passed']}/{len(receipt['cases'])}; failed={failed}")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
