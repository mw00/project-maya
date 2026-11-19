#!/usr/bin/env python3
"""Extract actual candidate Slot/lambdas with bounded host-only fault fixtures.

Reuses the existing 17-scenario fixture without editing it, including the
16GiB legacy spill default. Simulates model bytes, OS capacities, ordinary
filesystem I/O failures and small directory scan budgets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import test_glm_slot_spill as legacy

ROOT = Path(__file__).resolve().parents[1]
CASES = ("branch-policy-completion", "branch-policy-slot", "branch-policy-disabled", "branch-policy-zero", "lru", "budget", "zero-budget", "ram-pressure", "failed-put", "flush", "reindex", "mixed-flush",
         "mixed-reindex", "cli", "cli-reindex", "write-failure", "miss", "off-disk", "off-ram", "off-unset", "default-budgets",
         "branch-ready-serialize", "branch-ready-write", "branch-ready-publish", "branch-ready-seed", "branch-ready-reindex", "branch-ready-short", "branch-ready-short-disk", "branch-ready-short-reindex",
         "branch-learn", "branch-position", "branch-save-fail", "branch-direct-disk", "branch-flush", "branch-reindex",
         "branch-corrupt", "branch-supersede", "branch-low-memory", "branch-peak-disk", "branch-peak-reject", "branch-load-memory",
         "branch-disk-floor", "branch-write-fail", "branch-atomic", "branch-budget64", "branch-budget-cap", "branch-zero-budget",
         "branch-count", "branch-ordinary-low", "branch-legacy-meta", "branch-legacy-reindex",
         "branch-lru-ram", "branch-lru-disk", "branch-lru-reindex", "branch-lru-bytes", "branch-lru-tokens",
         "branch-lru-bound", "branch-lru-oversized", "branch-lru-save-fail", "branch-lru-meta-fail",
         "branch-lru-pressure", "branch-lru-staging", "branch-lru-staging-room",
         "branch-lru-staging-save-fail", "branch-lru-staging-meta-fail", "branch-lru-staging-bound",
         "branch-lru-staging-pressure", "branch-lru-staging-reindex",
         "branch-types-mixed", "branch-types-missing", "branch-types-sidecar", "branch-types-sidecar-legacy", "branch-types-unknown",
         "branch-scan-visit", "branch-scan-candidate", "branch-scan-visit-legacy", "branch-scan-candidate-legacy",
         "branch-scan-visit-direct-legacy", "branch-scan-visit-cleanup-legacy",
         "branch-cleanup-start-tmp", "branch-cleanup-start-meta", "branch-cleanup-start-bin",
         "branch-cleanup-start-iter", "branch-cleanup-start-exists",
         "branch-cleanup-rollback", "branch-cleanup-retire", "branch-cleanup-restart",
         "branch-cleanup-rollback-meta", "branch-cleanup-retire-meta", "branch-cleanup-retire-restart",
         "completion-off", "completion-length", "completion-stop", "completion-full", "completion-empty",
         "completion-position", "completion-cancel", "completion-shards", "completion-dflash",
         "completion-memory", "completion-save-fail", "completion-publish-fail", "completion-context",
         "completion-gap", "completion-single", "completion-threshold", "completion-default-min",
         "completion-image-text", "completion-text-image", "completion-image-text-gap",
         "completion-text-image-gap")
PAIRS = {"branch-ready-reindex": "branch-ready-seed", "branch-ready-short-reindex": "branch-ready-short-disk", "reindex": "flush", "mixed-reindex": "mixed-flush", "cli-reindex": "cli",
         "branch-reindex": "branch-flush", "branch-legacy-reindex": "branch-legacy-meta", "branch-lru-reindex": "branch-lru-disk",
         "branch-lru-staging-reindex": "branch-lru-staging", "branch-cleanup-restart": "branch-cleanup-rollback",
         "branch-cleanup-retire-restart": "branch-cleanup-retire-meta"}
INCLUDES = r'''
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include "strata/core/glm_prefix_admission.hpp"
#include "strata/core/glm_prefix_metadata.hpp"
#include "strata/core/glm_slot_directory_scan.hpp"
#include "strata/core/glm_cache_memory.hpp"
static uint64_t fixture_available = uint64_t{128} << 30;
static uint64_t fixture_disk_available = uint64_t{256} << 30;
static std::filesystem::path fixture_slot_dir;
static std::string fixture_remove_failure, fixture_status_failure;
static int fixture_io_errors = 0;
static bool fixture_orphan_iter_fail = false, fixture_orphan_exists_fail = false;
static std::size_t fixture_scan_visits = strata::core::SlotDirectoryScan::max_visits;
static std::size_t fixture_scan_candidates = strata::core::SlotDirectoryScan::max_candidates;
static std::filesystem::path fixture_scan_directory(const std::filesystem::path& path) {
  if (fixture_orphan_iter_fail) {
    ++fixture_io_errors;
    return path / "missing-directory";
  }
  return path;
}
static auto fixture_symlink_status(const std::filesystem::path& path, std::error_code& ec) {
  if ((fixture_orphan_exists_fail && path.extension()==".meta") ||
      (!fixture_status_failure.empty() && path.filename()==fixture_status_failure)) {
    ++fixture_io_errors; ec = std::make_error_code(std::errc::permission_denied);
    return std::filesystem::file_status(std::filesystem::file_type::none);
  }
  return std::filesystem::symlink_status(path, ec);
}
static bool fixture_remove(const std::filesystem::path& path, std::error_code& ec) {
  if (!fixture_remove_failure.empty() && path.string().ends_with(fixture_remove_failure)) {
    ++fixture_io_errors; ec = std::make_error_code(std::errc::permission_denied); return false;
  }
  return std::filesystem::remove(path, ec);
}
static auto fixture_space(const std::filesystem::path&, std::error_code& ec) {
  ec.clear(); return std::filesystem::space_info{fixture_disk_available, fixture_disk_available, fixture_disk_available};
}
'''
FAKE = r'''
struct FakeModel {
  int64_t live_pos = 0, snap_pos = 123; int context = 0, loaded_origin = -1;
  uint64_t snap_state = 987654321;
  int live_saves = 0, vector_saves = 0, file_saves = 0, snapshot_saves = 0, loads = 0;
  bool fail_live = false, shards_agree = true;
  bool fail_metadata = false, fail_file = false;
  uint64_t declared_state_bytes = 8;
  static constexpr size_t slot_live_copy_bytes = 4u << 20, slot_file_copy_bytes = 64u << 20;
  bool fast() const { return true; } bool has_mtp() const { return false; }
  bool ablating() const { return false; } bool ablate_loaded() const { return false; }
  struct G { int n_embd = 1; }; G geometry() const { return {}; }
  std::array<uint64_t, 4> state_fp() const { return {1,2,3,4}; }
  int64_t position() const { return live_pos; }
  int64_t snapshot_pos() const { return snap_pos; }
  bool snapshot_save() { ++snapshot_saves; snap_pos = live_pos; snap_state = uint64_t(context); return true; }
  uint64_t slot_flash_bytes(int64_t n) const { return n > 0 ? declared_state_bytes : 0; }
  uint64_t slot_save(std::vector<uint8_t>& out, std::string&) { ++vector_saves; out.assign(8, 0x5a); return 8; }
  uint64_t slot_save(const std::string& p, std::string&, bool = false) {
    ++file_saves; if (fail_file) return 0;
    std::ofstream f(p, std::ios::binary); const std::string b(8,'Z'); f.write(b.data(),8); f.close();
    if (fail_metadata && p.ends_with(".bin.tmp"))
      std::filesystem::create_directory(p.substr(0,p.size()-8)+".meta.tmp");
    return f?8:0;
  }
  bool slot_save_live(int64_t n, const std::function<bool(const void*,size_t)>& put, std::string&) {
    ++live_saves; const uint32_t record[2] = {uint32_t(n), uint32_t(context)};
    if (live_pos != n || !shards_agree || !put(record, sizeof record)) return false;
    if (fail_metadata) {
      // Simulate a real filesystem failure at sidecar publication without
      // changing the extracted policy or the old state files.
      for (const auto& e : std::filesystem::directory_iterator(fixture_slot_dir)) {
        const auto path = e.path().string();
        if (path.ends_with(".bin.tmp"))
          std::filesystem::create_directory(path.substr(0, path.size() - 8) + ".meta.tmp");
      }
    }
    return !fail_live;
  }
  bool slot_load(const std::vector<uint8_t>& data, int64_t n, std::string&) {
    ++loads; if (data.size()!=8) return false;
    if (data != std::vector<uint8_t>(8,0x5a)) {
      uint32_t record[2]{}; std::memcpy(record,data.data(),8);
      if (record[0] != n) return false; loaded_origin = int(record[1]);
    }
    live_pos = n; snap_pos = n; return true;
  }
  bool slot_load(const std::string& path, int64_t n, std::string& err, bool = false) {
    std::ifstream f(path,std::ios::binary); std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),{});
    return slot_load(data,n,err);
  }
};
using Glm5Model = FakeModel;
namespace strata::core { using Glm5Model = ::FakeModel; }
'''
SETUP = r'''
  const bool completion_case = which.rfind("branch-policy-", 0)==0 || which.rfind("completion-", 0)==0 || which.rfind("branch-ready-short", 0)==0;
  set_env("MAYA_GLM_LEARNED_PREFIX", which.rfind("branch-", 0)==0 || completion_case ? "1" : "");
  set_env("MAYA_GLM_COMPLETION_CACHE", completion_case && which!="completion-off" ? "1" : "");
  set_env("MAYA_GLM_COMPLETION_MIN_TOKENS", which=="completion-default-min" ? "" : which=="completion-threshold" ? "3" : "1");
  set_env("MAYA_GLM_SLOT_MEM_RESERVE_GIB", "32");
  set_env("MAYA_GLM_LEARNED_GIB", "2");
  set_env("MAYA_GLM_LEARNED_STATES", "16");
  set_env("MAYA_GLM_SLOT_ROUNDTRIP", "");
  if (which=="branch-policy-completion") set_env("MAYA_GLM_COMPLETION_MIN_TOKENS", "3");
  if (which=="branch-policy-slot") set_env("MAYA_GLM_SLOT_MIN", "2048");
  if (which=="branch-policy-disabled") set_env("MAYA_GLM_COMPLETION_CACHE", "");
  if (which=="branch-policy-zero") set_env("MAYA_GLM_COMPLETION_MIN_TOKENS", "0");
  fixture_slot_dir = root / "slots";
  if (which.rfind("branch-cleanup-", 0)==0) {
    set_env("MAYA_GLM_LEARNED_STATES", "2");
    if (which.rfind("branch-cleanup-start-", 0)==0) {
      std::filesystem::create_directories(fixture_slot_dir);
      const std::string suffix = which=="branch-cleanup-start-tmp" ? ".tmp" :
                                 which=="branch-cleanup-start-meta" ? ".meta" : ".bin";
      const std::string pending = "branch-cleanup" + suffix;
      std::ofstream(fixture_slot_dir / pending).put('x');
      fixture_orphan_iter_fail = which=="branch-cleanup-start-iter";
      fixture_orphan_exists_fail = which=="branch-cleanup-start-exists";
      if (!fixture_orphan_iter_fail && !fixture_orphan_exists_fail) fixture_remove_failure = pending;
    }
  }
  if (which.rfind("branch-", 0)==0 || completion_case) {
    set_env("MAYA_GLM_SLOT_RAM_GB", "0.001"); set_env("MAYA_GLM_SLOT_GB", "0.001");
    set_env("MAYA_GLM_SLOTS", "64");
  }
  if (which=="branch-direct-disk" || which=="branch-disk-floor" || which=="branch-write-fail" || which=="branch-zero-budget" || which=="completion-publish-fail")
    set_env("MAYA_GLM_SLOT_RAM_GB", "0");
  if (which=="branch-ready-serialize" || which=="branch-ready-write" || which=="branch-ready-publish" || which=="branch-ready-short-disk" || which=="branch-ready-short-reindex") set_env("MAYA_GLM_SLOT_RAM_GB", "0");
  if (which=="branch-ready-write") set_env("MAYA_GLM_SLOT_GB", "0.000014");
  if (which=="branch-ready-reindex") set_env("MAYA_GLM_LEARNED_STATES", "1");
  if (which=="branch-budget64") set_env("MAYA_GLM_SLOT_GB", "");
  if (which=="branch-budget-cap") set_env("MAYA_GLM_SLOT_GB", "128");
  if (which=="branch-zero-budget") set_env("MAYA_GLM_SLOT_GB", "0");
  if (which.rfind("branch-lru-", 0)==0) set_env("MAYA_GLM_LEARNED_STATES", "2");
  if (which=="branch-lru-bytes" || which=="branch-lru-oversized") {
    set_env("MAYA_GLM_LEARNED_STATES", "16");
    set_env("MAYA_GLM_LEARNED_GIB", "0.00000001490116119384765625"); // 16 bytes, two fake states
  }
  if (which=="branch-lru-tokens") set_env("MAYA_GLM_LEARNED_STATES", "16");
  if (which=="branch-lru-disk" || which=="branch-lru-reindex" || which=="branch-lru-meta-fail" ||
      which=="branch-lru-pressure" || which=="branch-lru-staging") set_env("MAYA_GLM_SLOT_RAM_GB", "0");
  if (which.rfind("branch-lru-staging", 0)==0) {
    set_env("MAYA_GLM_SLOT_RAM_GB", "0");
    set_env("MAYA_GLM_SLOT_GB", "0.0000186264514923095703125"); // 20000 bytes
  }
  if (which=="branch-lru-staging-room") set_env("MAYA_GLM_LEARNED_STATES", "16");
  if (which.rfind("branch-cleanup-", 0)==0) set_env("MAYA_GLM_SLOT_RAM_GB", "0");
  if (which.rfind("branch-types-",0)==0) {
    if (which.ends_with("-legacy")) set_env("MAYA_GLM_LEARNED_PREFIX", "");
    std::filesystem::path metadata;
    for (const auto& e : std::filesystem::directory_iterator(fixture_slot_dir))
      if (e.path().extension()==".meta") { metadata=e.path(); break; }
    if (metadata.empty()) fail("type fixture missing valid seed metadata");
    if (which=="branch-types-mixed") {
      std::filesystem::create_directory(fixture_slot_dir / "slot-directory.bin");
      std::ofstream(fixture_slot_dir / "slot-directory.bin" / "marker").put('d');
      std::filesystem::copy_file(metadata, fixture_slot_dir / "slot-directory.meta");
      std::filesystem::create_directory(fixture_slot_dir / "slot-directory-only");
      std::ofstream(fixture_slot_dir / "slot-directory-only" / "marker").put('m');
    } else if (which=="branch-types-missing") {
      std::filesystem::copy_file(metadata, fixture_slot_dir / "slot-missing.meta");
    } else if (which.rfind("branch-types-sidecar",0)==0) {
      std::ofstream(fixture_slot_dir / "slot-held.bin", std::ios::binary).write("payload!",8);
      std::filesystem::create_directory(fixture_slot_dir / "slot-held.meta");
      std::ofstream(fixture_slot_dir / "slot-held.meta" / "marker").put('s');
    } else if (which=="branch-types-unknown") {
      std::ofstream(fixture_slot_dir / "slot-unknown.bin", std::ios::binary).write("payload!",8);
      std::filesystem::copy_file(metadata, fixture_slot_dir / "slot-unknown.meta");
      std::filesystem::last_write_time(fixture_slot_dir / "slot-unknown.meta",
        std::filesystem::file_time_type::clock::now() - std::chrono::hours(1));
      fixture_status_failure="slot-unknown.bin";
    }
  }
  if (which.rfind("branch-scan-", 0)==0) {
    if (which.ends_with("-legacy")) set_env("MAYA_GLM_LEARNED_PREFIX", "");
    if (which.find("-direct-")!=std::string::npos || which.find("-cleanup-")!=std::string::npos) {
      set_env("MAYA_GLM_SLOT_SPILL", "0"); set_env("MAYA_GLM_SLOT_RAM_GB", "0");
      set_env("MAYA_GLM_SLOT_KEEP", which.find("-cleanup-")!=std::string::npos ? "0" : "1");
    }
    fixture_scan_visits = which.find("-visit")!=std::string::npos ? 2 : strata::core::SlotDirectoryScan::max_visits;
    fixture_scan_candidates = which.find("-candidate")!=std::string::npos ? 2 : strata::core::SlotDirectoryScan::max_candidates;
    std::ofstream(fixture_slot_dir / "slot-scan-pending.tmp").put('p');
    std::ofstream(fixture_slot_dir / "unrelated.txt").put('u');
  }
'''


def fixture(source: str) -> str:
    a, b = source.index(legacy.START), source.index(legacy.END)
    block = source[a:b]
    # Replace external capacity probes and inject ordinary remove I/O errors;
    # every cache policy branch remains extracted from production.
    block = block.replace("std::filesystem::remove(", "fixture_remove(")
    block = block.replace("fs::remove(", "fixture_remove(")
    block = block.replace("strata::core::SlotDirectoryScan scan(slot_dir, branch_enabled);",
                          "strata::core::SlotDirectoryScan scan(fixture_scan_directory(slot_dir), branch_enabled, "
                          "fixture_scan_visits, fixture_scan_candidates);")
    block = block.replace("fs::symlink_status(path, ec)", "fixture_symlink_status(path, ec)")
    if block.count("cache_memory::available_bytes()") != 1:
        raise RuntimeError("memory probe extraction changed")
    block = block.replace("cache_memory::available_bytes()", "fixture_available")
    block = block.replace("fs::space(slot_dir, ec)", "fixture_space(slot_dir, ec)")
    block = block.replace("std::filesystem::space(slot_dir, ec)", "fixture_space(slot_dir, ec)")
    cli_a = source.index("        const int rc = run_request(o.tokens, o.max_new, sp);")
    cli_b = source.index("        return rc;", cli_a) + len("        return rc;")
    cpp = legacy.fixture(block, source[cli_a:cli_b])
    cpp = cpp.replace('#define STRATA_VERSION "host-fixture"', '#define STRATA_VERSION "host-fixture"\n' + INCLUDES)
    fa, fb = cpp.index("struct FakeModel {"), cpp.index("static void fail(")
    cpp = cpp[:fa] + FAKE + cpp[fb:]
    ga, gb = source.index("template <typename F>\nstatic bool guarded"), source.index("// Whether the CUDA context itself is broken")
    cpp = cpp.replace("int main(int argc, char** argv) {", source[ga:gb] + "\nint main(int argc, char** argv) {")
    cpp = cpp.replace('which != "cli-reindex")', 'which != "cli-reindex" && which != "branch-reindex" && which != "branch-legacy-reindex" && which != "branch-lru-reindex" && which != "branch-ready-reindex" && which != "branch-ready-short-reindex")')
    cpp = cpp.replace('which != "cli-reindex" &&', 'which.rfind("branch-policy-", 0) != 0 && which != "cli-reindex" &&')
    cpp = cpp.replace('which != "cli-reindex" &&', 'which != "branch-lru-staging-reindex" && which != "cli-reindex" &&')
    cpp = cpp.replace('which != "cli-reindex" &&', 'which != "branch-cleanup-restart" && which != "cli-reindex" &&')
    cpp = cpp.replace('which != "cli-reindex" &&', 'which != "branch-cleanup-retire-restart" && which != "cli-reindex" &&')
    cpp = cpp.replace('which != "cli-reindex" &&', 'which.rfind("branch-scan-", 0) != 0 && which != "cli-reindex" &&')
    cpp = cpp.replace('which != "cli-reindex" &&', 'which.rfind("branch-types-", 0) != 0 && which != "cli-reindex" &&')
    cpp = cpp.replace('std::ofstream(root / "pack" / n).put(\'x\');', 'if (!std::filesystem::exists(root / "pack" / n)) std::ofstream(root / "pack" / n).put(\'x\');')
    cpp = cpp.replace("  Options o;", SETUP + "\n  Options o;")
    extra = (ROOT / "tests/core/glm_prefix_slot_cases.inc").read_text(encoding="utf-8")
    completion = (ROOT / "tests/core/glm_completion_slot_cases.inc").read_text(encoding="utf-8")
    if "const auto completion_save =" not in block:
        raise RuntimeError("completion-save extraction changed")
    da = source.index("    struct DeferredPause {")
    db = source.index("    std::function<void()> gap_reset;", da)
    fa = source.index("    const auto flush_pauses =")
    fb = source.index("    // a request's end:", fa)
    ra = source.index("        bool deferred = false;", fb)
    rb = source.index("        if (g.produced > 0)", ra)
    ba = source.index("        ctxs[(size_t) g.ctx].busy = deferred;", rb)
    bb = source.index("    const auto gen_sweep =", ba)
    single_start = source.index(
        '        if (pause_req.load() && std::strcmp(finish, "cancel")'
    )
    single = source[single_start:]
    single = single[:single.index("        // this machine's expert usage")]
    image_capture = "        g->has_images = !cur_img_pos.empty();"
    image_capture = image_capture if image_capture in source else ""
    routes = (
        "auto capture_request_images = [&](Gen& request) { auto* g = &request;\n"
        + image_capture + "\n};\n"
        + source[da:db] + source[fa:fb]
        + "auto finish_route = [&](Gen& g) {\n" + source[ra:rb]
        + source[ba:bb]
        + "auto single_route = [&](const auto& prompt, const auto& emitted, "
        + "const char* finish) {\n"
        + "const bool request_has_images = !cur_img_pos.empty();\n"
        + single + "};\n"
    )
    completion = completion.replace("// EXTRACTED_COMPLETION_ROUTES", routes)
    cpp = cpp.replace("  } else return 64;", "  } " + extra + completion + " else return 64;")
    return cpp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--ready", type=Path, help="bind tested production files to the developer READY manifest")
    ap.add_argument("--cxx", default=os.environ.get("CXX", "cl" if os.name == "nt" else "c++"))
    ap.add_argument("--cases", nargs="*", choices=CASES)
    args = ap.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as f: f.write('{"status":"started"}\n')
    inputs = ["src/program/generate.cpp", "include/strata/core/glm_prefix_admission.hpp",
              "include/strata/core/glm_prefix_metadata.hpp", "include/strata/core/glm_cache_memory.hpp",
              "include/strata/core/glm_slot_directory_scan.hpp",
              "tests/core/glm_prefix_slot_cases.inc",
              "tools/test_glm_slot_spill.py", "tools/test_glm_prefix_slots.py",
              "tests/core/glm_completion_slot_cases.inc"]
    hashes = lambda: {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in inputs}
    receipt = {
        "scope": (
            "actual Slot/completion-save and completion-routing blocks; mocked model "
            "serialization, context binding and OS capacities; no model/GPU/WSL"
        ),
        "command": [sys.executable, *sys.argv],
        "fixture_changes": [
            "replace memory/disk capacity probes; inject ordinary filesystem I/O errors and small scan budgets",
            "unchanged legacy default16 from PR #994",
            "position/context aware fake model; shard rejection is simulated, not CUDA validation",
            "extract single/multi completion routes and deferred queue verbatim; "
            "omit timing/output/ctl bookkeeping",
            "retain pack mtime for restart",
        ],
        "inputs": hashes(),
        "cases": [],
    }
    rc = 2
    try:
        if args.ready:
            ready = json.loads(args.ready.read_text(encoding="utf-8"))
            receipt["ready"] = ready
            for name, digest in receipt["inputs"].items():
                expected = ready["source_sha256"].get(name)
                if expected is not None and digest != expected:
                    raise RuntimeError(f"READY hash mismatch: {name}")
        source = (ROOT / "src/program/generate.cpp").read_text(encoding="utf-8")
        cxx = shutil.which(args.cxx)
        if not cxx: raise RuntimeError(f"C++ compiler not found: {args.cxx}")
        receipt["compiler"] = cxx
        with tempfile.TemporaryDirectory(prefix="maya-prefix-slots-") as temp:
            td = Path(temp).resolve(); cpp = td / "fixture.cpp"; exe = td / ("fixture.exe" if os.name == "nt" else "fixture")
            cpp.write_text(fixture(source), encoding="utf-8")
            receipt["fixture_sha256"] = hashlib.sha256(cpp.read_bytes()).hexdigest()
            if Path(cxx).stem.lower() in ("cl", "clang-cl"):
                cmd = [cxx, "/nologo", "/std:c++20", "/EHsc", "/W3", "/D_CRT_SECURE_NO_WARNINGS", f"/I{ROOT / 'include'}", f"/Fo{td / 'fixture.obj'}", str(cpp), f"/Fe:{exe}"]
            else:
                cmd = [cxx, "-std=c++20", "-O0", "-pthread", "-I", str(ROOT / "include"), str(cpp), "-o", str(exe)]
            built = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=60)
            receipt["compile"] = {"exit_code": built.returncode, "stdout": built.stdout, "stderr": built.stderr}
            print(built.stdout, end=""); print(built.stderr, end="")
            rc = built.returncode
            if rc == 0:
                for case in args.cases or CASES:
                    case_root = (td / PAIRS.get(case, case)).resolve()
                    if not case_root.is_relative_to(td): raise RuntimeError("case path escaped temporary directory")
                    if case.startswith(("branch-policy-", "branch-scan-", "branch-types-")):
                        seed_case = "branch-ready-short-disk" if case.startswith("branch-policy-") else "branch-legacy-meta"
                        seeded = subprocess.run([str(exe), seed_case, str(case_root)], cwd=td,
                                                capture_output=True, text=True, timeout=15)
                        receipt[case + "-seed"] = {"exit_code": seeded.returncode, "stdout": seeded.stdout,
                                                   "stderr": seeded.stderr}
                        if seeded.returncode:
                            raise RuntimeError(f"fixture seed failed: {case}")
                    ran = subprocess.run([str(exe), case, str(case_root)], cwd=td, capture_output=True, text=True, timeout=15)
                    assertion_error = ""
                    if case == "miss":
                        reasons = [line.removeprefix("glm slots: ") for line in ran.stderr.splitlines()
                                   if line.startswith("glm slots: miss:")]
                        expected = ["miss: no candidate", "miss: prompt is not longer than the saved token prefix",
                                    "miss: prompt is not longer than the saved token prefix", "miss: diverged at token 0",
                                    "miss: diverged at token 1", "miss: token prefix matches but images differ"]
                        if reasons != expected:
                            assertion_error = f"wrong runtime miss reasons: {reasons!r}"
                    if case.startswith(("branch-cleanup-", "branch-scan-")) and not case.endswith("restart"):
                        if "cache writes disabled" not in ran.stderr:
                            assertion_error = "cache failure lacks explicit fail-closed diagnostic"
                    receipt["cases"].append({"name": case, "exit_code": ran.returncode, "stdout": ran.stdout,
                                             "stderr": ran.stderr, "assertion_error": assertion_error})
                    print(f"{case}: rc={ran.returncode}")
                    if ran.returncode or assertion_error: print(ran.stdout + ran.stderr + assertion_error); rc = 1
            receipt["status"] = "passed" if rc == 0 else "failed"
            if rc:
                evidence = args.receipt.with_suffix(".fixture.cpp")
                evidence.write_text(cpp.read_text(encoding="utf-8"), encoding="utf-8")
                receipt["fixture_evidence"] = str(evidence)
        after = hashes(); receipt["inputs_unchanged"] = after == receipt["inputs"]
        if not receipt["inputs_unchanged"]:
            receipt["status"] = "invalid_source_changed"; receipt["inputs_after"] = after; rc = 2
    except Exception as exc:
        rc = 2
        receipt["status"] = "error"; receipt["exception"] = f"{type(exc).__name__}: {exc}"; print(receipt["exception"])
    finally:
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
