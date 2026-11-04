// STRATA_HIPBLASLT_TUNING's table list (src/prefill/hipblaslt_tuning.hpp split_table_list): ':' on Linux, ';' on
// Windows, where a drive letter's ':' stays in its path. Host code only: no GPU, no hipBLASLt.
#include "hipblaslt_tuning.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::prefill::hipblaslt::split_table_list;

static int failures = 0;

static void expect(const std::string& list, bool windows, const std::vector<std::string>& want) {
    const auto got = split_table_list(list, windows);
    if (got == want) return;
    ++failures;
    std::fprintf(stderr, "FAIL (%s) \"%s\" ->", windows ? "windows" : "linux", list.c_str());
    for (const auto& one : got) std::fprintf(stderr, " [%s]", one.c_str());
    std::fprintf(stderr, "\n");
}

int main() {
    // Linux: unchanged, ':' separates and nothing else does
    expect("/a/gfx1151-100202.txt", false, {"/a/gfx1151-100202.txt"});
    expect("/a/x.txt:/b/y.txt", false, {"/a/x.txt", "/b/y.txt"});
    expect("/a/x.txt::/b/y.txt:", false, {"/a/x.txt", "/b/y.txt"});
    expect("/a/x;y.txt", false, {"/a/x;y.txt"});
    expect("rel/x.txt:z.txt", false, {"rel/x.txt", "z.txt"});
    expect("", false, {});

    // Windows: ';' (what maya.py writes), the drive prefix kept whole
    expect("C:\\Maya\\tools\\hip\\x.txt", true, {"C:\\Maya\\tools\\hip\\x.txt"});
    expect("C:\\a\\x.txt;D:\\b\\y.txt", true, {"C:\\a\\x.txt", "D:\\b\\y.txt"});
    expect("c:/a/x.txt;d:/b/y.txt;", true, {"c:/a/x.txt", "d:/b/y.txt"});
    expect("\\\\server\\share\\x.txt;C:\\y.txt", true, {"\\\\server\\share\\x.txt", "C:\\y.txt"});
    expect("rel\\x.txt;C:\\y.txt", true, {"rel\\x.txt", "C:\\y.txt"});
    // Windows: a ':'-joined list from a config written before the switch still loads every table
    expect("C:\\a\\x.txt:C:\\a\\y.txt", true, {"C:\\a\\x.txt", "C:\\a\\y.txt"});
    expect("C:\\a\\x.txt:D:/b/y.txt:rel.txt", true, {"C:\\a\\x.txt", "D:/b/y.txt", "rel.txt"});
    // a one-letter relative name followed by ':' and no slash is a separator, not a drive
    expect("a:b.txt", true, {"a", "b.txt"});
    // the same Windows list read as Linux shows the old bug: the drive letter splits off
    expect("C:\\a\\x.txt", false, {"C", "\\a\\x.txt"});

    // the engine's call (no flag) follows the OS it was built for
#if defined(_WIN32)
    const std::vector<std::string> native_want{"C:\\a\\x.txt", "D:\\b\\y.txt"};
    const std::string native_list = "C:\\a\\x.txt;D:\\b\\y.txt";
#else
    const std::vector<std::string> native_want{"/a/x;y.txt", "/b/y.txt"};
    const std::string native_list = "/a/x;y.txt:/b/y.txt";
#endif
    if (split_table_list(native_list) != native_want) {
        ++failures;
        std::fprintf(stderr, "FAIL native split of \"%s\"\n", native_list.c_str());
    }

    if (failures) {
        std::fprintf(stderr, "%d case(s) failed\n", failures);
        return 1;
    }
    std::printf("hipblaslt_table_list_test: all cases passed\n");
    return 0;
}
