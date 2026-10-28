// Literal policy cases check the contract; fresh processes check cached environment reads and one startup log.
#include "../src/ds4/ds4_expert_prefetch.hpp"

#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {
struct PolicyCase {
    const char* samples;
    const char* legacy;
    int expected;  // -1 means the effective setting is invalid.
    const char* source;
};

bool check_policy(const PolicyCase& test) {
    try {
        const auto policy = strata::ds4::parse_expert_prefetch_policy(test.samples, test.legacy);
        if (policy.samples == test.expected && std::strcmp(policy.source, test.source) == 0) return true;
    } catch (const std::invalid_argument&) {
        if (test.expected == -1) return true;
    }
    std::fprintf(stderr, "ds4 prefetch policy test: samples=%s legacy=%s expected=%d\n",
                 test.samples == nullptr ? "unset" : test.samples,
                 test.legacy == nullptr ? "unset" : test.legacy, test.expected);
    return false;
}

void set_policy_variable(const char* name, const char* value) {
    if (value != nullptr) setenv(name, value, 1);
    else unsetenv(name);
}

bool check_cached_process(const char* executable, const PolicyCase& test) {
    int output[2];
    if (pipe(output) != 0) return false;
    const pid_t child = fork();
    if (child == 0) {
        close(output[0]);
        if (dup2(output[1], STDERR_FILENO) < 0) _exit(125);
        close(output[1]);
        set_policy_variable("DS4_PF_SAMPLES", test.samples);
        set_policy_variable("DS4_PF_ALL", test.legacy);
        char expected[16];
        std::snprintf(expected, sizeof expected, "%d", test.expected);
        execl(executable, executable, "--probe", expected, nullptr);
        _exit(125);
    }
    close(output[1]);
    std::string log;
    char buffer[1024];
    ssize_t bytes;
    while ((bytes = read(output[0], buffer, sizeof buffer)) > 0) log.append(buffer, static_cast<size_t>(bytes));
    close(output[0]);
    if (child < 0) return false;
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status)) return false;
    const std::string prefix = "ds4: expert prefetch samples=";
    const size_t first = log.find(prefix);
    const bool ok = WEXITSTATUS(status) == 0 && first != std::string::npos &&
                    log.find(prefix, first + 1) == std::string::npos;
    if (!ok) std::fprintf(stderr, "ds4 prefetch policy test: cached process exit=%d log=%s\n",
                          WEXITSTATUS(status), log.c_str());
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--probe") == 0) {
        try {
            const auto& first = strata::ds4::expert_prefetch_policy();
            const int samples = first.samples;
            if (samples != std::atoi(argv[2])) return 3;
            setenv("DS4_PF_SAMPLES", "invalid-after-first-use", 1);
            setenv("DS4_PF_ALL", "0", 1);
            const auto& again = strata::ds4::expert_prefetch_policy();
            return &first == &again && again.samples == samples ? 0 : 3;
        } catch (const std::invalid_argument& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 2;
        }
    }
#if defined(_WIN32)
    constexpr int default_samples = 8;
#else
    constexpr int default_samples = 0;
#endif
    const PolicyCase cases[] = {
        {nullptr, nullptr, default_samples, "default"},
        {nullptr, "0", default_samples, "default"},
        {nullptr, "garbage", default_samples, "default"},
        {nullptr, "1", 0, "DS4_PF_ALL"},
        {"8", "0", 8, "DS4_PF_SAMPLES"},
        {"4", "garbage", 4, "DS4_PF_SAMPLES"},
        {"4", "1", 0, "DS4_PF_ALL"},
        {"8", "-1", 0, "DS4_PF_ALL"},
        {"invalid", "1", 0, "DS4_PF_ALL"},
        {"0", nullptr, 0, "DS4_PF_SAMPLES"},
        {"1", nullptr, 1, "DS4_PF_SAMPLES"},
        {"2", nullptr, 2, "DS4_PF_SAMPLES"},
        {"4", nullptr, 4, "DS4_PF_SAMPLES"},
        {"8", nullptr, 8, "DS4_PF_SAMPLES"},
        {"16", nullptr, 16, "DS4_PF_SAMPLES"},
        {"0004", nullptr, 4, "DS4_PF_SAMPLES"},
        {"", nullptr, -1, ""},
        {"17", nullptr, -1, ""},
        {"-1", nullptr, -1, ""},
        {"+4", nullptr, -1, ""},
        {" 4", nullptr, -1, ""},
        {"4 ", nullptr, -1, ""},
        {"4x", nullptr, -1, ""},
        {"4.0", nullptr, -1, ""},
        {"99999999999999999999", nullptr, -1, ""},
    };
    bool ok = true;
    for (const auto& test : cases) ok = check_policy(test) && ok;
    for (const auto& test : {cases[0], cases[6], cases[8], cases[12]})
        ok = check_cached_process(argv[0], test) && ok;
    if (ok) std::puts("ds4_prefetch_policy_test: parser and cached-process checks passed");
    return ok ? 0 : 1;
}
