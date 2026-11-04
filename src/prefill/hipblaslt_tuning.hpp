#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace strata::prefill::hipblaslt {

enum class InputType : uint8_t { f16, bf16 };

struct TuningRow {
    InputType type = InputType::bf16;
    int n = 0;
    int k = 0;
    int ldy = 0;
    int t_bucket = 0;
    int solution_id = -1;
};

class TuningTable {
public:
    bool load(const std::string& path, const std::string& expected_arch, int expected_version, std::string& err) {
        std::ifstream input(path);
        if (!input) {
            err = "cannot open tuning file";
            return false;
        }

        bool saw_header = false;
        std::string line;
        size_t line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') continue;

            std::istringstream row(line.substr(first));
            if (!saw_header) {
                std::string magic;
                std::string arch;
                int version = 0;
                std::string extra;
                if (!(row >> magic >> arch >> version) || (row >> extra) || magic != "STRATA_HIPBLASLT_TUNING_V1") {
                    err = "invalid tuning header at line " + std::to_string(line_number);
                    return false;
                }
                if (arch != expected_arch) {
                    err = "tuning architecture mismatch: file=" + arch + " runtime=" + expected_arch;
                    return false;
                }
                if (version != expected_version) {
                    err = "hipBLASLt version mismatch: file=" + std::to_string(version) +
                          " runtime=" + std::to_string(expected_version);
                    return false;
                }
                arch_ = std::move(arch);
                version_ = version;
                saw_header = true;
                continue;
            }

            std::string type_name;
            int n = 0, k = 0, ldy = 0, t_bucket = 0, solution_id = -1;
            std::string extra;
            if (!(row >> type_name >> n >> k >> ldy >> t_bucket >> solution_id) || (row >> extra) ||
                (type_name != "f16" && type_name != "bf16") || n <= 0 || k <= 0 || ldy < n || t_bucket <= 0 ||
                solution_id < 0) {
                err = "invalid tuning row at line " + std::to_string(line_number);
                return false;
            }
            TuningRow parsed;
            parsed.type = type_name == "f16" ? InputType::f16 : InputType::bf16;
            parsed.n = n;
            parsed.k = k;
            parsed.ldy = ldy;
            parsed.t_bucket = t_bucket;
            parsed.solution_id = solution_id;
            for (const auto& existing : rows_) {
                if (existing.type == parsed.type && existing.n == parsed.n && existing.k == parsed.k &&
                    existing.ldy == parsed.ldy && existing.t_bucket == parsed.t_bucket) {
                    err = "duplicate tuning key at line " + std::to_string(line_number);
                    return false;
                }
            }
            rows_.push_back(parsed);
        }
        if (!saw_header) {
            err = "tuning file has no header";
            return false;
        }
        return true;
    }

    const TuningRow* closest(InputType type, int n, int k, int ldy, int t) const {
        const TuningRow* best = nullptr;
        int64_t best_distance = std::numeric_limits<int64_t>::max();
        for (const auto& row : rows_) {
            if (row.type != type || row.n != n || row.k != k || row.ldy != ldy) continue;
            const int64_t distance = row.t_bucket >= t ? int64_t(row.t_bucket) - t : int64_t(t) - row.t_bucket;
            if (distance < best_distance ||
                (distance == best_distance && (!best || row.t_bucket < best->t_bucket))) {
                best = &row;
                best_distance = distance;
            }
        }
        return best;
    }

    const std::vector<TuningRow>& rows() const { return rows_; }
    const std::string& arch() const { return arch_; }
    int version() const { return version_; }

private:
    std::string arch_;
    int version_ = 0;
    std::vector<TuningRow> rows_;
};

// The tuning-table list (STRATA_HIPBLASLT_TUNING): paths joined by the OS's path-list separator, ':' on Linux and
// ';' on Windows (maya.py joins them with os.pathsep). On Windows a ':' still separates too, for configs written
// before the switch, except the one of a drive prefix ("C:\..." or "C:/..."), which belongs to the path.
// Empty entries are skipped.
inline std::vector<std::string> split_table_list(const std::string& list, bool windows) {
    std::vector<std::string> out;
    std::string one;
    for (size_t i = 0; i < list.size(); ++i) {
        const char c = list[i];
        const bool drive = windows && c == ':' && one.size() == 1 &&
                           ((one[0] >= 'A' && one[0] <= 'Z') || (one[0] >= 'a' && one[0] <= 'z')) &&
                           i + 1 < list.size() && (list[i + 1] == '\\' || list[i + 1] == '/');
        if ((c == ':' && !drive) || (windows && c == ';')) {
            if (!one.empty()) out.push_back(one);
            one.clear();
        } else {
            one += c;
        }
    }
    if (!one.empty()) out.push_back(one);
    return out;
}

inline std::vector<std::string> split_table_list(const std::string& list) {
#if defined(_WIN32)
    return split_table_list(list, true);
#else
    return split_table_list(list, false);
#endif
}

}  // namespace strata::prefill::hipblaslt
