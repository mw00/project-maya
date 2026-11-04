#include "../src/ds4/ds4_draft_placement.hpp"

#include <cstdio>
#include <cstdlib>

static void require(bool ok) {
    if (!ok) { std::fputs("draft placement check failed\n", stderr); std::exit(1); }
}

int main() {
    using strata::ds4::draft_gpu_indices;
    std::vector<int> out;
    std::string err;
    // IDs intentionally differ from both vector indices and TP member numbers.
    const std::vector<int> selected{7, 2, 11};
    require(draft_gpu_indices("11", 3, selected, out, err));
    require(out == std::vector<int>({2, 2, 2}));
    require(draft_gpu_indices("7,11,2", 3, selected, out, err));
    require(out == std::vector<int>({0, 2, 1}));
    require(draft_gpu_indices("2,2,7", 3, selected, out, err));
    require(out == std::vector<int>({1, 1, 0}));
    for (const char* value : {"", "0", "-1", "+2", " 2", "2 ", "2,", ",2", "2,,7", "2,7",
                              "2,7,11,2", "2147483648", "999999999999999999999999", "2x", "2,3,11"}) {
        out = {99};
        require(!draft_gpu_indices(value, 3, selected, out, err));
        require(out.empty() && !err.empty());
    }
    require(!draft_gpu_indices("2", 0, selected, out, err));
    require(!draft_gpu_indices("2", 3, {}, out, err));
    require(draft_gpu_indices("2", 1, selected, out, err));
    require(out == std::vector<int>({1}));
    std::puts("draft placement ID mapping and invalid-input checks passed");
}
