// tests/glm_dsa/glm_pool_classes_test.cpp - glm-dsa B3.7 (CPU): the slow-path expert pool's size classes.
//
//   * MAYA_GLM_POOL_CLASSES parsing (unset, 0, garbage, negative -> no cap);
//   * the merge of 9 classes (the PACK9 shape: 10 MoE layers, the 10th in class 0, plus the NextN layer) to 8, 2 and
//     1 class: layers keep their class below the cap, the folded ones land in the last kept class, its stride is the
//     largest folded stride, the demand counters are summed, every slot still holds its layer's blob;
//   * a cap at or above the class count, or 0, changes nothing; inconsistent vectors are left alone;
//   * glm_pool_classes_last() reports the latest count.
#include "strata/core/glm_pool_classes.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using strata::core::glm_pool_class_cap;
using strata::core::glm_pool_classes_last;
using strata::core::glm_pool_merge_classes;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

struct Pool {
    std::vector<int> layer_class;
    std::vector<size_t> stride, weight;
    std::vector<int> layers;
    std::vector<size_t> blob;   // per layer: the bytes its experts need (= its original class stride)
};

// PACK9-like: layers 0, 1 dense (-1); layers 2..11 MoE in classes 0..8 then 0 again; layer 12 (NextN) in class 4
Pool pack9() {
    Pool p;
    const size_t strides[9] = {1000, 1300, 1150, 900, 1600, 1250, 1700, 1800, 1400};
    p.stride.assign(strides, strides + 9);
    p.weight.assign(9, 0);
    p.layers.assign(9, 0);
    p.layer_class = {-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 4};
    for (int c : p.layer_class) {
        p.blob.push_back(c < 0 ? 0 : strides[c]);
        if (c >= 0) {
            ++p.weight[(size_t) c];
            ++p.layers[(size_t) c];
        }
    }
    return p;
}

bool blobs_fit(const Pool& p) {
    for (size_t il = 0; il < p.layer_class.size(); ++il) {
        const int c = p.layer_class[il];
        if (c < 0) continue;
        if (c >= (int) p.stride.size() || p.stride[(size_t) c] < p.blob[il]) return false;
    }
    return true;
}

}  // namespace

int main() {
    std::printf("glm_pool_classes_test: the slow-path expert pool size classes (B3.7)\n");
    check(glm_pool_class_cap(nullptr) == 0 && glm_pool_class_cap("") == 0 && glm_pool_class_cap("0") == 0 &&
              glm_pool_class_cap("x") == 0 && glm_pool_class_cap("-3") == 0 && glm_pool_class_cap("8x") == 0,
          "MAYA_GLM_POOL_CLASSES: unset / 0 / garbage / negative mean no cap");
    check(glm_pool_class_cap("8") == 8 && glm_pool_class_cap("1") == 1, "MAYA_GLM_POOL_CLASSES=8 / 1 parse");
    check(glm_pool_classes_last() == -1, "glm_pool_classes_last() is -1 before any pool");

    {   // no cap, cap >= n: unchanged
        for (int cap : {0, 9, 12}) {
            Pool p = pack9(), q = pack9();
            const int n = glm_pool_merge_classes(p.layer_class, p.stride, p.weight, p.layers, cap);
            check(n == 9 && p.layer_class == q.layer_class && p.stride == q.stride && p.weight == q.weight &&
                      p.layers == q.layers && glm_pool_classes_last() == 9,
                  "cap " + std::to_string(cap) + ": 9 classes unchanged");
        }
    }
    for (int cap : {8, 2, 1}) {
        Pool p = pack9();
        const Pool q = pack9();
        const int n = glm_pool_merge_classes(p.layer_class, p.stride, p.weight, p.layers, cap);
        bool kept = true, folded = true;
        size_t want_stride = 0, want_w = 0;
        int want_l = 0;
        for (int c = cap - 1; c < 9; ++c) {
            want_stride = std::max(want_stride, q.stride[(size_t) c]);
            want_w += q.weight[(size_t) c];
            want_l += q.layers[(size_t) c];
        }
        for (size_t il = 0; il < q.layer_class.size(); ++il) {
            const int c0 = q.layer_class[il], c1 = p.layer_class[il];
            if (c0 < cap - 1) kept = kept && c1 == c0;
            else folded = folded && c1 == cap - 1;
        }
        for (int c = 0; c < cap - 1; ++c)
            kept = kept && p.stride[(size_t) c] == q.stride[(size_t) c] && p.weight[(size_t) c] == q.weight[(size_t) c];
        const std::string tag = "cap " + std::to_string(cap) + ": ";
        check(n == cap && p.stride.size() == (size_t) cap && p.weight.size() == (size_t) cap &&
                  p.layers.size() == (size_t) cap && glm_pool_classes_last() == cap,
              tag + "9 classes -> " + std::to_string(n));
        check(kept, tag + "classes below the cap keep their layers, stride and demand");
        check(folded, tag + "every folded layer lands in class " + std::to_string(cap - 1) + " (dense layers stay -1)");
        check(p.stride[(size_t) cap - 1] == want_stride && p.weight[(size_t) cap - 1] == want_w &&
                  p.layers[(size_t) cap - 1] == want_l,
              tag + "the merged class has the largest stride (" + std::to_string(want_stride) + ") and the summed demand");
        check(blobs_fit(p), tag + "every MoE layer's blob fits its class slot");
        int moe = 0;
        for (int l : p.layers) moe += l;
        check(moe == 11, tag + "11 MoE layer entries in total (10 trunk + NextN)");
    }
    {   // inconsistent vectors: nothing merged
        Pool p = pack9();
        p.weight.pop_back();
        const std::vector<int> lc = p.layer_class;
        const int n = glm_pool_merge_classes(p.layer_class, p.stride, p.weight, p.layers, 2);
        check(n == 9 && p.layer_class == lc && p.stride.size() == 9, "inconsistent class vectors are not merged");
    }
    std::printf("glm_pool_classes_test: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
