// include/strata/core/glm_pool_classes.hpp - glm-dsa B3.7: the slow path's device expert pool, any number of size
// classes, and an optional cap on their count.
//
// Glm5Model::load_pack (glm_model.cu, Maya-owned) builds one pool class per distinct (gate/up row bytes, down row
// bytes) pair over the MoE layers: class c's slots are size-homogeneous and evict only within their own range.  Maya
// sized the four class arrays [8]; a GGUF with more than 8 expert size classes (audit check A13) overflowed them.
// They are std::vector now (glm_model.hpp), and the classes are passed through glm_pool_merge_classes, which folds
// every class past the cap into the last kept one (its stride the largest of the folded classes), so the same weights
// can run with 9, 8 or 1 classes and must give the same logits (tests/glm_dsa/glm_dsa_pack9_slow_test.cpp).
//
// Environment: MAYA_GLM_POOL_CLASSES=<n> caps the class count (0 or unset: no cap; 1: Maya's uniform slots).
// Plain C++ (no CUDA), on every backend.
#pragma once

#include <cstddef>
#include <vector>

namespace strata::core {

/// MAYA_GLM_POOL_CLASSES as a cap (0 = none); a negative or unparsable value is 0.
int glm_pool_class_cap(const char* env_value);

/// Folds classes [cap-1, n) into class cap-1 when n > cap > 0.  layer_class maps a layer to its class (-1 = no MoE
/// layer, untouched); stride is the bytes per slot of each class (the merged class takes the largest), weight and
/// layers are per-class demand counters (summed).  Every vector must have n entries (n = stride.size()); returns the
/// class count afterwards and records it for glm_pool_classes_last().  A cap <= 0 or >= n changes nothing.
int glm_pool_merge_classes(std::vector<int>& layer_class, std::vector<size_t>& stride, std::vector<size_t>& weight,
                           std::vector<int>& layers, int cap);

/// The class count of the most recent glm_pool_merge_classes call in this process (-1 before any): tests read it
/// after a load to know how many classes the pool really used.
int glm_pool_classes_last();

}  // namespace strata::core
