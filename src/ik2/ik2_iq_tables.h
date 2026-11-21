// src/ik2/ik2_iq_tables.h - IQ3_XXS's grid (iq3xxs_grid) and IQ4_XS's values (kvalues_iq4nl): ggml's own tables, from
// the copy Maya's other engines build with (third_party/ggml/ggml-common.h, MIT, third_party/ggml/LICENSE), not a
// second copy.  The header's tables are read twice, through its own CUDA and C++ branches: ik2_iq::dev (a device
// copy, read with __ldg - lanes index it divergently) and ik2_iq::host (the host's decode and the tests).  Include
// this before any other ggml-common.h.
#pragma once
#include <cstdint>

#if defined(GGML_COMMON_IMPL) || defined(GGML_TABLE_BEGIN)
#error "ik2_iq_tables.h reads ggml-common.h's tables itself: include it before another ggml-common.h"
#endif

namespace ik2_iq {
namespace dev {
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#undef GGML_COMMON_IMPL_CUDA
#undef GGML_COMMON_IMPL
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END
}  // namespace dev
namespace host {
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#undef GGML_COMMON_IMPL_CPP
#undef GGML_COMMON_IMPL
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END
}  // namespace host
}  // namespace ik2_iq
