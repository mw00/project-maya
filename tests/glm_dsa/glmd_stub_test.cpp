// B0.12: an glm-dsa test registered through cmake/strata.cmake links against strata_engine.
#include <cstdio>
#include <cstring>
namespace glmd { const char* build_tag(); }
int main() {
  const char* t = glmd::build_tag();
  if (std::strcmp(t, "strata-glmd") != 0) { std::fprintf(stderr, "bad tag %s\n", t); return 1; }
  std::printf("glmd_stub_test OK\n");
  return 0;
}
