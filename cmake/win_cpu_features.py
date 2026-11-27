"""Prints this Windows PC's CPU SIMD features for ggml-cpu's build (cmake/win_cpu_guard.cmake), e.g. avx,avx2,fma,f16c:
asked from the CPU itself (setup.win_simd_features), so nothing is compiled or run that Smart App Control could block.
Prints nothing when the CPU cannot be asked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup  # noqa: E402  (nothing runs on import)

print(",".join(setup.win_simd_features()) if sys.platform == "win32" else "")
