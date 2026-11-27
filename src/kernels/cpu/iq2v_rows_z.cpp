// src/kernels/cpu/iq2v_rows_z.cpp - MSVC only (#1235): iq2v_rows.cpp's AVX-512 form compiled on its own under
// /arch:AVX512 (IQ2V_PART 2: that form and pass_z, nothing else - its rows run only where the caller found the CPU
// takes them, and the file has no code that runs at start).  GCC / Clang build iq2v_rows.cpp alone.
#define IQ2V_PART 2
#include "iq2v_rows.cpp"
