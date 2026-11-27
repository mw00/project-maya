// src/kernels/cpu/iq3v_rows_z.cpp - MSVC only (#1235): iq3v_rows.cpp's AVX-512 forms compiled on their own under
// /arch:AVX512 (IQ3V_PART 2: those forms and pass_z, nothing else - their rows run only where the caller found the CPU
// takes them, and the file has no code that runs at start).  GCC / Clang build iq3v_rows.cpp alone.
#define IQ3V_PART 2
#include "iq3v_rows.cpp"
