// src/kernels/cuda/glmd_arch.cuh - the glm-dsa kernels' spellings for a Pascal (sm_6x) device pass and for HIP.
//
// The maya target is built for Pascal too (maya.py MIN_CC 60, experimental), and nvcc compiles every .cu file of it
// once per architecture.  Two compute_70 features stop a Pascal pass (nvcc 12.8, sm_61):
//   __grid_constant__  "annotation is only allowed for architecture compute_70 or later": STRATA_GRID_CONSTANT leaves
//                      it out below sm_70, as glm_fast.cu's macro of the same name (the arguments are const either way);
//   WMMA               <mma.h> declares namespace nvcuda only for the host pass and sm_70+: GLMD_WMMA is 0 in a Pascal
//                      device pass, where the tensor-core kernels' bodies trap (Pascal has no tensor cores).
// The host pass and every sm_70+ pass are unchanged.  HIP (no __CUDA_ARCH__; hip_compat defines __grid_constant__ and
// __trap): there is no nvcuda::wmma at all, so GLMD_WMMA is 0 in every HIP pass (the rocWMMA versions are S7).
//
// GLMD_WMMA is the one guard of nvcuda::wmma code: <mma.h>, the wmma names and types, the WMMA kernels' bodies.  Host
// code that picks or launches a WMMA kernel (the _ok() / _enabled() checks, the launches, cudaFuncSetAttribute) is
// guarded by STRATA_USE_HIP instead: nvcc wants a kernel template's instantiations to be the same in the host pass and
// in every device pass, and GLMD_WMMA differs between them on Pascal.
#pragma once

#ifndef STRATA_GRID_CONSTANT
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define STRATA_GRID_CONSTANT __grid_constant__
#else
#define STRATA_GRID_CONSTANT
#endif
#endif

#if defined(STRATA_USE_HIP)
#define GLMD_WMMA 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define GLMD_WMMA 1
#else
#define GLMD_WMMA 0
#endif
