@echo off
rem maya-ik2 (Inkling 2 Mini) on an AMD card under native Windows: configures the tree for HIP and builds only the
rem engine's targets - maya-ik2, its kernel library and ik2_kernels_test - instead of every engine of the repository.
rem The configure step is tools\hip\build_maya_windows.bat's; two environment fixes are added:
rem   * AMD's HIP SDK installer leaves HIP_PATH (and HIP_PATH_64) pointing at its own ROCm, and clang then reads
rem     ^<hip/hip_runtime.h^> from there even with --rocm-path - it stops on '__AMDGCN_WAVEFRONT_SIZE' - so both are
rem     set to the wheels' ROCm for this run;
rem   * rocm-sdk asks the GPU which family it is unless ROCM_SDK_TARGET_FAMILY names it, so the family is pinned to
rem     the first architecture of MAYA_HIP_ARCHS.
rem Needs: Visual Studio 2022 or 2026 Build Tools (C++ x64), CMake and Ninja on PATH, and a Python venv with TheRock's
rem ROCm wheels (the rocm-sdk package) named by ROCM_VENV (docs/AMD_MAYA.md, Windows).
rem   set ROCM_VENV=C:\path\to\rocm-venv
rem   set MAYA_HIP_ARCHS=gfx1201           (default; "gfx1201;gfx1100" builds one binary for RDNA4 and RDNA3)
rem   tools\hip\build_ik2_windows.bat [clean]
setlocal enabledelayedexpansion
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined ROCM_VENV (echo set ROCM_VENV to the venv that holds TheRock's rocm-sdk & exit /b 1)
if not exist "%ROCM_VENV%\Scripts\rocm-sdk.exe" (echo no rocm-sdk.exe in "%ROCM_VENV%\Scripts" & exit /b 1)
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-hip-win"
if not defined MAYA_HIP_ARCHS set "MAYA_HIP_ARCHS=gfx1201"
set "MAYA_HIP_FAMILY=%MAYA_HIP_ARCHS%"
for /f "delims=; " %%A in ("%MAYA_HIP_ARCHS%") do set "MAYA_HIP_FAMILY=%%A"
if /i "%~1"=="clean" (if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%")

set "ROCM_SDK_TARGET_FAMILY=%MAYA_HIP_FAMILY%"
set "ROCM="
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
if not defined ROCM (echo rocm-sdk path --root gave nothing & exit /b 1)
if not exist "%ROCM%\lib\llvm\bin\clang++.exe" (echo no clang++ in "%ROCM%\lib\llvm\bin" - run: "%ROCM_VENV%\Scripts\rocm-sdk.exe" init & exit /b 1)
set "ROCM_F=%ROCM:\=/%"
set "BITCODE=%ROCM_F%/lib/llvm/amdgcn/bitcode"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS="
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
if not defined VS (echo Visual Studio Build Tools not found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "HIP_PATH_64=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
where cmake >nul 2>nul || (echo cmake not found on PATH & exit /b 1)
where ninja >nul 2>nul || (echo ninja not found on PATH & exit /b 1)

rem (configured every time: a few seconds, and what makes a changed MAYA_HIP_ARCHS reach the build)
cmake -G Ninja -S "%SRC%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
  -DMAYA_ENABLE_HIP=ON -DMAYA_ENABLE_CUDA=OFF -DMAYA_BUILD_TESTS=ON -DMAYA_PREFILL_MMQ=ON ^
  -DMAYA_NATIVE_EXPERTS=ON -DMAYA_PORTABLE=ON "-DCMAKE_HIP_ARCHITECTURES=%MAYA_HIP_ARCHS%" ^
  "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
  "-DCMAKE_HIP_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%ROCM_F%" ^
  "-DCMAKE_PREFIX_PATH=%ROCM_F%" "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%BITCODE%" || exit /b 1
cmake --build "%BUILD_DIR%" --target ik2_kernels maya-ik2 ik2_kernels_test || exit /b 1
rem Windows takes a program's DLLs from its own folder, then System32 - where AMD's driver puts its own HIP runtime -
rem and PATH only after: the runtime the engine was built with goes next to it, as maya.py's copy_hip_dlls does.
for %%D in ("%ROCM%\bin\amdhip64*.dll" "%ROCM%\bin\amd_comgr*.dll" "%ROCM%\bin\rocm_kpack*.dll") do (
  copy /y "%%~fD" "%BUILD_DIR%\" >nul || echo could not copy %%~nxD next to maya-ik2.exe
)
echo BUILD OK - %BUILD_DIR%\maya-ik2.exe; hipBLAS and the other ROCm DLLs it needs are in %ROCM%\bin
endlocal
