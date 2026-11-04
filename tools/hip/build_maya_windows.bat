@echo off
rem Project Maya on AMD under native Windows (experimental), after Strata's tools\hip\build_windows.bat.
rem Needs: Visual Studio 2022 or 2026 Build Tools (C++ x64), CMake + Ninja on PATH, and a Python venv with TheRock's ROCm wheels
rem (the rocm-sdk package; Strata's build_windows.bat pins the same ones) named by ROCM_VENV.
rem   set ROCM_VENV=C:\path\to\rocm-venv
rem   tools\hip\build_maya_windows.bat [tests]
rem No zip package: run from the build directory (BUILD_DIR, default build-hip-win). STRATA_HIP_ARCHS defaults to gfx1100.
setlocal enabledelayedexpansion
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined ROCM_VENV (echo set ROCM_VENV to the venv that holds TheRock's rocm-sdk & exit /b 1)
if not exist "%ROCM_VENV%\Scripts\rocm-sdk.exe" (echo no rocm-sdk.exe in %ROCM_VENV%\Scripts & exit /b 1)
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-hip-win"
if not defined STRATA_HIP_ARCHS set "STRATA_HIP_ARCHS=gfx1100"
set "TESTS=OFF"
if /i "%~1"=="tests" set "TESTS=ON"
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
set "ROCM_F=%ROCM:\=/%"
set "BITCODE=%ROCM_F%/lib/llvm/amdgcn/bitcode"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS="
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
if not defined VS (echo Visual Studio Build Tools not found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
if not exist "%BUILD_DIR%\build.ninja" (
  cmake -G Ninja -S "%SRC%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=%TESTS% -DSTRATA_PREFILL_MMQ=ON ^
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_HIP_ARCHITECTURES=%STRATA_HIP_ARCHS%" ^
    "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
    "-DCMAKE_HIP_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%ROCM_F%" ^
    "-DCMAKE_PREFIX_PATH=%ROCM_F%" "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%BITCODE%" || exit /b 1
)
cmake --build "%BUILD_DIR%" -- -k 0 || exit /b 1
echo BUILD OK
endlocal
