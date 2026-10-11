# Maya on Windows

Back to the [README](../README.md).

Experimental: the same installer sets Maya up natively on Windows 10/11 (64-bit), and the engine and the image
encoder compile there with Visual Studio 2022 and CUDA 12.8 (CUDA 13 works for RTX 20 and newer). Maya is developed
and measured on Linux; on Windows 11 a user runs Maya-L on an RTX 4090, built with CUDA 13.4 (the speed table above,
[#63](https://github.com/mw00/project-maya/issues/63)). Tell us how it runs on yours.

1. Install once:
   - The NVIDIA driver and [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive).
     CUDA 13 also works for RTX 20 and newer; V100 needs CUDA 12.x. The CUDA version shown by `nvidia-smi` is the
     driver's capability, not proof that the toolkit (`nvcc`) is installed.
   - [Visual Studio **2022** Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe), with
     **Desktop development with C++**, **MSVC v143 - VS 2022 C++ x64/x86 build tools** and a **Windows 10 or 11 SDK**.
     The full Visual Studio IDE is optional. A newer Visual Studio alone does not satisfy the installer's check;
     2022 can be installed alongside it.
   - **64-bit Python 3.10+**, for example `winget install -e --id Python.Python.3.12 --scope user`.
   - [Git for Windows](https://git-scm.com/download/win) to clone and update the repository, or download and extract
     its source zip instead.
2. Open a **new PowerShell or Command Prompt** after installing the prerequisites. Clone the repository (or open
   the extracted folder), then check the PC:

   ```powershell
   git clone https://github.com/mw00/project-maya.git
   cd project-maya
   .\START-MAYA.bat --check
   ```

   `--check` reports the GPUs, toolkit, compiler, RAM and page file without downloading the model or building the
   engine. Maya finds Visual Studio's compiler environment itself; a Developer Command Prompt is not needed.
3. Start the setup, putting the model on an NVMe SSD with enough [free space](../README.md#what-you-need) for the chosen model.
   Change the drive and folder to suit your PC:

   ```powershell
   .\START-MAYA.bat --setup --models-dir "D:\Maya-models"
   ```

   Choose the model and context when asked. The launcher creates its own `.venv`; setup installs its Python
   packages, CMake and Ninja there, builds the engine, and asks before downloading the model. Later, run
   `.\START-MAYA.bat` (or double-click it) to start the installed model, then open `http://127.0.0.1:8080`.
   All `./maya.sh` options also work here; PowerShell needs the leading `.\`. Add `--plain` for plain-text setup.

- **Check the page-file size, even with "System managed" enabled.** Windows' GPU allocations and Maya's pinned
  RAM tier consume commit capacity (RAM + page file). A small page file can keep the RAM tier below the free RAM
  available, leaving more experts on the SSD. `--check` recommends roughly the selected GPUs' total VRAM plus
  9 GB of page-file capacity (about 57 GB for a 48 GB card). If it warns, set a custom size at least as large as it
  recommends, with enough free disk space: System > About > Advanced system settings > Performance > Settings >
  Advanced > Virtual memory. Restart Windows after changing it. "System managed" can remain too small on a
  large-RAM PC ([#20](https://github.com/mw00/project-maya/issues/20)).
- With several CUDA toolkits installed, setup chooses the newest compatible one. To select one explicitly, add
  `--nvcc "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"` to the setup command.
- After setup, stop any running Maya server and run `.\START-MAYA.bat --calibrate --no-start` to tune the CPU/PCIe
  split and thread count for your PC. Then start Maya normally. See [Tuning](SETTINGS.md) for the optional RAM-resident
  tier and other settings, and [Something went wrong?](../README.md#something-went-wrong) for `--bench` / `--report`.
- Not WSL2: Strata measured that WSL2's GPU driver pins only about 1 GB of RAM, and Maya pins tens of GB. Run
  `START-MAYA.bat` in Windows itself.
- A Tesla V100 runs it too, built with CUDA 12.4 ([#57](https://github.com/mw00/project-maya/issues/57)). Run
  `--calibrate` once: it tunes Maya for your PC, and a setup for another context keeps its settings.
- AMD on Windows (more experimental still), Strix Halo / Gorgon Halo included: `START-MAYA.bat --backend hip --gpu 0
  --setup` builds the HIP engine with AMD's HIP SDK, or offers AMD's ROCm SDK wheels in `.venv` when there is none
  ([docs/AMD_MAYA.md](AMD_MAYA.md#windows)).
  `tools\hip\build_maya_windows.bat` builds it by hand (an RX 7900 XTX,
  [#54](https://github.com/mw00/project-maya/pull/54)).
