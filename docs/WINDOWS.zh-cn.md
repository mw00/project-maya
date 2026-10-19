# Windows 上的 Maya

<p align="center"><a href="WINDOWS.md">English</a> | 简体中文</p>

回到 [README](../README.zh-cn.md)。

实验性：同一个安装器也能在 Windows 10/11（64 位）上原生安装 Maya，引擎与图像编码器用 Visual Studio 2022
和 CUDA 12.8 编译（RTX 20 及更新的卡可用 CUDA 13）。Maya 在 Linux 上开发和测量；有用户在 Windows 11 上
用 CUDA 13.4 构建并在 RTX 4090 上运行 Maya-L（见上方速度表，
[#63](https://github.com/mw00/project-maya/issues/63)）。欢迎告诉我们它在你的机器上跑得如何。

1. 安装一次：
   - NVIDIA 驱动和 [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)。
     RTX 20 及更新也可用 CUDA 13；V100 需要 CUDA 12.x。`nvidia-smi` 显示的 CUDA 版本只是驱动的能力上限，
     不代表已安装工具包（`nvcc`）。
   - [Visual Studio **2022** Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe)，勾选
     **Desktop development with C++**、**MSVC v143 - VS 2022 C++ x64/x86 build tools** 和一个
     **Windows 10 或 11 SDK**。完整的 Visual Studio IDE 可选。只装更新版本的 Visual Studio 过不了
     安装器的检查；2022 可以与它并存安装。
   - **64 位 Python 3.10+**，例如 `winget install -e --id Python.Python.3.12 --scope user`。
   - [Git for Windows](https://git-scm.com/download/win) 用来克隆和更新仓库；也可以直接下载源码 zip 解压。
2. 装好前置组件后，打开一个**新的 PowerShell 或命令提示符**。克隆仓库（或打开解压出的文件夹），然后检查电脑：

   ```powershell
   git clone https://github.com/mw00/project-maya.git
   cd project-maya
   .\START-MAYA.bat --check
   ```

   `--check` 会报告显卡、工具包、编译器、内存和页面文件，不下载模型也不编译引擎。Maya 自己会找 Visual
   Studio 的编译器环境，不需要 Developer Command Prompt。
3. 开始安装，把模型放到有足够[剩余空间](../README.zh-cn.md#运行要求)的 NVMe 固态硬盘上。
   把盘符和文件夹改成你自己的：

   ```powershell
   .\START-MAYA.bat --setup --models-dir "D:\Maya-models"
   ```

   按提示选模型和上下文长度。启动器会自建 `.venv`；setup 在里面安装 Python 包、CMake 和 Ninja，编译引擎，
   下载模型前会先询问。之后运行 `.\START-MAYA.bat`（或双击它）启动已装好的模型，再打开
   `http://127.0.0.1:8080`。`./maya.sh` 的所有选项在这里同样可用；PowerShell 需要开头的 `.\`。
   想用纯文本安装就加 `--plain`。

- **检查页面文件大小，即使开着"系统托管"。** Windows 的 GPU 分配和 Maya 锁定的内存层消耗的是提交容量
  （内存 + 页面文件）。页面文件太小会让内存层拿不满实际可用的空闲内存，更多专家留在固态硬盘上。
  `--check` 建议的容量约为所选显卡显存总和再加 9 GB（48 GB 显卡约 57 GB）。如果它给出警告，就按它建议的
  大小设置自定义页面文件，并留足磁盘空间：系统 > 关于 > 高级系统设置 > 性能 > 设置 > 高级 > 虚拟内存。
  改完后重启 Windows。大内存电脑上"系统托管"仍可能太小
  （[#20](https://github.com/mw00/project-maya/issues/20)）。
- **显存被塞满的显卡会换页而不是报错。** 显卡显存不够时，Windows 驱动会把 GPU 内存搬到系统内存，解码速度
  随即掉到每秒几个 token。专家池写入后会再读一次空闲显存（`--expert-cache auto`，默认），引擎日志在启动时
  报告每张卡的空闲量（低于 256 MiB 标 `LOW`）。drafter（DFlash2）在首个请求上还要多占几百 MB。如果日志
  警告，或第一个请求之后回答变慢，往配置的 `"args"` 里加 `"--vram-reserve-mib", "1500"`（举例），或用
  `--vram-reserve-mib 1500` 重新安装（[专家池的大小](SETTINGS.zh-cn.md#专家池的大小)）。
- 装了多个 CUDA 工具包时，setup 会选最新的兼容版本。想明确指定某个，就在安装命令后加
  `--nvcc "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"`。
- 安装完成后，先停掉正在运行的 Maya 服务器，再运行 `.\START-MAYA.bat --calibrate --no-start`，为你的电脑
  调好 CPU/PCIe 分工和线程数，然后正常启动 Maya。可选的内存常驻层等设置见[调参](SETTINGS.zh-cn.md)，
  `--bench` / `--report` 见[遇到问题？](../README.zh-cn.md#遇到问题)。
- 不要用 WSL2：WSL2 的 GPU 驱动只能锁定约 1 GB 内存，而 Maya 要锁定几十 GB。直接在 Windows 里运行
  `START-MAYA.bat`。
- Tesla V100 也能跑，用 CUDA 12.4 构建（[#57](https://github.com/mw00/project-maya/issues/57)）。跑一次
  `--calibrate`：它会为你的电脑调好 Maya，之后换上下文长度的安装会保留这些设置。
- Windows 上的 AMD（更加实验性），包括 Strix Halo / Gorgon Halo：`START-MAYA.bat --backend hip --gpu 0
  --setup` 用 AMD 的 HIP SDK 编译 HIP 引擎，或者在没有 HIP SDK 时提供 AMD 的 ROCm SDK wheel 装进
  `.venv`（[docs/AMD_MAYA.md](AMD_MAYA.md#windows)）。
  `tools\hip\build_maya_windows.bat` 可以手动编译（一块 RX 7900 XTX，
  [#54](https://github.com/mw00/project-maya/pull/54)）。
