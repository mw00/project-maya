<h1 align="center">Project Maya</h1>

<p align="center"><a href="README.md">English</a> | 简体中文</p>

<p align="center"><b>在你自己的 GPU 上运行 3210 亿参数的 AI 模型</b><br>
GLM-5.3-Flash，私密且快速 · 单块 NVIDIA GPU 或最多 16 块 · 支持 AMD 与 Windows（实验性）· 聊天、图片、
兼容 OpenAI 与 Anthropic 的 API · 多请求并发</p>

<p align="center">由 <b>Peasant Smith</b>（<a href="https://github.com/mw00">@mw00</a>）创建并主导，与
<b><a href="https://github.com/needmorevram">@needmorevram</a></b> 共同维护</p>

<p align="center"><a href="https://buymeacoffee.com/peasantsmith">☕ 支持 Project Maya —— 请作者喝杯咖啡</a></p>

这种规模的模型通常需要数据中心。而 Maya 能在你已有的硬件上运行
**[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**（zai-org，MIT 许可证）：321 B 参数，每个
token 约 18 B 激活，上下文最长可达 1M token。它的引擎专为这一个模型打造：把你的对话最常用到的专家（expert）
放在 GPU 上，次常用的放在内存里，其余的放在 NVMe 固态硬盘上，并随着你聊天而动态搬运。它会检测你的 GPU、CPU、
内存和固态硬盘，并据此自动调优。任何数据都不会离开你的机器。

- **快：** 四张 RTX 4090 上最高 118 tokens/s，单张 RTX PRO 6000 上 81 tokens/s，单张 RTX 5090 上 30+
  tokens/s。下面是真实用户的真实数据。
- **接近原模型：** Maya 自研量化在零样本精度上保留了 FP8 模型的 97.7–99.2%。
- **接入你的工具：** 提供 OpenAI 与 Anthropic 兼容 API，支持工具调用（tool calls），可直接对接编码 agent。
- **多请求并发：** 最多可同时处理 8 个请求——适合运行多个会话的 agent，或一台电脑多人共用。
- **开箱即用：** 浏览器内置聊天与实时监控、图片、思考等级、一条命令完成安装。

## 模型

Project Maya 自研的 GLM-5.3-Flash 量化版本，基于 Z.ai 发布的 FP8 权重制作，并使用 FP8 模型的统计信息校准。
每个版本都保留了模型的 MTP 模块（用于提前起草 token）。

| 模型 | 体积 | 适合 | 相对 FP8 模型 |
| --- | ---: | --- | --- |
| **Maya-S**（默认） | 96.5 GB | 内存池较小（内存与显存合计）的电脑 | 零样本精度为 FP8 的 97.9%；83% 的情况下预测出相同的下一个 token |
| **Maya-S24** | 94.7 GB | 24 GB 及以下的显卡：注意力用 4-bit，给专家多腾出约 1.5 GB 显存，解码 +14% | 97.7%；83% |
| **Maya-M** | 116 GB | 更大的内存池；在关键处分配更多位数，针对工具调用与前端代码校准 | 97.9%；86% |
| **Maya-M-Derisked** | 116 GB | 在 Maya-M 基础上由 [Blackfrost_AI](https://x.com/Blackfrost_AI) 做的权重微调，减少一刀切的拒答（[其仓库](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF)）；实验性 | 尚未测得 |
| **Maya-L** | 156.3 GB | 最大的内存池 | **99.2%**；90% |

安装程序会询问要下载哪一个，并为你的电脑给出推荐。它们的制作与测量方法：[MAYA-S](bench/results/MAYA-S.md)、
[MAYA-S24](bench/results/MAYA-S24.md)、[MAYA-M](bench/results/MAYA-M.md)、[MAYA-L](bench/results/MAYA-L.md)。
模型位于 [Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)。

<details>
<summary>同样能在 Maya 上运行的社区量化版本</summary>

| 模型 | 体积 | 用法 | 备注 |
| --- | ---: | --- | --- |
| [EXL3 3.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw)（turboderp 出品） | 125.3 GB | 安装时：`--model EXL3-3.05bpw` | 仅 NVIDIA，实验性 |
| [EXL3 4.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/4.05bpw)（turboderp 出品） | 165.2 GB | 安装时：`--model EXL3-4.05bpw` | 仅 NVIDIA，实验性 |

</details>

## 有多快？

真实机器、真实数据，多数用 `./maya.sh --bench` 测得。一个 token 约等于 ¾ 个英文单词。

| 机器 | 模型 | 解码（生成回答） | 预填充（读取你的提示） | 来源 |
| --- | --- | ---: | ---: | --- |
| **4x RTX 4090 24 GB**（PCIe 4.0 x16），Xeon w7-3445，62 GB 内存 | Maya-S24 | **118 tokens/s** | 4129 tokens/s | @0xPreDa（v1.0.31，#89） |
| **RTX PRO 6000 Blackwell 96 GB**，Ryzen 7 9800X3D，61 GB 内存 | Maya-M，256K 上下文 | **81.0 tokens/s** | 3102 tokens/s | @yundddd（v1.1.7，#172） |
| **2x NVIDIA CMP 170HX 64 GB**（每卡 PCIe Gen2 x4），2x Xeon E5-2690 v4，91 GB 内存 | Maya-M，全部专家驻留显存 | 61.3 tokens/s | 514 tokens/s | @ZackO2o（v1.0.6，#22） |
| **2x RTX 5880 Ada 45 GB**（PCIe 4.0 x16），Xeon Gold 5420+，393 GB 内存 | Maya-L，256K 上下文 | 54.5 tokens/s | 1401 tokens/s | @vjsyong（v1.1.4，#149） |
| **2x Tesla V100 32 GB**（每卡 PCIe 3.0 x16），2x Xeon Gold 6138，183 GB 内存 | Maya-S24 | 54.5 tokens/s | 542 tokens/s | 维护者（v1.1.13，#226） |
| **2x RTX 3090 24 GB**（PCIe 4.0 x16，功耗限制在 225 W），Threadripper 3970X，128 GB 内存 | Maya-S24 | 53.0 tokens/s | 885 tokens/s | @briepala（v1.1.2，#135） |
| **4x Tesla V100 32 GB**（PCIe 3.0 x16），24 个 CPU 核心，100 GiB 内存上限 | Maya-L，32K 上下文 | 48.3 tokens/s（`--calibrate` 之后） | 499 tokens/s | @areslp（v1.1.3，[#130](https://github.com/mw00/project-maya/issues/130)) |
| **2x RTX 3090 24 GB**（PCIe 4.0 x16，功耗限制在 250 W），EPYC 7452，256 GB 内存 | Maya-M / Maya-L，128K 上下文 | 44.0 / 30.2 tokens/s | 860 / 801 tokens/s | @orochifj（v1.1.4，#165） |
| **2x RTX 4090 24 GB**（PCIe 3.0 + 4.0 x16），2x Xeon Silver 4309Y，188 GB 内存 | Maya-S24 | 40.5 tokens/s | 1271 tokens/s | @fuad00（v1.1.0，#122） |
| **2x Tesla V100 32 GB**（PCIe 3），Xeon E5-2690 v4，30 GB 内存 | Maya-S | 最高 40 tokens/s | 最高 670 tokens/s | 维护者 |
| **2x Tesla V100 32 GB**（每卡 PCIe 3.0 x8），Ryzen 9 7945HX3D，96 GB 内存 | Maya-M / Maya-L | 33.8 / 13.3 tokens/s | 490 / 372 tokens/s | @nkhaooat（v1.1.11，#208） |
| **RTX 3090 24 GB + RTX 3060 12 GB**（PCIe 4.0 x16），EPYC 7702P（8 通道 DDR4），141 GB 内存（虚拟机） | Maya-M，128K 上下文 | 33.7 tokens/s | 682 tokens/s | @erikash（v1.1.4，#157） |
| **RTX 5090 32 GB**，Ryzen 9 7950X，96 GB DDR5 | Maya-M | 30+ tokens/s | ~2000 tokens/s | @siganos（#43） |
| **AMD Radeon 8065S**（Ryzen AI Max+ PRO 495，192 GB LPDDR5X，其中 160 GB 分给 GPU），Windows 11 | Maya-S，32K 上下文 | 最高 29.6 tokens/s（8K token 提示后为 23.2） | 278 tokens/s | 维护者（v1.1.10） |
| **1x RTX 3090 24 GB**（PCIe 4.0 x16），EPYC 7702P（8 通道 DDR4），141 GB 内存（虚拟机） | Maya-M，128K 上下文 | 27.0 tokens/s | 1069 tokens/s | @erikash（v1.1.4，#157） |
| **1x Tesla V100 32 GB**（PCIe 3），2x Xeon Gold 6138，183 GB 内存 | Maya-S24 | 26.8 tokens/s | 541 tokens/s | 维护者（v1.1.7，#171） |
| **RTX 4080 SUPER 32 GB**（32 GB 改装卡），Threadripper 3970X，157 GB 内存，SATA 固态硬盘 | Maya-S24，262K 上下文 | 26.7 tokens/s | 1552 tokens/s | @Micro-ATP（v1.1.11，#200） |
| **3x Tesla V100 16 GB**（PCIe 3.0 x16），2x Xeon E5-2690 v4，62 GB 内存 | Maya-S，128K 上下文 | 26.6 tokens/s | 497 tokens/s | @saatvik-viniak（v1.1.11，#211） |
| **4x RTX 3080 20 GB**，64 GB DDR4 | Maya-M，512K 上下文 | 26.3 tokens/s（一整天使用） | - | @CoolJohn-lab（#104） |
| **1x RTX 3090 24 GB**（功耗限制在 225 W），Threadripper 3970X，128 GB 内存 | Maya-S24 | 26.0 tokens/s | 913 tokens/s | @briepala（v1.1.2，#135） |
| **RTX PRO 4000 Blackwell 24 GB**，Core i9-13900K，128 GB DDR5 | Maya-S24 | 24.7 tokens/s | 1146 tokens/s | @morrisfamily（v1.1.7，#174） |
| **RTX 5090 32 GB**（PCIe 5.0 x16），Core i9-13900K，64 GB 内存，Windows 11 | Maya-S，128K 上下文 | 23.0 tokens/s | 2173 tokens/s | @gianbing（v1.1.3，#139） |
| **4x Tesla T4 16 GB**，Xeon E5-2660 v2，220 GB 内存 | Maya-S | 22.6 tokens/s | 416 tokens/s | @ksanislo（v1.0.16，#36） |
| **1x Tesla V100 32 GB**（PCIe 3），Core i5-12600T，64 GB 内存 | Maya-S | 最高 19 tokens/s | 最高 620 tokens/s | 维护者 |
| **AMD Radeon 8060S**（Ryzen AI Max+ 395，128 GB，其中 96 GB 分给 GPU），Windows 11 | Maya-S | 18.3 tokens/s | 320 tokens/s | @LIN-LAB-AI（v1.0.30，#102） |
| **RTX 3090 Ti 24 GB**（PCIe 3.0 x16，虚拟机），EPYC 7302P，240 GB 内存 | Maya-L，32K 上下文 | 18.2 tokens/s（调优后，#177） | 791 tokens/s | @Jw88888（v1.1.7，#177） |
| **RTX 4090 24 GB**，Ryzen 9 9950X3D，192 GB DDR5，Windows 11 | Maya-L，128K 上下文 | 16.4 tokens/s | 1371 tokens/s | @npc97（v1.0.24，#63） |
| **RTX 5090 Laptop 24 GB**，Core Ultra 9 275HX，64 GB 内存，Windows 11（笔记本） | Maya-S24 | 14.5 tokens/s | 994 tokens/s | @klvnblst（v1.0.30，#67） |

预填充按 8K token 的提示计。速度不会随对话变长而下降；刚启动后的最初几轮回答最慢，因为缓存还在填充你常用的
内容。欢迎分享你的数据：先 `--bench`，再 `--report`，提交到
[GitHub issue](https://github.com/mw00/project-maya/issues)。

**多请求并发**（`--parallel`）能叠加提速：在单块 GPU 上，多个请求各取一个 token 一起穿过模型，稠密权重只需
读取一次。以所有请求合计的 tokens/s 对比逐个问同样的问题：在从内存流式读取 Maya-S 大部分专家的 V100 上，
2 个请求 x1.41、4 个请求 x1.35；在把全部专家都放进 GPU 显存的 Gorgon Halo（Radeon 8065S，160 GB 给 GPU）上，
2 个请求 x1.67、4 个请求 x2.02。显存装下的模型越多，提升越大（[详情](docs/SETTINGS.zh-cn.md#多个请求同时)）。

## 运行要求

| | |
| --- | --- |
| **GPU** | NVIDIA，计算能力 7.0 或更新（V100、RTX 20 及以后）：单卡，或最多 16 卡共同承载模型。显存越多越快。Pascal（Tesla P40 / P100、GTX 10）为实验性支持，尚未充分验证。**AMD（实验性）：** RX 7900 XT / XTX、Radeon AI PRO R9700 / RX 9070、Strix Halo 与 Gorgon Halo（Radeon 8060S / 8065S）；RX 6800 / 6900 与 Radeon PRO V620 / W6800 尚未验证；仅支持文本——[docs/AMD_MAYA.md](docs/AMD_MAYA.md) |
| **内存** | 32 GB 即可运行。内存越多，驻留的专家越多，速度越快 |
| **硬盘** | **一块高速 NVMe 固态硬盘**，需容纳模型（Maya-S 约 100 GB，Maya-L 约 160 GB）及其图片编码器（1.1 GB）。机械硬盘不行 |
| **系统** | Linux（x86-64；CPU 最好支持 AVX2）、NVIDIA 驱动、CUDA 12.x（RTX 20 及以后可用 CUDA 13，V100 与 Pascal 不支持 CUDA 13）、g++、Python 3.10+。Windows 10/11（实验性）：[docs/WINDOWS.zh-cn.md](docs/WINDOWS.zh-cn.md)。不支持 WSL2。AMD：ROCm 7 |

安装程序会检查以上所有内容，对缺失的项打印出确切的安装命令，且不会改动系统级的任何东西。

## 安装

```sh
git clone https://github.com/mw00/project-maya.git && cd project-maya
./setup.sh
```

回答几个问题（直接回车即采用推荐选项）。安装程序会在每个下载开始前先展示它，一次性完成全部构建（20–40 分钟，
下载时间另计），然后**启动模型**。接着打开 `http://127.0.0.1:8080`。Ctrl+C 停止运行，再执行 `./setup.sh`
即可立刻重新启动。

- **Windows：** 用 `START-MAYA.bat` 代替 `./setup.sh`——见 [docs/WINDOWS.zh-cn.md](docs/WINDOWS.zh-cn.md)。
- **AMD：** 见 [docs/AMD_MAYA.md](docs/AMD_MAYA.md)。
- **更新：** 通过控制台的 **About > Update**，或 `git pull` 后再 `./setup.sh`。模型不会重新下载。
- **全部选项**（显卡、上下文、已有的模型文件夹、远程访问等）：[docs/SETUP.zh-cn.md](docs/SETUP.zh-cn.md)。

> **启动需要几分钟：** 引擎会把大部分空闲内存（除约 6 GB 外的全部）固定为专家层专用，因此 Maya 运行期间
> 其他程序可用的内存很少。

## 使用

- **在浏览器里：** 聊天（记录保存在浏览器本地、支持每会话自定义指令、编辑并重新生成、HTML 预览）、展示模型与
  硬件状态的实时 **Monitor**（细到每块显卡显存里装了什么），以及在设置里可修改的上下文长度。
- **你的应用与编码 agent：** OpenAI 兼容 API 位于 `http://127.0.0.1:8080/v1`，Anthropic 兼容 API 位于
  `http://127.0.0.1:8080/v1/messages`（Claude Code：`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`）。工具调用会
  随模型输出流式返回；请求要求时（`"return_progress": true`）也会流式返回长提示的读取进度。允许其地址后，
  来自其他地址的网页也能调用（`"cors_origins"`，
  [docs/SETUP.zh-cn.md](docs/SETUP.zh-cn.md#在其他地址上的网页)）。
- **思考：** 关闭、低、高（默认）或最大，可在设置里改，也可在单个请求里指定。
- **图片：** 在聊天里直接附加，或通过任一 API 发送图片部分。
- **从其他设备访问：** `./maya.sh --setup --host 0.0.0.0 --api-key <secret>`。务必设置 API key。
- **调优：** `./maya.sh --calibrate` 用 10–15 分钟把引擎调到适配你的硬件。全部设置项：
  [docs/SETTINGS.zh-cn.md](docs/SETTINGS.zh-cn.md)。
- **多请求并发：** `./maya.sh --parallel 4`（Windows：`START-MAYA.bat --parallel 4`；最多 8）会写入配置，
  控制台会显示正在生成几个回答。每个额外请求都要在显存里保留自己的上下文，因此显存能容纳的专家会略少
  （[详情](docs/SETTINGS.zh-cn.md#多个请求同时)）。
- **长上下文：** 新安装会把每个注意力层的缓存放在内存、显存里只留一个窗口（`--kv-streaming on`；已有配置执行
  `./maya.sh --setup` 可加上），从而显存能容纳更多专家——在 RTX 3090 上 512K 上下文时，提示处理快 4–9 倍、
  解码 +5%；1M 上下文单张 3090 就能装下（[详情](docs/SETTINGS.zh-cn.md)）。
- **用 DFlash2 起草器更快出答案：** `./maya.sh --setup --draft dflash2`（NVIDIA RTX 30 / A100 及以后）会下载
  一个小型起草器，每轮最多起草 7 个 token 且答案不变——对比 MTP 模块，Maya-L 在 9 卡上 +44%、2 卡上 +12%，
  单张 RTX 3090 上 +19%（[详情](docs/SETTINGS.zh-cn.md#使用-dflash2-起草)）。实际收益取决于电脑：在从内存读取
  大部分模型的单卡上常常反而更慢。因此速度校准会在你的电脑上实测，只在更快的地方启用。该起草器的许可证为
  非商用（CC BY-NC-ND 4.0）。

## 遇到问题？

在回答变慢或报错后立刻运行 **`./maya.sh --report`**（Windows：`START-MAYA.bat --report`），把它生成的
`maya-report.txt` 附到 [GitHub issue](https://github.com/mw00/project-maya/issues) 里。文件包含你的硬件、
配置和引擎的速度日志；不会自动向任何地方发送任何内容。引擎自身的日志是 Maya 文件夹里的 `maya-<model>.log`。

- **“nvcc ... cannot build for these GPUs”：** Volta 需要 CUDA 12.x，Blackwell 需要 12.8 或更新。安装程序
  会自动选用已安装的、匹配的最新工具包。
- **缺少 `cuda_profiler_api.h` 或 `nvtx3/nvToolsExt.h`：** Maya 的 CUDA 构建除了 `nvcc` 和运行库之外，
  还需要 profiler API 与 NVTX 头文件。请为你的工具包安装这些开发组件。对于拆分安装的 CUDA 13 工具包，
  NVIDIA 的官方头文件 wheel 也可以放在项目本地文件夹：

  ```sh
  python3 -m pip install --target .cuda-headers \
    nvidia-cuda-profiler-api==13.3.27 nvidia-nvtx==13.3.29
  CPATH="$PWD/.cuda-headers/nvidia/cu13/include${CPATH:+:$CPATH}" ./maya.sh --setup
  ```

  请选择与你的工具包匹配的 wheel 版本。此头文件布局在 CUDA 13.3、`nvidia-cuda-profiler-api==13.3.27` 和
  `nvidia-nvtx==13.3.29` 下验证过；其他 CUDA 大版本可能使用不同的布局。更新后重新构建时请使用相同的
  `CPATH`。无需改动任何共享的工具包文件。
- **“unsupported GNU version”：** `./maya.sh --setup --host-compiler g++-12`（先安装 `g++-12`）。
- **很慢、固态硬盘一直在读写：** 空闲内存不够放专家了。关掉一些程序，或者加内存。
- **Windows：回答突然慢了几倍：** 显卡显存太满，Windows 把 GPU 内存搬到内存里去了。给显存留些余量：
  `START-MAYA.bat --setup --vram-reserve-mib 2048`
  （[专家池的大小](docs/SETTINGS.zh-cn.md#专家池的大小)）。
- **下载中断：** 再运行一次 `./maya.sh`，会从中断处继续。
- **8080 端口被占用：** Maya 已经在运行了；或者用 `--port 8081` 再启动一个。

## 支持

Project Maya 免费且开源。如果它对你有用，可以在
**[buymeacoffee.com/peasantsmith](https://buymeacoffee.com/peasantsmith)** 支持它。谢谢！欢迎提交测速报告、
bug 报告和 PR：[CONTRIBUTING.md](CONTRIBUTING.md) 说明了哪些贡献最有帮助。

## 致谢与许可证

- **人员：** Project Maya 由 Peasant Smith（[@mw00](https://github.com/mw00)、
  [X](https://x.com/PeasantSmith)）创建并主导，他拥有该项目、其名称以及 Maya 模型。
  [needmorevram](https://github.com/needmorevram)（[X](https://x.com/needmorevram)）是共同维护者，贡献了
  多项重要特性：DFlash2 起草、EXL3 支持、KV 流式传输、会话检查点、层拆分搜索、专家画像与 CPU 专家内核。
  同时感谢所有提交代码和测速报告的朋友
  （[贡献者](https://github.com/mw00/project-maya/graphs/contributors)）。
- **基于 [Strata](https://github.com/Niko1221/Strata)**（MIT 许可证，Copyright (c) 2026 Niko1221 及 Strata
  贡献者）：Maya 的引擎源自 Strata，并针对 GLM-5.3-Flash 重写（横跨显存、内存与固态硬盘的专家分层、层拆分、
  MTP 解码），其服务端与控制台也是从 Strata 发展而来。**以及基于
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)**（MIT 许可证，Copyright (c) 2023-2026 The ggml
  authors）：量化格式、CPU 点积和提示路径的 MMQ 内核，基于固定的 commit 构建（`third_party/ggml/LICENSE`）。
- **EXL3 支持** 遵循 [exllamav3](https://github.com/turboderp-org/exllamav3) 的格式、码本、CPU 专家内核与
  张量核分块解码（MIT 许可证，Copyright (c) 2025 Turboderp；
  `third_party/exllamav3/LICENSE`）。**DFlash2 起草器的前向计算** 遵循
  [beellama.cpp](https://github.com/Anbeeld/beellama.cpp) 的实现（MIT 许可证，Copyright (c) 2023-2026
  The ggml authors 与 Anbeeld；`third_party/beellama.cpp/LICENSE`）。
- 模型：[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)，来自 zai-org（MIT）；Maya 的量化
  版本与图片编码器文件均制作自 zai-org 发布的权重，并沿用其许可证。可选的 DFlash2 起草器
  （[GLM-5.3-Flash-DFlash2-GGUF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF)，Anbeeld 出品）
  为 CC BY-NC-ND 4.0，仅限非商用，且只在你选择时才下载。
- 控制台的字体：Outfit（SIL Open Font License 1.1，`serve/web/fonts/OFL.txt`）。
- Maya 依据 [MIT 许可证](LICENSE)开源；Strata、ggml、exllamav3 与 beellama.cpp 的声明随每一份副本保留。
