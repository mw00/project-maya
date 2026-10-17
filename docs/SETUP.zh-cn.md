# Maya 的安装

<p align="center"><a href="SETUP.md">English</a> | 简体中文</p>

安装详解：第一次运行会做什么、再次启动与更新，以及全部选项。回到
[README](../README.zh-cn.md)。

## 第一次运行会做什么

需要 20–40 分钟，下载时间另计：

1. 检查电脑（显卡、驱动、CUDA 工具包、编译器、内存、CPU）；
2. 询问用哪几张显卡（默认全部，最多 16 张）、上下文多长（推荐 32K）、下载哪个模型：Maya-S（推荐；
   24 GB 及以下的卡选 Maya-S24）、Maya-M、Maya-M-Derisked 或 Maya-L，以及起草方式：模型的 MTP 模块
   （默认）或 [DFlash2 起草器](SETTINGS.zh-cn.md#使用-dflash2-起草)（NVIDIA RTX 30 / A100 及更新）；
3. 把它的 Python 包装进 `.venv`，并取回固定在某个 commit 的 llama.cpp 源码（两者都会列出并询问）；
4. 为你的显卡编译引擎（10–30 分钟，仅一次）；
5. **模型**：展示来源、体积（Maya-S：96.5 GB）和确切的 `curl` 命令，你回答 `y` 之后才下载；每个文件都会
   与公布的 sha256 核对。你也可以自己运行那些命令，或使用已有的文件：`./maya.sh --gguf-dir DIR`——
   这四种之外的量化属于实验性：能跑，但 Maya 没有用它们测量过。用 DFlash2 时还会把起草器（0.7 GB，
   按其 sha256 核对）下载到模型旁边；
6. 构建 *pack*——引擎对模型文件的索引，约 1 GB，写入模型文件夹。已经存在的 Maya 模型文件会先按 sha256
   核对（每个一次）；pack 记录这些文件，之后它们一旦改变，启动会拒绝使用它（`--repack` 重新构建）；
7. **图片**：编译图像编码器（10–20 分钟，仅一次）并取回它的文件（1.1 GB，先展示再询问）；
   `--no-vision` 可跳过；
8. 写出 `maya-<model>.json` 和 `run-maya-<model>.sh`；用 DFlash2 时会运行[速度校准](SETTINGS.zh-cn.md#速度校准)
   （5–20 分钟：只有在你的电脑上确实更快，DFlash2 才保留）；然后套用之前[为这台电脑调好的设置](SETTINGS.zh-cn.md)，
   或者提供调优（约 10–15 分钟；`./maya.sh --calibrate` 随时可用）——最后启动控制台。

> **启动需要几分钟：** 引擎会把大部分空闲内存（除约 6 GB 外的全部）固定为专家层专用，并预热缓存。
> Maya 运行期间，其他程序能拿到的内存很少。

## 下次启动，与更新

**下次启动**，直接再运行 `./setup.sh`（或模型的 `./run-maya-<model>.sh`）：它立刻启动，不会重复下载，
同一个屏幕上——控制台显示 Monitor 的数据（速度、专家住在哪、显卡、请求），Tab 键看日志。Ctrl+C 停止。

**更新：** 控制台的 **About** 会提示新版本，点它的 **Update** 按钮即可：Maya 下载新代码（git），只编译
改动过的引擎文件，并重新加载同一个模型——模型不会重新下载，你的设置也保留。手动方式：`git pull`，然后
`./setup.sh`。（这个检查向 GitHub 查询最新 release，至多每六小时一次；`MAYA_UPDATE_CHECK=0` 可关闭。）

## 选项

| 选项 | |
| --- | --- |
| `--setup` | 重新安装（换显卡、上下文、模型文件夹） |
| `--check` | 只检查电脑 |
| `--gguf-dir DIR` | 使用你已有的 GLM-5.3-Flash GGUF 文件：给文件夹，或文件夹里有多个模型时给那个 `.gguf` 文件（拆分模型的第一个）。文件夹必须可写：pack 放在里面 |
| `--models-dir DIR` | 下载的模型放哪里，每个模型一个以其命名的文件夹——`DIR/Maya-L/` 放该模型的 GGUF 文件（默认 `../Maya-data/models`）；放到 NVMe 上 |
| `--download-model` | 不询问直接下载模型（命令与体积仍会打印） |
| `--model NAME` | 不询问直接指定模型：`Maya-S-v2-IQ2_XXS`（Maya-S）、`Maya-S24`、`Maya-M`、`Maya-M-Derisked`、`Maya-L`，或 `EXL3-3.05bpw` / `EXL3-4.05bpw`——3.05 或 4.05 bits/weight 的实验性 EXL3 量化（[3.05bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw)，125.3 GB；[4.05bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/4.05bpw)，165.2 GB；均带 MTP 模块），引擎直接从它的 safetensors 原地读取：仅限 NVIDIA，3.05bpw 在 Linux 上测过，两者都没有与 FP8 模型对比测量 |
| `--no-vision` | 只要文本：不装图像编码器 |
| `--draft mtp` / `--draft dflash2` | 模型如何提前起草 token：它的 MTP 模块（默认），或 DFlash2 起草器——下载（0.7 GB）到 `DIR/GLM-5.3-Flash-DFlash2/` 并以 `--dflash` 传入（Maya-L：9 卡 +44%，2 卡 +12%，单卡 +19%；NVIDIA RTX 30 / A100 及更新——更老的卡除非显式传入，否则保留 MTP 模块；起草器许可证为 CC BY-NC-ND 4.0，仅限非商用）。[速度校准](SETTINGS.zh-cn.md#速度校准)之后，只有在你的电脑上确实更快才会保留它 |
| `--gpu N` / `--gpus 0,1,2,3` | 单卡，或几张卡（最多 16 张）共同拆分模型的层 |
| `--context N` | 上下文长度（token 数）：8192、32768（默认）、65536、131072 |
| `--vram-reserve-mib N` | 专家池在每张卡上留出的空闲显存（MiB，默认 700），保存在配置里——桌面或其他程序占着显存时调大（[专家池的大小](SETTINGS.zh-cn.md#专家池的大小)） |
| `--port N`, `--host 0.0.0.0 --api-key KEY` | 换端口；让局域网内可以访问（务必设置 key；多个：`KEY1,KEY2`） |
| `--env KEY=VALUE` | 写入配置的一条引擎设置（见[调参](SETTINGS.zh-cn.md)） |
| `--host-compiler g++-12` | 当你的 g++ 比 CUDA 接受的版本新时（"unsupported GNU version"） |
| `--rebuild`, `--repack` | 重新编译引擎 / 重新构建 pack |
| `--calibrate` | 为这台电脑调优引擎的 CPU 通道（见[调参](SETTINGS.zh-cn.md)），然后启动；配 `--no-start` 则只调优 |
| `--speed-calibration` | 再测一次 DFlash2 与 MTP 模块在这台电脑上解码谁更快（[方法](SETTINGS.zh-cn.md#速度校准)；5–20 分钟，需先停 Maya）并保留更快的一个，然后启动；配 `--no-start` 则只测量 |
| `--no-speed-calibration` | 配合 `--draft dflash2`：不做速度校准直接保留 DFlash2 |
| `--yes` | 采用推荐选项（模型下载仍需 `--download-model`） |
| `--plain` | 纯文本安装、手动回答问题，Maya 显示在终端而不是独立屏幕（管道输入也会这样；`--yes` 配纯文本安装）；安装屏幕的全部输出在 `maya-setup.log` |

## 在其他地址上的网页

浏览器会阻止一个来自其他地址的网页（比如 `http://127.0.0.1:8090` 上的本地聊天客户端）调用 Maya 的 API，
除非 Maya 允许那个页面。把页面列进配置 `maya-<model>.json`：

```json
"cors_origins": ["http://127.0.0.1:8090"]
```

写确切地址（协议、主机和端口，不带路径）；`["*"]` 放行所有页面——用它时务必同时设置 `api_key`，否则你
打开的任何页面都能用你的模型。它只开放 `/v1/*`，其他什么都不开放：设置和工具仍只属于 Maya 自己的页面。
不配置则一切照旧。重新 `--setup` 会重写配置：之后要把它补回去。
