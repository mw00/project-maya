<p align="center"><a href="README.md">English</a> | <a href="README.ja.md">日本語</a> | 简体中文 | <a href="README.pt-BR.md">Português</a></p>

<h1 align="center">Project Maya</h1>

<p align="center"><b>在你自己的 GPU 上运行 GLM-5.3-Flash (321 B)、DeepSeek-V4.1-Flash-1017 (748 B) 和 MiniMax-M3.1-Flash (428 B)</b><br>
私密且快速 · 1 块到最多 16 块 NVIDIA GPU · 支持 AMD 和 Windows (实验性) · 聊天、图片、兼容 OpenAI 和 Anthropic 的 API · 同时处理多个请求</p>

<p align="center">
<a href="https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/glm.svg" alt="GLM-5.3-Flash Maya downloads"></a>
<a href="https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/glm-m-derisked.svg" alt="GLM-5.3-Flash Maya-M-Derisked downloads"></a>
<a href="https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/deepseek.svg" alt="DeepSeek-V4.1-Flash Maya downloads"></a>
<a href="https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/minimax.svg" alt="MiniMax-M3.1-Flash Maya downloads"></a>
</p>

<p align="center">由 <b>Peasant Smith</b> (<a href="https://github.com/mw00">@mw00</a>) 创建并主导，与 <b><a href="https://github.com/needmorevram">@needmorevram</a></b> 共同维护</p>

<p align="center"><a href="https://github.com/sponsors/mw00"><img src="https://img.shields.io/badge/Sponsor-mw00-ea4aaa?logo=githubsponsors&logoColor=white&style=for-the-badge" alt="Sponsor on GitHub" height="50"></a>
<a href="https://buymeacoffee.com/peasantsmith"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Project Maya 是免费的。如果它在你的电脑上运行良好，欢迎赞助，用于购买测试硬件和支持开发。</sub></p>

这么大的模型通常需要数据中心。Maya 让你用手头现有的硬件运行其中三个：
**GLM-5.3-Flash**、**DeepSeek-V4.1-Flash-1017** 和 **MiniMax-M3.1-Flash**。每个模型都有专门为它打造的引擎。引擎把你的对话最常用的专家 (experts) 放在 GPU 上，接下来会用到的放在内存里，其余的放在 NVMe 固态硬盘上，并随着聊天不断调整。它会测量你的 GPU、CPU、内存和固态硬盘并自动调优。数据不会离开你的机器。

- **快：** 两块 RTX 5090 加一块 RTX 4090 最高 149 tokens/s，两块 H100 为 124，四块 RTX 4090 为 118，一块 RTX PRO 6000 为 81，单块 RTX 5090 也能达到 30 以上。[真实用户的实测数据](#how-fast-is-it)。
- **接近原版：** Maya 自己的 GLM-5.3-Flash 量化版本保留了 FP8 模型零样本准确率的 97.7-99.2%；DeepSeek V4.1 · Maya-S 与原版选择相同下一个 token 的概率为 92.6%，MiniMax M3.1 · Maya-S 为 87.6% (各类别平均值)。
- **接入你的工具：** 兼容 OpenAI 和 Anthropic 的 API，支持工具调用，可直接用于编程智能体。
- **同时处理多个请求：** GLM-5.3-Flash、DeepSeek-V4.1-Flash 和 MiniMax-M3.1-Flash 最多可同时回答 8 个请求，适合运行多个会话的智能体，或多人共用一台电脑。
- **开箱即用：** 浏览器里的聊天和实时监控、所有模型都支持图片、思考级别、可下载 / 加载 / 删除模型的 Models 页面、Turbo 按钮，以及一条命令完成的安装。

## <a name="quick-start"></a>快速开始

```sh
git clone https://github.com/mw00/project-maya.git && cd project-maya
./setup.sh
```

安装程序会询问你想用哪个模型，下载它，一次性构建所有内容，并**启动模型**。然后打开 `http://127.0.0.1:8080`。Windows 请用 `START-MAYA.bat` 代替 `./setup.sh`。详见[安装](#install)。

## <a name="the-models"></a><a name="deepseek-v41-flash-1017-new"></a><a name="minimax-m31-flash-new"></a>选择你的模型

每个 Maya 量化版本都以模型名加档位命名：XS、S、M 或 L，按专家保留的位数划分。安装程序会为你的电脑推荐一个；想自己选，请给出它的安装 ID：`./setup.sh --model glm-maya-s24`。

| 模型 | Maya 量化 · 安装 ID | 大小 | 适用场景 |
| --- | --- | ---: | --- |
| **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**<br><sub>zai-org · MIT · 321 B，激活 18 B · 最高 1M 上下文</sub> | **Maya-S** (默认) · `glm-maya-s` | 96.5 GB | 内存池 (内存与显存之和) 较小的电脑 |
| | Maya-S24 · `glm-maya-s24` | 94.7 GB | 24 GB 及以下的显卡：显卡多出约 1.5 GB 用于专家，解码 +14% |
| | Maya-XS · `glm-maya-xs` | 73.4 GB | 64 GB 内存加 24 GB 显卡的电脑，128K 上下文 |
| | Maya-M · `glm-maya-m` | 116 GB | 更大的内存池；针对工具调用和前端代码做了校准 |
| | Maya-M-Derisked · `glm-maya-m-derisked` | 116 GB | 经 [Blackfrost_AI](https://x.com/Blackfrost_AI) 修改权重、减少一刀切拒绝的 Maya-M；实验性 |
| | Maya-L · `glm-maya-l` | 156.3 GB | 最大的内存池；最接近原版 |
| **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**<br><sub>DeepSeek · MIT · 748 B</sub> | **Maya-S** (默认) · `deepseek-maya-s` | 340 GB | 任意内存大小：显存和内存放不下的部分从固态硬盘读取 |
| | Maya-XS (v2) · `deepseek-maya-xs` | 277 GB | 整个模型装进 128 GB 电脑的内存 (Engram 表留在固态硬盘上)；v2 比 v1 更接近原版约 12%，写作速度快约 20% - 再次运行安装程序会提示更新 |
| **[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)**<br><sub>MiniMax · [MiniMax Community License](LICENSE.minimax-m31) · 428 B，激活 23 B · 最高 1M 上下文</sub> | Maya-XS · `minimax-maya-xs` | 93 GB | 整个模型装进 128 GB 电脑的内存 |
| | Maya-S · `minimax-maya-s` | 122 GB | 更接近原版 |

它们都能在 V100 / RTX 20 及更新的 NVIDIA 显卡上运行，也支持 AMD 和 Windows (实验性；见[你需要什么](#what-you-need))。DeepSeek 的大小包含其 Engram 表。安装程序还提供 unsloth 的 GGUF 和 LibertAIDAI 的 DeepSeek EXL3 量化、orcarouter 的 abliterated DeepSeek、Mia-AiLab 的 MiniMax EXL3 量化、turboderp 的 GLM EXL3 量化，以及以 Q5_K_M GGUF 提供、支持图片的 Qwen-3.8-Flash-Next (CYBER-FROST) ([其设置](docs/SETTINGS.md#qwen-38-flash-next-cyber-frost))。**每个模型的文件、草稿模型、图片和选项：[docs/MODELS.md](docs/MODELS.md)。**

## <a name="quality"></a>质量

Maya 的量化版本由各模型公开发布的权重制作，并与原版对比测量：

- **GLM-5.3 · Maya 量化**保留了 FP8 模型零样本准确率的 97.7-99.2%，下一个 token 与原版一致的比例为 83-90% (Maya-L：99.2%，90%)。制作与测量方法：[MAYA-S](bench/results/MAYA-S.md)、[MAYA-S24](bench/results/MAYA-S24.md)、[MAYA-M](bench/results/MAYA-M.md)、[MAYA-L](bench/results/MAYA-L.md)。
- **DeepSeek V4.1 · Maya-S** 与原版选择相同下一个 token 的概率为 92.6%。
- **MiniMax M3.1 · Maya-S** 为 87.6% (各类别平均值)。

更多内容见各模型在 [Hugging Face](https://huggingface.co/peasantsmith) 上的模型卡。

## <a name="how-fast-is-it"></a>有多快？

以下是用户在真实机器上的测量结果，大多使用 `./maya.sh --bench` (解码：三次贪心解码的 256 token 回答；预填充：8K token 的提示，除非该行另有说明)。想分享你的结果：先运行 `--bench`，再运行 `--report`，然后发到 [GitHub issue](https://github.com/mw00/project-maya/issues)。

<details><summary><b>GLM-5.3-Flash 速度基准</b></summary>

| 机器 | 模型 | 解码 (写出回答) | 预填充 (读取提示) | 提交者 |
| --- | --- | ---: | ---: | --- |
| **2x RTX 5090 32 GB + RTX 4090 48 GB** (PCIe 5.0 x8, 5.0 x8, 4.0 x4), Core i7-13700K, 188 GB RAM | Maya-S / Maya-M-Derisked (DFlash2), 32K context | **148.6 / 123.4 tokens/s** | 2605 / 2341 tokens/s | @prismaillya17 (v1.1.26, #498) |
| **2x H100 80 GB** (PCIe 5.0), Xeon Platinum 8488C, 504 GB RAM | Maya-L / Maya-M-Derisked, 1M context | **122.4 / 123.7 tokens/s** | 2099 / 1986 tokens/s | @Molyleaf (v1.1.11, #258, #259) |
| **4x RTX 4090 24 GB** (PCIe 4.0 x16), Xeon w7-3445, 62 GB RAM | Maya-S24 | **118 tokens/s** | 4129 tokens/s | @0xPreDa (v1.0.31, #89) |
| **8x RTX 3090 24 GB** (PCIe 4.0, six x16 and two x8), EPYC 7402P, 126 GB RAM | EXL3-4.05bpw (DFlash2), 1M context | **117.4 tokens/s** | 1470 tokens/s | @MarkMakers (v1.1.15, #414) |
| **4x RTX 3090 24 GB** | Maya-S24 / Maya-M (DFlash2), 256K context | **100.8 / 86.5 tokens/s** | 979 / 907 tokens/s | @Whamp (v1.1.19, #349) |
| **2x A100 80 GB** (PCIe 4.0 x16), EPYC 7742 (a VM, 25 threads), 161 GB RAM | Maya-M, 128K context | **91.9 tokens/s** | - | @driona-will-fluff (v1.1.19, #326) |
| **RTX PRO 6000 Blackwell 96 GB**, Ryzen 7 9800X3D, 61 GB RAM | Maya-M, 256K context | **81.0 tokens/s** | 3102 tokens/s | @yundddd (v1.1.7, #172) |
| **4x Tesla V100 32 GB** (PCIe 3.0 x16), 2x Xeon E5-2630 v4, 126-504 GB RAM, SATA SSD | Maya-S / Maya-M / Maya-L, 32K context | **69.7 / 67.2 / 43.3 tokens/s** | 730 / 713 / 627 tokens/s | @HansStrupat (v1.1.36, #327, #340, #653) |
| **4x Tesla V100-SXM2 32 GB** (NVLink 2.0) | Maya-S, 32K context | 68.4 tokens/s | 549 tokens/s | @jeremiahlangner (v1.1.18, #333) |
| **1x H100 80 GB** (PCIe 5.0 x16), Xeon Platinum 8488C, 504 GB RAM | Maya-M-Derisked, 1M context | 61.4 tokens/s | 2014 tokens/s | @Molyleaf (v1.1.15, #260) |
| **2x NVIDIA CMP 170HX 64 GB** (PCIe Gen2 x4 each), 2x Xeon E5-2690 v4, 91 GB RAM | Maya-M, every expert in VRAM | 61.3 tokens/s | 514 tokens/s | @ZackO2o (v1.0.6, #22) |
| **2x RTX A6000 48 GB** (PCIe 4.0 x8), Core i9-13900K, 128 GB RAM, SATA SSD | Maya-S / Maya-M / Maya-L, 131K-token prompts | 60.7 / 50.9 / 32.1 tokens/s | 1726 / 1629 / 1298 tokens/s (131K) | @everson (v1.1.20, #371) |
| **2x RX 7900 XTX 24 GB** (PCIe 4.0 x4 each) + **Radeon 8060S** helper (Ryzen AI Max+ 395, 128 GB) | Maya-M, up to 512K context | 56.0 tokens/s (50.1 after a 456K-token prompt) | 907 tokens/s (114K) | @pepuscz (v1.1.25 + #454-#457, #458) |
| **2x RTX 5880 Ada 45 GB** (PCIe 4.0 x16), Xeon Gold 5420+, 393 GB RAM | Maya-L, 256K context | 54.5 tokens/s | 1401 tokens/s | @vjsyong (v1.1.4, #149) |
| **2x NVIDIA CMP 170HX 40 GB** (PCIe Gen2 x16 each), Xeon w3-2423, 128 GB DDR5 | Maya-S, 128K context | 54.3 tokens/s | 831 tokens/s | @nvsofts (v1.1.15, #263) |
| **2x RTX 3090 24 GB** (PCIe 4.0 x16, NVLink, power-capped at 252 W), Threadripper PRO 3975WX, 512 GB RAM | Maya-M / Maya-L, 128K context | 54.1 / 40.3 tokens/s | 936 / 850 tokens/s | @ksim63-gif (v1.1.23, #404) |
| **2x RTX 3090 24 GB** (PCIe 4.0 x16, power-capped at 225 W), Threadripper 3970X, 128 GB RAM | Maya-S24 | 53.0 tokens/s | 885 tokens/s | @briepala (v1.1.2, #135) |
| **3x Tesla V100 32 GB** (PCIe 3.0 x16), 2x Xeon E5-2696 v3, 128 GB DDR3L, Windows 10 | Maya-S, 128K context | 52.9 tokens/s | 616 tokens/s | @noahark (v1.1.20, #346) |
| **RTX PRO 6000 Blackwell 96 GB** (PCIe 5.0 x16), Ryzen 9 9950X3D, 253 GB RAM, Windows 11 | Maya-L, 512K context | 48.8 tokens/s | 3810 tokens/s | @Andrey0710 (v1.1.43, #730) |
| **4x Tesla V100 32 GB** (PCIe 3.0 x16), 24 CPU cores, a 100 GiB RAM limit | Maya-L, 32K context | 48.3 tokens/s (after `--calibrate`) | 499 tokens/s | @areslp (v1.1.3, [#130](https://github.com/mw00/project-maya/issues/130)) |
| **2x Radeon AI PRO R9700 32 GB** (gfx1201, PCIe 4.0 x16 each, 240 W), EPYC 7313, 256 GB RAM | Maya-M-Derisked, 256K context | 44.4 tokens/s | 703 tokens/s | @knoodrake (v1.1.28, #509) |
| **4x RTX 3090 24 GB** (PCIe 4.0 x8 behind one switch), Xeon Platinum 8260M, 376 GB RAM | Maya-L, 1M context | 44.2 tokens/s (tuned) | 698 tokens/s | @foureight84 (v1.1.17, #302) |
| **2x Radeon AI PRO R9700 32 GB** (gfx1201, PCIe 4.0 x8 each), Ryzen 5 3600, 64 GB RAM | Maya-S24 / Maya-M-Derisked, 256K context | 44.1 / 24.9 tokens/s | 704 / 617 tokens/s | @gegnep (v1.1.20, #395) |
| **2x RTX 3090 24 GB** (PCIe 4.0 x16, power-capped at 250 W), EPYC 7452, 256 GB RAM | Maya-M / Maya-L, 128K context | 44.0 / 30.2 tokens/s | 860 / 801 tokens/s | @orochifj (v1.1.4, #165) |
| **2x RTX 4090 24 GB** (PCIe 3.0 + 4.0 x16), 2x Xeon Silver 4309Y, 188 GB RAM | Maya-S24 | 40.5 tokens/s | 1271 tokens/s | @fuad00 (v1.1.0, #122) |
| **RTX PRO 6000 Blackwell 96 GB** (PCIe 4.0 x16), Ryzen 5 5600, 92 GB DDR4 | Maya-L, 1M context | 39.6 tokens/s | 2708 tokens/s (3246 warm) | @QueTensors (v1.1.28, #532) |
| **4x Tesla T10 15 GB** (PCIe 3.0 x16), Xeon E5-2680 v4, 63 GB RAM | Maya-S24, 100K context | 38.4 tokens/s | 393 tokens/s | @zeallag (v1.1.16, #291) |
| **2x Tesla V100 32 GB** (PCIe 3.0 x8 each), Ryzen 9 7945HX3D, 96 GB RAM | Maya-M / Maya-L | 33.8 / 13.3 tokens/s | 490 / 372 tokens/s | @nkhaooat (v1.1.11, #208) |
| **RTX 3090 24 GB + RTX 3060 12 GB** (PCIe 4.0 x16), EPYC 7702P (8-channel DDR4), 141 GB RAM (a VM) | Maya-M, 128K context | 33.7 tokens/s | 682 tokens/s | @erikash (v1.1.4, #157) |
| **AMD Radeon 8065S** (Ryzen AI Max+ PRO 495, 192 GB, 160 GB of it for the GPU), Windows 11 | Maya-S, 32K context | 31.5 tokens/s | 342 tokens/s | @tobitege (v1.1.37, #663) |
| **RTX 5090 32 GB**, Ryzen 9 7950X, 96 GB DDR5 | Maya-M | 30+ tokens/s | ~2000 tokens/s | @siganos (#43) |
| **2x RTX 3090 24 GB** (PCIe 4.0 x16, NVLink), Threadripper PRO 3955WX, 247 GB RAM | EXL3-3.05bpw, 128K context | 29.2 tokens/s | - | @catonooka (v1.1.9, #187) |
| **2x Tesla P40 24 GB** (Pascal, PCIe 3.0 x16), EPYC 7302P (a VM), 184 GB RAM | Maya-S24 / Maya-L | 28.0 / 16.9 tokens/s | 202 / 194 tokens/s | @Jw88888 (v1.1.11 + #214, #212) |
| **RX 7900 XTX 24 GB + Radeon 8060S** helper (Ryzen AI Max+ 395, 128 GB) | Maya-S24, 114K-token prompts | 27.5 tokens/s | 836 tokens/s (114K) | @pepuscz (v1.1.17, #311) |
| **1x RTX 3090 24 GB** (PCIe 4.0 x16), EPYC 7702P (8-channel DDR4), 141 GB RAM (a VM) | Maya-M, 128K context | 27.0 tokens/s | 1069 tokens/s | @erikash (v1.1.4, #157) |
| **RTX 4080 SUPER 32 GB** (a 32 GB modded card), Threadripper 3970X, 157 GB RAM, SATA SSD | Maya-S24, 262K context | 26.7 tokens/s | 1552 tokens/s | @Micro-ATP (v1.1.11, #200) |
| **3x Tesla V100 16 GB** (PCIe 3.0 x16), 2x Xeon E5-2690 v4, 62 GB RAM | Maya-S, 128K context | 26.6 tokens/s | 497 tokens/s | @saatvik-viniak (v1.1.11, #211) |
| **4x RTX 3080 20 GB**, 64 GB DDR4 | Maya-M, 512K context | 26.3 tokens/s (a day of use) | - | @CoolJohn-lab (#104) |
| **1x RTX 3090 24 GB** (power-capped at 225 W), Threadripper 3970X, 128 GB RAM | Maya-S24 | 26.0 tokens/s | 913 tokens/s | @briepala (v1.1.2, #135) |
| **1x L40S 48 GB** (vGPU), Xeon Platinum 8580 (a VM, 32 vCPUs), 504 GB RAM | UD-Q4_K_XL / UD-Q5_K_XL / Q8_0, 512K context | 26.0 / 22.3 / 17.8 tokens/s | 1987 / 1877 / 1750 tokens/s (185K) | @talisp (v1.1.32, #577) |
| **RTX 4090 24 GB** (PCIe 4.0 x16), 2x Xeon Silver 4310, 251 GB RAM | Maya-S24, 32K context | 25.7 tokens/s | 1963 tokens/s | @oscarlius (v1.1.24, #420) |
| **RTX PRO 4000 Blackwell 24 GB**, Core i9-13900K, 128 GB DDR5 | Maya-S24 | 24.7 tokens/s | 1146 tokens/s | @morrisfamily (v1.1.7, #174) |
| **2x Tesla V100 16 GB** (PCIe 3.0 x16), 2x Xeon E5-2696 v4, 128 GB DDR3-1866 | Maya-M, 256K context | 24.6 tokens/s (normal sampling) | 1092 tokens/s (12K) | @c200762620-max (v1.1.49, #790) |
| **RTX 5090 32 GB** (PCIe 5.0 x16), Core i9-13900K, 64 GB RAM, Windows 11 | Maya-S, 128K context | 23.0 tokens/s | 2173 tokens/s | @gianbing (v1.1.3, #139) |
| **4x Tesla T4 16 GB**, Xeon E5-2660 v2, 220 GB RAM | Maya-S | 22.6 tokens/s | 416 tokens/s | @ksanislo (v1.0.16, #36) |
| **2x RTX 2080 Ti 22 GB** (modded, NVLink, PCIe 3.0 x16, 250 W), Xeon E5-2686 v4, 247 GB RAM, SATA SSD | Maya-M, 1M context | 22.0 tokens/s | 533 tokens/s | @blackxwhite88 (v1.1.36, #675) |
| **Radeon AI PRO R9700 32 GB** (gfx1201), Core i5-14400F, 128 GB RAM | Maya-S | 21.4 tokens/s | - | @potatopote-debug (v1.1.12, #213) |
| **1x NVIDIA CMP 170HX 64 GB** (PCIe Gen2 x16), Core i7-9700K, 32 GB RAM | Maya-S24, 32K context | 20.6 tokens/s | 797 tokens/s | @Kanorin-chan (v1.1.8, #281) |
| **2x Quadro RTX 5000 16 GB** (Turing, PCIe 3.0 x16 each, 200 W), Xeon W-2133, 94 GB RAM | Maya-S, 128K context | 18.8 tokens/s | 494 tokens/s | @arpleasa (v1.1.39, #707) |
| **AMD Radeon 8060S** (Ryzen AI Max+ 395, 128 GB, 96 GB of it for the GPU), Windows 11 | Maya-S | 18.3 tokens/s | 320 tokens/s | @LIN-LAB-AI (v1.0.30, #102) |
| **RTX 3090 Ti 24 GB** (PCIe 3.0 x16, a VM), EPYC 7302P, 240 GB RAM | Maya-L, 32K context | 18.2 tokens/s (tuned, #177) | 791 tokens/s | @Jw88888 (v1.1.7, #177) |
| **3x Tesla P100 16 GB** (Pascal; PCIe 3.0 x8, 3.0 x8, 2.0 x4), Ryzen 5 2600, 64 GB RAM | Maya-S24, 64K context | 17.9 tokens/s | 105 tokens/s | @maxlawrie (v1.1.28 + #479, #504) |
| **RTX 4090 24 GB**, Ryzen 7 9700X, 64 GB RAM | Maya-S24, 32K context | 16.6 tokens/s | 1320 tokens/s | @MagikEh (#274) |
| **RTX 5070 12 GB** (PCIe 4.0 x16), EPYC 7302 (4-channel DDR4), 126 GB RAM | Maya-S24, 64K context | 16.6 tokens/s | 323 tokens/s | @hugou74130 (v1.1.45, #755) |
| **NVIDIA DGX Spark** (GB10, 121 GB unified memory) | Maya-S, 128K context | 16.5 tokens/s | ~730 tokens/s (6K prompt) | @ericoh88 (v1.1.15, #282) |
| **RTX 4090 24 GB**, Ryzen 9 9950X3D, 192 GB DDR5, Windows 11 | Maya-L, 128K context | 16.4 tokens/s | 1371 tokens/s | @npc97 (v1.0.24, #63) |
| **RTX 5090 Laptop 24 GB**, Core Ultra 9 275HX, 64 GB RAM, Windows 11 (a laptop) | Maya-S24 | 14.5 tokens/s | 994 tokens/s | @klvnblst (v1.0.30, #67) |
| **2x RTX 4060 Ti 16 GB** (PCIe 3.0 x8 each), Core i7-10700K, 128 GB DDR4, SATA SSD | UD-IQ3_XXS, 128K context | 13.0 tokens/s | 385 tokens/s | @oisee (v1.1.42, #723) |

</details>

<details><summary><b>DeepSeek-V4.1-Flash 速度基准</b></summary>

| 机器 | 模型 | 解码 (写出回答) | 预填充 (读取提示) | 提交者 |
| --- | --- | ---: | ---: | --- |
| **8x RTX 5090 32 GB**, Xeon Gold 6530, 1 TB RAM | Maya-S, 512K context, `--parallel 4` | **184.2 tokens/s** | 2510 tokens/s | @etevrfed34-commits (v1.1.34, #585) |
| **10x RTX 3090 24 GB** (tensor parallel 2, PCIe 4.0 x4 each, 200 W), Xeon (24 threads), 64 GB RAM | EXL3 2.9 bpw, 1M context | **97.0 tokens/s** | 2390 tokens/s (128K) | @DoktorMincs (v1.1.28, #494) |
| **RTX 6000D 85 GB** (PCIe 5.0 x16), Threadripper PRO 9965WX, 251 GB RAM | EXL3 2.9 bpw, 512K context | **89.3 tokens/s** | 1687 tokens/s (2836 at 33K) | @Ross-Yang-94 (v1.1.25, #423) |
| **2x A100-SXM4 80 GB**, Xeon Platinum 8360Y, 1 TB RAM | Maya-S, 1M context | **87.4 tokens/s** | 2084 tokens/s | @SunYong0821 (v1.1.34, #445, #594) |
| **RTX PRO 6000 Blackwell 96 GB** (PCIe 5.0 x16), Ryzen 9 9950X3D, 253 GB RAM, Windows 11 | Maya-S, 256K context | 72.3 tokens/s | 1757 tokens/s | @Andrey0710 (v1.1.43, #737) |
| **2x RTX PRO 5000 Blackwell 72 GB**, Ryzen 9 9950X3D, 249 GB DDR5 | Maya-S, 512K context | 68-71 tokens/s (prose), up to 162 (code), tuned | ~3700 tokens/s (42K) | @realsquall1980-art (v1.1.28, #424) |
| **1x A100-SXM4 80 GB**, Xeon Platinum 8360Y, 1 TB RAM | Maya-S, 1M context | 60.2 tokens/s | 241 tokens/s | @SunYong0821 (v1.1.34, #594) |
| **4x AMD Instinct MI100 32 GB** (gfx908, PCIe 4.0 x16 each), EPYC 7402, 252 GB RAM | Maya-XS, 32K context | 51.8 tokens/s | 172 tokens/s | @nsaxploit (v1.1.22, #396) |
| **4x RTX 3090 24 GB** (PCIe 3.0, SM clocks capped at 1500 MHz), Threadripper 1950X, 126 GB RAM | Maya-S, 256K context | 50-53 tokens/s (code), 40-43 (prose) | 947 tokens/s (10K), 1954 (90K) | @mgabor3141 (v1.1.26, #448) |
| **2x NVIDIA A40 48 GB** (PCIe 4.0 x16), 2x Xeon Gold 6330, 256 GB RAM, Windows Server 2022 | Maya-S, 256K context | 48.9 tokens/s (tuned) | ~660 tokens/s (3K) | @yjyz1011 (v1.1.28, #508) |
| **RTX 4090 24 GB** (PCIe 4.0 x8), Ryzen 9 9950X3D, 192 GB DDR5 | Maya-XS, 64K context | 44.1 tokens/s | 539 tokens/s | @naeq (v1.1.27, #503) |
| **4x Tesla V100 32 GB** (PCIe 3.0 x16), 2x Xeon E5-2630 v4, 504 GB RAM | Maya-S / UD-Q4_K_XL / UD-Q8_K_XL, 64K context | 44.0 / 28.6 / 29.7 tokens/s | 592 / 236 / 233 tokens/s | @HansStrupat (v1.1.49, #785) |
| **1x L40S 48 GB** (vGPU), Xeon Platinum 8580 (a VM, 32 vCPUs), 504 GB RAM | UD-Q8_K_XL, 256K context | 43.2 tokens/s | 1862 tokens/s (174K) | @talisp (v1.1.32, #577) |
| **AMD Radeon 8065S** (Ryzen AI Max+ PRO 495, 192 GB, 160 GB of it for the GPU), Windows 11 | orcarouter abliterated Mix-128G, 256K context, DSpark | 43.2 tokens/s | 453 tokens/s | @tobitege (v1.1.37, #663) |
| **2x RX 7900 XTX 24 GB** (tensor parallel 2, PCIe 4.0 x4 each) + **Radeon 8060S** helper (Ryzen AI Max+ 395, 128 GB), the model on a USB4 NVMe | Maya-S, 512K context | 41.3 tokens/s (46.1 after a 130K-token prompt, cached) | 1291 tokens/s (130K; 482 at 8K) | @pepuscz (v1.1.28 + #511-#519, #510) |
| **4x RTX 3090 24 GB** (PCIe 4.0 x16), Threadripper PRO 3975WX, 502 GB RAM | UD-Q8_K_XL, 256K context | 37.1 tokens/s | 596 tokens/s | @bandito1 (v1.1.50, #766) |
| **4x RX 7900 XT 20 GB** (gfx1100, `--tp 4`, tuned), 2x Xeon E5-2699C v4 (a Dell T7910), 1 TB DDR4-2400 | UD-Q4_K_XL | 35.0 tokens/s | - | @Cass67 (v1.1.32, #566) |
| **2x RTX 3090 24 GB** (PCIe 4.0 x8, 270 W), Core Ultra 7 265K, 192 GB DDR5 | Maya-S, 1M context, tuned | 33.3 tokens/s | 467 tokens/s | @Jellepepe (v1.1.35, #614) |
| **3x Tesla V100 32 GB** (PCIe 3.0 x16), 2x Xeon E5-2696 v3, 128 GB DDR3L, Windows 10 | Maya-S, 64K context | 23-26 tokens/s (long answers) | 377 tokens/s | @noahark (v1.1.26, #450) |
| **2x RTX 2080 Ti 22 GB** (modded, PCIe 3.0 x16, power-capped at 250 W), 2x Xeon E5-2696 v4, 94 GB RAM | Maya-XS, 128K context | 23.1 tokens/s | 317 tokens/s | @NorwinYu (v1.1.36, #659) |
| **2x RTX 4090 24 GB** (PCIe 3.0 + 4.0 x16), 2x Xeon Silver 4309Y, 188 GB RAM | Maya-XS, 64K context | 21.5 tokens/s | 573 tokens/s | @fuad00 (v1.1.22, #393) |
| **RTX 4000 Ada 20 GB**, Xeon w7-3455, 255 GB DDR5, Windows 11 | Maya-XS, 256K context | 21.1 tokens/s | 518 tokens/s (20K) | @pietroMastro92 (v1.1.29, #534) |
| **2x Tesla V100 32 GB** (PCIe 3.0 x8 each), Ryzen 9 5900XT, 128 GB DDR4-3200 | Maya-S, 64K context | 20.8 tokens/s | 344 tokens/s | @pirosap (v1.1.34, #586) |
| **RX 7900 XTX 24 GB** (gfx1100, PCIe 4.0 x16), Core i7-14700KF, 63 GB RAM, ROCm 10.2 | Maya-XS, 64K context, DSpark | 18.4 tokens/s | 395 tokens/s (146 at 2K) | @Cayaf89 (v1.1.46, [#757](https://github.com/mw00/project-maya/issues/757)) |

</details>

<details><summary><b>MiniMax-M3.1-Flash 速度基准</b></summary>

| 机器 | 模型 | 解码 (写出回答) | 预填充 (读取提示) | 提交者 |
| --- | --- | ---: | ---: | --- |
| **RTX PRO 6000 Blackwell 96 GB** (PCIe 5.0 x16), Ryzen 9 9950X3D, 253 GB RAM, Windows 11 | Maya-S, 256K context, DSpark | **48.6 tokens/s** | 757 tokens/s | @Andrey0710 (v1.1.43, #738) |
| **RTX 3090 24 GB + 8x RTX 5060 Ti 16 GB** (PCIe 3.0 x8 each), 183 GB RAM | EXL3 3.5 bpw, tensor parallel 4 + DSpark, 8K context | 38.2 tokens/s (code), 32.4 (prose) | - | @needmorevram (#477) |
| **1x L40S 48 GB** (vGPU), Xeon Platinum 8580 (a VM, 32 vCPUs), 504 GB RAM | Maya-S / EXL3 3.55 bpw, 256K context | 23.4 / 16.7 tokens/s | 332 / 273 tokens/s (158K) | @talisp (v1.1.32, #577) |
| **2x RTX 3090 24 GB** (power-capped at 230 W), EPYC 7543 (a VM, 32 vCPUs), 224 GB RAM | Maya-S, 200K context, DSpark | 22.0 tokens/s (code), 18.4 (prose) | 163 tokens/s (10K) | @noonghunna (v1.1.30, #549) |
| **AMD Radeon 8065S** (Ryzen AI Max+ PRO 495, 192 GB, 160 GB of it for the GPU), Windows 11 | Maya-S, 128K context, DSpark | 21.0 tokens/s | 139 tokens/s | @tobitege (v1.1.37, #663) |

</details>

一个 token 约等于 ¾ 个英文单词。随着对话变长，速度保持不变；启动后的最初几次回答最慢，因为缓存还在根据你的使用内容逐步填充。

## <a name="using-it"></a>使用方式

- **在浏览器里：** 聊天 (保存在你的浏览器中，支持搜索、置顶和文件夹，每个聊天可单独设置指令和思考级别，可编辑和重新生成，支持 HTML 预览)，模型与硬件的实时**监控** (细到每块 GPU 的显存都被什么占用)，可在设置中更改的上下文大小和端口，以及 **Models**：这台机器上已安装的所有模型、切换到另一个模型 (Maya 会用它重新启动)，以及每个模型的引擎设置 (上下文、GPU、草稿、图片、内存)，保存在它的 `maya-<model>.json` 中。
- **Models 页面：** 目录列出每个 Maya 模型及其大小。可以带进度条下载、加载、卸载正在运行的模型或删除模型，无需命令行。
- **Turbo：** 模型参数旁边的一个按钮，即可为该模型设置最快的配置 (贪心采样、不思考、跳过专家、最快的草稿方式)。这是用一点质量换速度；关闭后会精确恢复你原来的设置。
- **已节省：** 侧边栏的 Saved 一行会累计相同 token 若用模型自家 API (DeepSeek、Z.ai 或 MiniMax) 会收取的费用 - Maya 从缓存复用的提示按该 API 的缓存价格计算，DeepSeek 的高峰时段按高峰价格计算 - 按模型保存在 `token-savings.json` 中；价格可在 **Settings › Savings** 中修改。
- **你的应用和编程智能体：** OpenAI 兼容 API 位于 `http://127.0.0.1:8080/v1`，Anthropic 的位于 `http://127.0.0.1:8080/v1/messages` (Claude Code：`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`)。工具调用会随模型书写实时流式返回，请求时指定 (`"return_progress": true`) 还可流式返回长提示的读取进度。允许来自其他地址的网页的地址后，该网页也可以调用它 (`"cors_origins"`，[docs/SETUP.md](docs/SETUP.md#web-pages-on-another-address))。
- **Responses API：** `/v1/responses` 支持文本 / 图片对话、流式输出、函数工具调用、经模式校验的输出、可取消和重放的后台任务，以及可选的共享内存历史 (`store:true`、`previous_response_id`)。默认不保存响应。选项与限制：[docs/RESPONSES.md](docs/RESPONSES.md)。
- **模型 API：** `/api/tags`、`/api/ps`、`/api/show`、`/api/chat`、`/api/generate`、`/api/pull` 和 `/api/delete` 可让任何应用或脚本列出、加载、下载和删除模型。请求一个尚未加载的模型会切换到它，`keep_alive` 会在空闲时卸载它。`/models/load`、`/models/unload` 和 `/models/sse` 也默认开启 (`"model_management": false` 可关闭)。
- **思考：** Off、Low、High (默认) 或 Max，可在设置中或按请求指定；还有一个**思考预算**字段，设置模型在回答前最多可以思考多少个 token。
- **循环防护：** 默认开启 - 当智能体用同样的结果反复发出同一个工具调用时，会被引导走出循环。
- **图片：** 在聊天中附加，或通过任一 API 发送图片内容。每个模型都能读取图片 (各模型的安装细节：[docs/MODELS.md](docs/MODELS.md))。
- **从其他设备访问：** `./maya.sh --setup --host 0.0.0.0 --api-key <secret>`。务必设置密钥。
- **调优：** `./maya.sh --calibrate` 用 10-15 分钟让引擎适配你的硬件。全部设置：[docs/SETTINGS.md](docs/SETTINGS.md)。
- <a name="deepseek-engram-prompt-read-ahead"></a><a name="deepseek-conversations-kept"></a><a name="deepseek-loading-the-experts"></a>**各模型自己的选项** - GLM-5.3-Flash 的同时多请求 (`--parallel`)、长上下文和 DFlash2 草稿模型；DeepSeek 的会话保留、专家加载和 Engram 预读；MiniMax 的思考级别：[docs/MODELS.md](docs/MODELS.md)。

## <a name="what-you-need"></a>你需要什么

| | |
| --- | --- |
| **GPU** | NVIDIA，计算能力 7.0 或更高 (V100、RTX 20 及更新)：1 块，或最多 16 块共同承载模型。显存越大越快。Pascal (Tesla P40 / P100、GTX 10) 为实验性，尚未验证。**AMD (实验性)：** RX 7900 XT / XTX、Radeon AI PRO R9700 / RX 9070、Strix Halo 和 Gorgon Halo (Radeon 8060S / 8065S)；RX 6800 / 6900 和 Radeon PRO V620 / W6800 尚未验证；GLM-5.3-Flash 和 DeepSeek-V4.1-Flash-1017 在 AMD 上也能读取图片 - [docs/AMD_MAYA.md](docs/AMD_MAYA.md) |
| **内存** | 32 GB 即可运行。内存越多，能让更多专家就近存放，速度也越快 |
| **磁盘** | **一块高速 NVMe 固态硬盘**，空间足够放下模型 (GLM-5.3 · Maya-XS 约 75 GB、GLM-5.3 · Maya-S 约 100 GB、GLM-5.3 · Maya-L 约 160 GB、DeepSeek V4.1 · Maya-S 约 350 GB、其 unsloth Q8 GGUF 约 470 GB、MiniMax M3.1 · Maya-XS 约 95 GB、MiniMax-M3.1-Flash 的 EXL3 量化约 200 GB)，外加它的图片编码器 (1.1 GB)。不能用机械硬盘 |
| **系统** | Linux (x86-64；最好带 AVX2 的 CPU)、NVIDIA 驱动、CUDA 12.x (CUDA 13 适用于 RTX 20 及更新，不适用于 V100 或 Pascal)、g++、Python 3.10+。Windows 10/11 (实验性)：[docs/WINDOWS.md](docs/WINDOWS.md)。不支持 WSL2。AMD：ROCm 7 |

安装程序会检查以上所有项目，对缺少的内容打印出确切的命令，并且不会在系统范围内安装任何东西。

## <a name="install"></a>安装

运行[快速开始](#quick-start)中的两条命令并回答几个问题 (按 Enter 即采用推荐选项)。安装程序会在开始每次下载前先展示内容，一次性构建所有内容 (除下载外需要 20-40 分钟)，并**启动模型**。然后打开 `http://127.0.0.1:8080`。Ctrl+C 停止，`./setup.sh` 可立即重新启动。

- **Windows：** 用 `START-MAYA.bat` 代替 `./setup.sh` - 见 [docs/WINDOWS.md](docs/WINDOWS.md)。
- **AMD：** [docs/AMD_MAYA.md](docs/AMD_MAYA.md)。
- **Apple Silicon 的 Mac (实验性，仅 GLM-5.3-Flash)：** `./maya.sh --backend metal` (在 Mac 上，`./maya.sh` 会自动选择 Metal) - 见 [docs/METAL.md](docs/METAL.md)。已在 GitHub 的 Mac 上构建并做过单元测试，但我们还没有用真实模型测试过 (我们没有 Mac)：如果你运行了它，请在 issue 中告诉我们结果。
- **Intel Arc：** Maya 的引擎不支持；社区关于在 llama.cpp 的 SYCL 后端上运行 Maya-XS DeepSeek GGUF 的报告：[docs/INTEL_ARC.md](docs/INTEL_ARC.md)。
- **更新：** 页面的 **Settings › About and updates**，或 `git pull` 后运行 `./setup.sh`。不会重新下载模型。
- **用其他方式下载的模型文件** (下载管理器、另一台电脑、Hugging Face 缓存)：`./maya.sh --import-model FOLDER` 会按大小和校验和在 FOLDER 中找到 Maya 的文件 (不管叫什么名字)，并移到正确位置 (FOLDER 在另一个盘上时用 `--copy`；原文件保留)。不匹配的文件不会被动。
- **所有选项** (GPU、上下文、你已有的模型文件夹、远程访问等)：[docs/SETUP.md](docs/SETUP.md)。

> **启动需要几分钟：** 引擎会为专家层锁定大部分空闲内存 (除了约 6 GB 以外的全部)，所以 Maya 运行期间其他程序可用的内存很少。

## <a name="something-went-wrong"></a>出问题了？

在回答变慢或出现错误之后，立即运行 **`./maya.sh --report`** (Windows：`START-MAYA.bat --report`)，并把生成的 `maya-report.txt` 附到 [GitHub issue](https://github.com/mw00/project-maya/issues) 上。其中包含你的硬件、你的安装设置和引擎的速度日志；不会自动发送到任何地方。引擎自己的日志是 Maya 文件夹中的 `maya-<model>.log`。

- **构建中止** ("nvcc ... cannot build for these GPUs"、"unsupported GNU version"，或缺少 `cuda_profiler_api.h` / `nvtx3/nvToolsExt.h`)：每种情况的解决办法见 [docs/SETUP.md](docs/SETUP.md#if-the-build-fails)。
- **很慢，而且固态硬盘一直忙：** 空闲内存不够放专家。关闭一些程序或增加内存。
- **Windows：回答突然慢了好几倍：** 显卡太满，Windows 把显存挪到了内存。多留出一些：`START-MAYA.bat --setup --vram-reserve-mib 2048` ([专家池的大小](docs/SETTINGS.md#the-expert-pools-size))。
- **Windows 上的 Strix Halo / Gorgon Halo：其他程序内存不足，或安装程序提示页面文件问题：** 按安装程序打印的大小设置页面文件 ([docs/AMD_MAYA.md](docs/AMD_MAYA.md))。
- **下载中断了：** 再次运行 `./maya.sh`，它会从中断处继续。
- **端口 8080 被占用：** Maya 已经在运行，或者用 `--port 8081` 再启动一个。Settings > Port 可以永久把 Maya 移到另一个端口 (它会在新端口重新启动)。

## <a name="support"></a>支持

Project Maya 免费且开源。每个版本在发布前都会在我们自己的 NVIDIA 和 AMD 机器上测试 (Mac 支持目前只在 GitHub 的虚拟 Mac 上测试)，赞助用于购买这些测试硬件 (当前目标：50 位月度赞助者，用于购置一台真正的 Mac 和新的 GPU，让 Maya 在更多硬件上运行良好)、制作新的量化版本和模型，以及更快地修复问题。如果 Maya 帮你省下了云服务账单，请考虑拿出其中的一小部分来赞助 - 谢谢！

<p align="center"><a href="https://github.com/sponsors/mw00"><img src="https://img.shields.io/badge/Sponsor-mw00-ea4aaa?logo=githubsponsors&logoColor=white&style=for-the-badge" alt="Sponsor on GitHub" height="50"></a>
<a href="https://buymeacoffee.com/peasantsmith"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>

### 赞助者

感谢每一位赞助 Project Maya 的人。每月 $15 及以上的赞助者会列在这里 - [成为第一位](https://github.com/sponsors/mw00)。

欢迎提交速度报告、错误报告和 pull request：[CONTRIBUTING.md](CONTRIBUTING.md) 说明了最有帮助的方式。

## <a name="credits-and-license"></a>致谢与许可证

- **人员：** Project Maya 由 Peasant Smith ([@mw00](https://github.com/mw00)、[X](https://x.com/PeasantSmith)) 创建并主导，他拥有该项目、其名称和 Maya 模型。[@needmorevram](https://github.com/needmorevram) ([X](https://x.com/needmorevram)) 是共同维护者，贡献了主要功能：DFlash2 草稿、EXL3 支持、KV 流式传输、会话检查点、层拆分搜索、专家画像和 CPU 专家内核，以及为 `maya-ds4` 贡献的 3.5 位 EXL3 选项、4 / 8 块 GPU 上的张量并行和多线程的专家缓存交换 (#276)。同时感谢所有贡献代码和速度报告的人 ([贡献者](https://github.com/mw00/project-maya/graphs/contributors))。
- **基于 [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT License，Copyright (c) 2023-2026 The ggml authors)：量化格式、CPU 点积以及提示处理路径的 MMQ 内核，均由固定的提交构建 (`third_party/ggml/LICENSE`)。
- **EXL3 支持**遵循 [exllamav3](https://github.com/turboderp-org/exllamav3) 的格式、码本、CPU 专家内核和分块的张量核心解码 (MIT License，Copyright (c) 2025 Turboderp；`third_party/exllamav3/LICENSE`)。**DFlash2 草稿模型的前向计算**遵循 [beellama.cpp](https://github.com/Anbeeld/beellama.cpp) 的实现 (MIT License，Copyright (c) 2023-2026 The ggml authors、Anbeeld；`third_party/beellama.cpp/LICENSE`)。
- 模型：[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) 来自 zai-org (MIT)；Maya 的量化版本和图片编码器文件由 zai-org 公开发布的权重制作，沿用其许可证。[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) 来自 DeepSeek (MIT)；Maya 的量化版本、其 Engram 表和草稿模型由 DeepSeek 公开发布的权重制作，沿用其许可证；视觉塔来自 DeepSeek 的检查点，GGUF 选项来自 [unsloth 的 GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)，EXL3 选项来自 LibertAIDAI ([2.9](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) 和 [3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw))。可选的 DFlash2 草稿模型 ([GLM-5.3-Flash-DFlash2-GGUF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF)，作者 Anbeeld) 采用 CC BY-NC-ND 4.0，仅限非商业使用，只有在你选择时才会下载。
- 仪表盘的字体：Outfit (SIL Open Font License 1.1，`serve/web/fonts/OFL.txt`)；图标：[Lucide](https://lucide.dev) (ISC License，`serve/web/LICENSE.lucide.txt`)；UI 库：[Preact](https://preactjs.com) (MIT License) 和 [htm](https://github.com/developit/htm) (Apache License 2.0)，连同它们的声明一起打包在 `serve/web/preact.js` 中。
- Maya 以 [MIT License](LICENSE) 开源；[LICENSE](LICENSE) 和 `third_party/` 中的声明会随每份副本保留。

---

*本译文仅供参考。英文版 [README.md](README.md) 为准，且可能比本译文更新。*
