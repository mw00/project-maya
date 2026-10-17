# 调参与设置

<p align="center"><a href="SETTINGS.md">English</a> | 简体中文</p>

本页所有内容都是可选的：引擎会按它运行的电脑自行定尺寸。回到
[README](../README.zh-cn.md)。

引擎会自行定尺寸：把层拆分到交给它的 GPU 上（最多 16 块；超过两块时，每块 GPU 按空闲显存比例分层），
用每块 GPU 的空闲显存填充专家，按空闲内存定内存层的大小，并在启动时实测 CPU 与 PCIe 链路的速度，
决定 CPU 自己计算多少个内存层专家。

**为你的电脑调优**（`./maya.sh --calibrate`，安装结束时也会提供）：启动时的那个猜测只各测一次一个专家；
调优改为在一次引擎运行中实测真实输出速度（约 10-15 分钟，大部分时间花在加载模型上），尝试内存层专家在
CPU 与 PCIe 链路之间的几种分工、以及更少的 CPU 线程——线程多于内存能喂饱的量只会干等，而在混合架构 CPU
上，能效核会拖住其余部分——并且，当模型有一部分要从固态硬盘读取时，还尝试更小的磁盘读取块（最佳大小
取决于硬盘及其驱动）。每次测量都回答它此前没回答过的提示，因为聊天的文本对专家层来说是新的（反复问同样
几个问题会让显存恰好装着那几个答案用到的专家，报出的速度是任何聊天都达不到的），并且引擎不保存学到的
专家画像（`--expert-profile-save`）、在工作文件的副本上运行（`STRATA_GLM_USAGE`），所以调优的回答不会改变
下次启动先加载什么。一项设置只有比引擎自己的选择快 3% 以上才会保留。结果写入配置的 `"env"`
（`STRATA_GLM_PCIE_SHARE`、`STRATA_GLM_CPU_LANE`、`STRATA_GLM_READ_CHUNKS`），并写入这台电脑、这个模型、
这个上下文的 `~/.config/project-maya/calibration.json`，因此重新安装会保留它。先停掉正在运行的服务器：
调优需要用到 GPU。

以下设置会改变引擎的选择（用 `--env` 写进配置，或放进配置的 `"env"` 块）：

| 设置 | 默认值 | |
| --- | --- | --- |
| `STRATA_GLM_EXPERT_HELPER` | 未设置（关） | 实验性的原生 GGUF 专家助手：填可见的 GPU 序号（`1`，不是布尔值）。GPU 0 保留主干与 MTP；助手持有一个互补的专家缓存，带热度协调的替换与 GPU 量化/归约。配合 `STRATA_GLM_SPLIT=0` 使用，不要 `--layer-split`；取消该变量即禁用。UMA 助手与锁定的内存层共用地址；独立显卡助手用自己的显存。见 [AMD 助手设置与范围](AMD_MAYA.md#experimental-expert-helper) |
| `STRATA_GLM_RAM_HEADROOM_GB` | 6 | 定内存层大小时留给系统的空闲内存；它还多承担一块 GPU 的提示缓冲——在测量之后锁定。三块及以上 GPU 的拆分还会在其上预留其余 GPU 的份额，且每块后加载的 GPU 重新测量并至少留出一半。锁定层不能被换出，所以内核一开始换页的 Linux 桌面（多显示器、繁忙的会话）需要更大：一台 128 GB、三块 GPU、三个显示器的桌面在 16 时保持流畅，在 12 时开始换页（#77） |
| `STRATA_GLM_RAM_GB` | 按空闲内存 | 固定内存层大小（GB） |
| `STRATA_GLM_COMPLETE_POOL` | AMD：开 | 当 GPU 的专家预算装得下全部专家、每层 3 个备用槽和提示路径借用的槽位时（Windows 上 Ryzen AI Max APU 的内存——Radeon 8065S 装下 Maya-S 还剩约 60 GB），每层恰好保留那么多：启动加载全部专家，提示只借用它们之外的槽位（不搬出任何专家，也不再从固态硬盘读取），解码永不失手，CPU 通道无事可做（8065S、Maya-S：解码 +2.6-4%，8K token 提示后 +7%，答案相同）。`0` = 关；`1` = NVIDIA 也开（那里未测量）。Linux APU 保持其统一内存策略 |
| `STRATA_GLM_RESIDENT_SKIP` | AMD：开 | 当全部专家都在显存时，解码跳过每条路由的等待与取回启动——它们会无事可做（每层每 token 两次）；`0` = 照常启动；`1` = NVIDIA 也开 |
| `STRATA_GLM_FLUSH`、`STRATA_GLM_PREFILL_FLUSH` | 关、关 | 仅 Windows + AMD，实验性：HIP 运行时会把排队的内核扣住直到主机等待；`n` = 解码在第一个层入队后立刻提交、之后每 n 层一次；`1` = 提示路径每个子批与每层专家（`cudaStreamQuery`，不会等待）。在 Radeon 8065S 上每 4 层比不开慢约 4%；提示没有更快 |
| `STRATA_GLM_RAM_RESIDENT` | 关 | `1`（或 `--glm-ram-resident`）：内存层装下所有不在显存里的专家，且不向磁盘逐出任何东西。做不到则启动失败。适合内存装得下全部卡外内容的电脑（128 GB 内存的 4090 D：配 `PROMOTE_MIN=6` 解码快 1.9 倍）。`STRATA_GLM_RAM_SLACK` 是每个 MoE 层的额外槽位（默认 16） |
| `STRATA_GLM_SPLIT` | `auto`：2-4 块 GPU 上做拆分搜索（DFlash2 流水线窗口下更多卡也搜），更多卡按空闲显存 | 每块后续 GPU 的第一层（`24`，或九块卡的 `7,12,17,22,27,32,37,41`）；`0` = 单卡。配置的 `"layer_split"` 作用相同。`auto` 在加载前给每种放置方案估价：每块卡扣掉其层的权重与状态后能装专家的余量、其专家能承接的每层路由份额（你的使用计数；没有计数时用内置覆盖率曲线——一层的份额 f 的专家承接 1 - (1 - f)^3 的路由，EXL3 包用 ^1.4——`STRATA_SPLIT_COVER_B` 设指数；DFlash2 下的 EXL3-4.05bpw，一张 RTX 3090 加八张 5060 Ti：3090 拿 9 层而不是 7 层，63.8 / 32.6 / 64.4 -> 79.8 / 40.4 / 81.9 tok/s）、每层经该卡内存带宽的读取，以及每次失手按主机内存速度的代价——然后取每块卡都能启动的最快方案（在双卡拆分上，用模型的 MTP 模块起草时两块卡处理相邻 token，所以一个 token 花掉更慢那块卡的时间：RTX 3090 + 5060 Ti 用 NextN 模块起草，24 -> 26，两卡每 token 25.3 / 32.2 -> 28.1 / 29.8 ms，解码 +3-5%）。在 DFlash2 流水线窗口下（三块及以上 GPU）最慢的卡给每次起草定节奏：搜索可在任意卡数上运行（五块起，对最慢卡上的每个上限做按层的 DP），每层还计固定 0.4 ms（`STRATA_GLM_SPLIT_LAYER_MS`），失手按共享 CPU 池速度的一半计价（DFlash2 下的 Maya-L，一张 RTX 3090 加八张 5060 Ti：3090 拿 8 层而不是按显存份额给的 10 层，57.1 / 58.9 -> 61.6 / 63.1 tok/s）。`STRATA_GLM_SPLIT_LOG=1` 打印每种放置方案的预测 |
| `STRATA_GLM_PARALLEL_LOAD` | 开 | 拆分的各 GPU 同时加载，每块一个线程：它们按 GPU 顺序依次规划各自的内存层（每块得到逐个加载时会给它的层），然后同时锁定并预热（4 块 Tesla T4，Maya-L，冷页缓存：启动到首个 token 142 -> 96 s）。`0` = 一块接一块 |
| `STRATA_GLM_CPU_LANE` | 每个物理核一个线程（单卡，或 3 块及以上 GPU 拆分共享的一个池：最多一个 NUMA 节点的；两块卡：各一半；无 SMT 的核心：三分之二） | CPU 通道计算内存层专家的线程数；`0` = 关（由调优设定） |
| `STRATA_GLM_CPU_LANE<n>` | 同上 | 单独为 CUDA`<n>` 设置（`STRATA_GLM_CPU_LANE0=6`、`STRATA_GLM_CPU_LANE1=2`）：链路速度不同的两块卡需要不同数量——链路越慢，CPU 线程帮助越大。PCIe 3.0 x4 上的 RTX 4070 Ti SUPER + PCIe 4.0 x16 上的 RTX 5070 Ti，拆分 17：6 / 2 线程胜过默认的 4 / 4 |
| `STRATA_GLM_CPU_PIN` | Linux：两个及以上 NUMA 节点的主机上、两块卡时开。Windows：从不开 | 每块 GPU 各有 CPU 池的拆分（两块卡，或 `STRATA_GLM_CPU_SHARED=0`）：每块 GPU 的 CPU 通道绑在自己的 CPU 上，每个物理核一个线程（SMT 兄弟线程留空）——只要够用就整块 NUMA 节点（GPU 挂在不同节点时用它自己的节点），否则均分一个节点的物理核——这样一块卡的空闲 worker 不会在另一块卡正用的核心上空转（2 块 Xeon Gold 6152，RTX 3090 + 5060 Ti：解码 15.4 -> 19.8 tok/s）；单 NUMA 节点上保持关闭——那里各部分共享所有核心的池更快（一颗 14 核 Xeon 上两块 V100：不绑 25.8 tok/s，绑了 24.7）。`1` = 任何 Linux 主机上都绑；`0` = 不绑。仅 Linux：Windows 上无论怎么设引擎都不绑，且单卡在任何系统上都不绑，所以那里 `0` 不改变任何事 |
| `STRATA_GLM_CPU_SHARED` | 3 块及以上 GPU 时开 | 三块及以上 GPU 的拆分让各卡依次运行，所以一个 CPU 池轮流服务它们全部，按单卡大小定尺寸；`0` = 每块 GPU 一个池、核心在它们之间分配（A/B）；`1` = 两块卡也共用一个池。被单独指定线程数的卡（`STRATA_GLM_CPU_LANE<n>`）保留自己的池 |
| `STRATA_GLM_CPU_CONCURRENT` | 开 | 拆分共享的池：多块 GPU 的路由同时在池上跑，后面 GPU 的任务优先（DFlash2 流水线窗口下每块 GPU 持有比前一块更早的 token）；`0` = 一次只跑一条路由，其余整条等它（A/B）。两种方式答案相同 |
| `STRATA_GLM_CPU_CAL` | 未设置 | 一个文件（如 `~/.config/project-maya/cpu-cal.txt`）：保存启动时 CPU 通道的计时（CPU 上一个专家、经 PCIe 一个专家；每块 GPU 约 1.7 s），一行一块 GPU，之后的启动直接读取。新的引擎构建、另一个包、换卡、槽位或线程数会重新计时；解码连续五次发现某行偏差超过 15% 就丢弃该行，下次启动重新计时 |
| `STRATA_GLM_KDA_GRAPH` | 关 | 实验性：`1` 把单卡 KDA 解码层用 CUDA graph 重放；DSA、MTP/拆分、profiling、dump、预取与 lookahead 保持直接执行。v1.0.21 上两组 thinking-Max 32K/2048：30.73 -> 31.45 tok/s（+2.35%）。当前主干的完整原生/质量验证尚未完成；见[范围与检查](GLM-KDA-GRAPHS.md) |
| `STRATA_GLM_CPU_SPLIT` | 总共约 48 块 | 解码：CPU 通道每个线程在一个 token 里分到的内存层专家切成这么多块（1-64）依次认领，这样一个被磁盘读取拖住的线程不会拖住整个 token。线程少时每线程认领多块（6 线程：8——一块 V100、Maya-S：17.1 tok/s，单块时 14.3）；40 块及以上时每线程认领一块，对内存流式最有利 |
| `STRATA_GLM_PCIE_SHARE` | 启动时实测 | 一个 token 的内存层专家中经 PCIe 拷贝到 GPU 上跑、而不是在 CPU 上算的比例：`0` = CPU 全算，`1` = 一个都不（由调优设定；`STRATA_GLM_CPU_PLAN` = 按专家数的分工，数字对应 0..8） |
| `STRATA_GLM_CPU_RETIME` | 开 | 启动计时选定的分工（上面两个设置都没设时）会在解码中重新计时：在前几个 2,000 个 CPU 通道专家的窗口里（最多八个；第一个只预热），当通道自身每个专家的成本连续两个窗口都与制定分工时用的值相差 15% 以上，就按它们的均值重新规划分工，最多三次，每次打印一行 `CPU lane timed again in decode`。繁忙 CPU 上的启动（某次计时到 0.611 ms/专家，平时约 0.24）不再把它的分工用一辈子。`0` = 关 |
| `STRATA_GLM_DUAL_PREFILL` | 关 | `1`，两块能互访内存的 GPU（NVLink）：单块提示的拆分前后依次运行，等待的 GPU 和它的 PCIe 链路闲着。开启后，等待的 GPU 为工作半边各 staging 组的第二个专家做中转：经自己的链路从主机拷到自己的（空闲）staging 缓冲，再经 NVLink 点对点传过去。一组数据于是经两条链路到达。更长的提示（多块）在 GPU 间流水线化，按原方式中转。除非提示的 CPU 份额计划（它从拷贝时间学习）给 CPU 分了不同的专家，答案不变（`STRATA_GLM_SCORE_PREFILL=1`，3000 token 上下文：KL 0.000000）。Maya-M 在 2x RTX 3090（各 x8 + NVLink）、Ryzen 3950X、128 GB：提示 1K 283 -> 362 tokens/s（+28%）、4K 622 -> 772（+24%）、16K 1034 -> 1142（+10%）、64K 不变。在仅 PCIe 的点对点访问上，拷贝反正要穿过工作 GPU 的链路，所以那里不要指望收益。启动会检查点对点拷贝完好（各方向一个 4 MB 模式）：报告支持点对点却传错字节的机器会保持关闭并说明 |
| `STRATA_GLM_PREFETCH_N` | 0 | `1` 或 `2`（更大按 2 处理；引擎日志在启动时说明设置）：每层在计算时，把下一层预测出的、不在显存里的至多这么多专家（用本层输入上的 router）拷进该层的备用槽。`STRATA_GLM_PREFETCH_AT`：拷贝何时开始——`route`（默认：立刻，与本层自己的 PCIe 取回和 CPU 通道并行）、`fetch`（本层取回之后）或 `cpu`（CPU 通道之后）；`STRATA_GLM_PREFETCH_BLOCKS`：拷贝的 block 数（默认一半 SM）；`STRATA_GLM_PREFETCH_RANK`：只拷贝预测的前 n 个 rank（它最初的猜测几乎总是对的，第 5-8 名多半不对）。Maya-M 在 RTX 3090 + 3060、`RAM_RESIDENT=1`、21K 解码：关 19.64、`N=1` 17.0、`N=1 AT=fetch BLOCKS=4` 19.75、`+ RANK=2` 20.22 tokens/s（显存命中 64 -> 69%）；Maya-L 在那里（专家更大、链路更忙）：没有收益 |
| `STRATA_GLM_NUMA` | 2 个及以上 NUMA 节点时开 | `0` = 分配内存层时不按页在两个插槽的内存间铺开 |
| `STRATA_GLM_READ_CHUNKS` | 8 | 回答期间（以及提示期间）从固态硬盘读取的专家：每个专家的每个部分一次分这么多块读，1-16。最佳大小取决于硬盘、驱动和 CPU——Linux NVMe 上 8，Windows 笔记本 Intel RST RAID 上 4（解码 +15%）——模型有一部分在固态硬盘上时，调优会在 8 之外再测 4 和 2 |
| `STRATA_GLM_PREFILL_CHUNK` | 未设置 | 来自环境的 `--prefill N`（较老的设置；优先于配置的 `--prefill`） |
| `STRATA_PREFILL_LEND_PCT`、`STRATA_GLM_PREFILL_MB` | 90 或 85，未设置 | `--prefill auto`：提示可从每块卡专家池槽位中借用的比例：当至少 90% 的专家字节被锁定（在显存或锁定的内存层）时为 90，否则 85——或以 MB 计的固定预算 |
| `STRATA_GLM_PREFILL_TAIL_SKIP` | 开 | 没有加载 NextN/MTP 模块的单设备提示：更新完最后一层的全部缓存后，跳过该层的注意力输出投影与 FFN；`0` 恢复完整计算。拆分与 `GLM_CB_DIR` 接缝 dump 保持完整路径 |
| `STRATA_GLM_PREFILL_WINDOW` | 一块提示的 token 数 | 提示：一次留在 GPU 上的专家输出行——窗口填满过程中每组专家的行逐步累加进该层输出，而不是所有路由行等一次合并（每 token 约 40 KB 而不是约 185，所以一块能装约 2 倍的 token）；`0` = 全部行（旧布局） |
| `STRATA_GLM_PREFILL_FUSED` | 仅 gfx1151 开 | 带 MMQ 构建的 HIP：GLM 路由专家的融合 int8 WMMA gate/up、带 clip 的 SwiGLU 与 down（4096/2048 几何与被覆盖的 IQ 类型）。`0` 保留 MMQ 及其 scratch 布局。基于 v1.0.16 的原型在 Strix Halo 上测得提示速度 4K/8K/16K +13.2/+7.5/+4.8%；配 Radeon 8065S（Windows、Maya-S）上的 wmma2 注意力，3K / 8K token 提示 233 / 253 -> 271 / 278 tokens/s。其他 AMD 架构保留 MMQ。 |
| `STRATA_GLM_USAGE` | 关 | 在多次启动之间保存你的专家使用情况的文件（每个请求后写入；预热时据此先于专家画像加载你的专家）。关闭时，启动只按专家画像来（见下）；包旧的 `expert_usage.txt` 不再被读取，除非这里指向它 |
| `STRATA_GLM_SLOTS` | 4 | 在固态硬盘上保留的会话数，切回某个会话时不必重读它的提示；`0` = 关 |
| `STRATA_GLM_CKPTS`、`STRATA_GLM_CKPT_EVERY` | 4、16384 | 在模型中比下一轮续写位置更早的位置上，把会话的检查点放在内存里（GLM-5.3-Flash 每个约 150 MB）：系统提示末尾（第一个用户轮）、提示最后一轮开头、以及每 `_EVERY` 个新读取的提示 token（`0` = 不含这些），且此前至少有 512 个新 token。让会话提前分叉的提示——编辑或替换最后一条消息、另一个用相同系统提示与工具的聊天、改写了较早一轮的客户端——恢复它以最长前缀匹配的检查点，只读取其余部分，而不再从 token 0 全部重读（引擎日志里的 `glm checkpoints`）。最多 `STRATA_GLM_CKPTS` 个（至多 16；`0` = 关），最久未用的被替换；一次一个会话（不与 `--parallel` 同用），仅文本提示 |
| `STRATA_GLM_SLOT_MIN`、`STRATA_GLM_SLOT_GB`、`STRATA_GLM_SLOT_DIR` | 1024、16、`<pack>/slots` | 保留会话的最短长度（token 数，最低到 1）、它们在磁盘上的总大小（GB）、以及文件夹 |
| `STRATA_GLM_SLOT_KEEP`、`STRATA_GLM_SLOT_DAYS` | 关、7 | `1`：保留的会话比引擎活得久——启动时不清空文件夹（槽位的 `.meta` 写明本模型、引擎版本与包才保留，否则删除），引擎退出时保留模型中的会话，`STRATA_GLM_SLOT_DAYS` 天未用的槽位删除。配合空闲时停掉模型的守护进程（llama-swap）：重启后的下一个请求只读取它的新 token（一个 4K token 会话 0.2 s 回来，而不是重读） |
| `STRATA_GLM_SEQS`、`STRATA_GLM_SEQ_CTX` | 1、上下文长度 | 前者即 `--parallel`（见上）：n 个会话同时，n 份序列状态——第一份占整个上下文，其余各占 `STRATA_GLM_SEQ_CTX` 个 token。拆分上每个 token 依次穿过各 GPU，所以每块 GPU 在处理不同的会话（4x RTX 4090、Maya-S24：单独 110 tok/s，两个共 132，四个 225）；每份额外状态从专家池里拿空间（三份 64K：3.5 GB） |
| `STRATA_GLM_SPLIT_ROWS` | `0`（关） | 仅多 GPU 解码：`1`–`8` 在每个部分上最多批量这么多并发请求。每组共享稠密权重读取，而各序列特有的注意力与路由专家仍按请求处理；各组在拆分上流水线化。每个部分在定专家池大小之前先分配行激活（每块 GPU 约 8 MiB）。未设置或 `0` 保持现有流水线。四块 RTX 4000 Ada、Maya-M、八个请求时，`2` 实测合计 94.5 tok/s（比 74.42 tok/s 的旧路径中位数 +27%）；更大的组更慢。单请求解码与预填充在对照检查中保持在 ±5% 以内。 |
| `STRATA_GLM_MULTI_ROOM`、`STRATA_GLM_MULTI_BOUNDARY`、`STRATA_GLM_MULTI_SPEC` | 16384、32、开 | 配合 `STRATA_GLM_SEQS`：一个会话状态在提示后必须留下的空间（更长的提示等第一个大状态就绪；max tokens 被裁到状态大小）、两次专家层更新之间一个会话解码多少 token、以及单独一个请求是否使用推测解码 |
| `STRATA_GLM_PROMOTE` | 8 | 当 CPU 计算所有内存层专家时（它比 PCIe 链路快），每个 token 在后台在显存与内存之间搬移的专家数，让显存跟着你的使用走；`0` = 关 |
| `STRATA_GLM_PROMOTE_MIN` | 0 | 取回的专家只有其老化路由计数至少达到此值才留在显存。`0` 或 `1` 是旧规则（有备用槽就留）。平坦的路由表会提升一次性专家又把它们逐出；128 GB 内存的单块 4090 D 上 `6` 止住了这种抖动；内存紧张时有代价（2x V100、30 GB：29.6 -> 28.3 tokens/s） |
| `STRATA_GLM_PROMOTE_FILL` | 24 | 显存有空闲专家槽时（提示归还借用的部分之后），每个 token 把至多这么多最热的内存层专家拷进去（一层可多个），而不是按 `STRATA_GLM_PROMOTE` 的节奏；`0` = 与搬移同节奏。无论 CPU 通道的计划如何，回填都会运行（只有全 CPU 计划才有显存与内存之间的搬移）：九块 5060 Ti 每 7-8 个内存层专家才经 PCIe 送一个的机器上，长提示后曾有约 13% 的显存一直空着 |
| `STRATA_GLM_VRAM_EVICT` | 单会话 `lru`，多会话 `least` | 一层的备用槽通过逐出其最久未用的专家来补充；`least`（或 `0`）改为按老化计数逐出使用最少的，新近度用于打破平局（成为默认之前的规则）。多个会话同时（`--parallel`）时默认保持 `least`：它们轮流用，而新近度规则让最新会话的专家挤掉所有会话共用的。专家被提升（经 PCIe 取回而非 CPU 通道计算）的程度决定了它的影响：Maya-S 在 3x Tesla V100 16 GB（62 GB 内存、NVMe 层、单会话 128K）上交替 `--bench`，解码 24.8 -> 26.1 tok/s；Maya-L 在 RTX 3090 + 3060、带内存层与 NVMe 层、21K 解码 +2.6%（显存命中 54.5 -> 55.9%）；那里 Maya-M 配 `STRATA_GLM_CPU_PLAN=012234567`（每条路由至多一次 PCIe 取回）不变 |
| `STRATA_GLM_MV_RDNA` | AMD：Ryzen AI Max APU（gfx115x）开 | 解码的稠密 GEMV（Q6_K 与 BF16 投影）放进一个 RDNA3 内核，让两个 block 的载入同时在途，与原内核逐位一致（Radeon 8065S、Maya-S：解码 +3.1%）；`1` = 任何 gfx11 都开（RX 7900：尚未测量）；`0` = 关。NVIDIA：无影响 |
| `STRATA_GLM_MTP_GGUF` | 模型自带的 | 两块 GPU 及以上：一个 GGUF，存放用来起草的 MTP 草稿模块——用于没有发布草稿模块的模型，或比其自带的更精确的模块（`tools/maya_quant/mtp_gguf.py` 可单独写出一个模型的模块）；`STRATA_GLM_NO_MTP=1` = 不起草 |
| `STRATA_GLM_MTP_KEEP` | 2 | 两块 GPU 及以上：草稿模块只在其路由得分前 n 名中计算它缺显存的专家（CPU 通道 / PCIe 同任意路由），其余跳过——草稿只需要是个好猜测，但被否决的草稿要花第一块 GPU 重做。`2` 比全部跳过起草得更好且从未更慢（2x V100：接受率 82-83% 对 76-81%；RTX 3090 + 3060：19.5 -> 20.6 tok/s）；`0` = 跳过所有缺失专家；`STRATA_GLM_MTP_MISS=1` = 全部取回 |
| `STRATA_GLM_MTP_DECODE` | Ryzen AI Max APU（Strix / Gorgon Halo，gfx115x）开 | 单卡：GPU 装得下整个模型时回答更快（Radeon 8065S、Maya-S：短提示的 256-token 回答 +44-50%，3-8K token 提示后 +26-27%），答案与不开完全相同。`0` = 关；`1` = 其他单卡也开（专家从内存流式读取的地方通常更慢）。不与 `--parallel` 或 DFlash2 起草器同用 |
| `STRATA_GLM_TRUNK_KEEP`、`STRATA_GLM_TRUNK_KEEP_MAP` | 未设置（8） | **会改变答案**——给宁愿快一点的内存受限拆分用：模型的层从路由第 n 名起跳过它缺显存的专家，与草稿模块相同；`_MAP` = 每层一个数字（`8` = 全部专家）。排名最低的专家权重最小、也最冷；GLM-5.3 的早期层比晚期层对缺失专家敏感得多（一次只在一个层段保留 5 个，Maya-M，留出文本上对完整模型的 KL：层 3-7 0.040，23-29 0.012，30-44 0.002-0.003）。RTX 3090 + 3060、21K 解码：Maya-M 配 `_MAP=8888888888888888888888855555554444444444444448`（层 23-29 保留 5、30-44 保留 4）与拆分 28 时 20.0 -> 24.6 tok/s，KL 0.018（代码 0.017）；统一的 `TRUNK_KEEP=7` 是 21.8、KL 0.032；Maya-L 同一 map、拆分 27，+25%，KL 0.017。验证窗口（DFlash2、批量窗口）跳过与 token 路径相同的专家 |
| `STRATA_GLM_DISK_SKIP_W`、`STRATA_GLM_DISK_SKIP_FROM` | 关（0、8） | **一种近似，给显存 + 内存装不下的专家用**：既不在显存也不在内存层的路由专家，当其占路由权重的份额低于 `STRATA_GLM_DISK_SKIP_W`、或其 rank（0-7，路由顺序：得分 + bias）不小于 `STRATA_GLM_DISK_SKIP_FROM` 时，直接从 token 中跳过而不是从固态硬盘读取。权重规则跳过最不重要的：一块 3090、60 GB 内存、74 GB 专家、PCIe 4 NVMe，`--bench` 解码在 `_W=0.12` 时 13.3 -> 21.6 tokens/s；对完整计算的 KL（CPU 通道关，575 token）：`_W=0.12` 0.12、`_W=0.14` 0.15、`_FROM=6` 0.08、`_FROM=4` 0.11（Maya-S 距 FP8 是 0.43；CPU 通道自身的算术使输出移动约 0.06）。引擎日志的 `glm skip` 行统计被跳过的专家。lookahead（`STRATA_GLM_AHEAD`）届时只读取路由会等待的那些 rank |
| `STRATA_GLM_MISS_SKIP_W`、`STRATA_GLM_MISS_SKIP_FROM` | 关（0、0） | **会改变答案**——给内存受限的解码：一个不在显存里、且占路由权重份额低于此值的路由专家被直接跳过（不 PCIe 取回、不 CPU 通道计算），从 `_FROM` 层起——即 `TRUNK_KEEP` 的思路按权重而非 rank，所以只路由到一两个强专家的 token 比均匀分布的 token 丢掉更多轻量失手。GLM-5.3 的早期层比晚期层敏感得多：`_FROM` 保持在 23 左右（每层都开时，0.04 已用约 1% 速度换来 KL 0.041）。单块 RTX 3090、Ryzen 7 9700X、DDR5-3600、`_FROM=23`、完整套件（聊天 / agent tok/s，1,746 token 上对未裁剪模型的 KL）：Maya-S24 `TRUNK_KEEP_MAP` 第 4 档（`...5555555444...`）27.4 / 27.3，KL 0.027；0.12：29.3 / 27.7，KL 0.037；0.14：30.1 / 29.0，KL 0.052；0.16：30.9 / 29.6，KL 0.075（相同 KL 的 `TRUNK_KEEP_MAP`：29.7 / 28.8）。Maya-M 第 4 档 19.9 / 18.1，KL 0.038；0.12：21.3 / 19.5，KL 0.047；0.14：21.8 / 21.0，KL 0.060。请求可以为一次扫描自带参数而无需重启：`"strata_tune": {"miss_skip_w": 0.14, "miss_skip_from": 23}`（0：对它关闭） |
| `STRATA_GLM_SPEC_HEAD` | GPU 数的一半 | 两块以上 GPU：拆分的头几块 GPU 中有几块组成推测解码的头组——头组运行草稿的位置，其余完成当前 token，与两块 GPU 拆分的两半相同（4x Tesla T4、Maya-S：解码 14.1 -> 21.0 tok/s，对比不起草）；`STRATA_GLM_NO_SPEC=1` = 逐 token |
| `STRATA_GLM_SPEC_CANCEL` | 关 | 两块 GPU（头组一块）：`1` = 当尾部的 token 表明草稿错了时，头部的推测趟在下一层停下（它的重内核空转，路由空着出去），重做立刻开始、与尾部的起草并行——而不是等一趟已知要作废的趟跑完。token 相同（贪心、CPU 通道与层映射关：三个请求与逐 token 完全一致，其中一个 166 次取消）。RTX 3090 + 3060、Maya-M：每个被否决的草稿省下第一块 GPU 5.4-6.5 ms（贪心、文本相同：解码 +0.9 到 +2.6%）；21K 采样下落在逐次波动之内（+0.6-1.2%）——在第一块 GPU 定节奏更明显、否决更多的地方收益更大。配 `STRATA_GLM_SPEC_SAMPLE=1` 两者叠加：那台机器上合计 +5.3%（第一块 GPU 平均一趟 31.6 -> 30.4 ms） |
| `STRATA_GLM_SPEC_SAMPLE` | 关 | 两块 GPU 及以上（流水线解码）、采样请求：`1` = 推测采样——草稿模块的草稿从它自己的分布 q 中抽取（其 logits 上的请求采样链、自己的 Philox 流），token 以概率 min(1, p/q) 取草稿，否则从 max(0, p - q) 抽取：token 精确保持普通抽取的分布，而草稿平均有 sum(min(p, q)) 的时间成立，而不是 p（草稿模块的 top token）。固定种子仍会重复它的答案——但不是它不开此项时给的答案。贪心请求（以及此路径不支持的 penalties）不变。RTX 3090 + 3060、Maya-M、21K、默认采样：草稿保留率 68.6 -> 74.7%，单独解码 +1.6-2.7%，配 `STRATA_GLM_SPEC_CANCEL=1` +5.3%（24.30 -> 25.57 tok/s） |
| `STRATA_GLM_SERVICE_IDLE_MS` | 200 | 引擎的层线程（每 GPU 一个）在解码路由专家时自旋，超过这么久没有解码就睡眠，所以空闲引擎用一个核的约 1%，而不是每块 GPU 一个核；`0` = 一直自旋 |
| `STRATA_GLM_PREFILL_CPU` | 开 | 提示：路由最少的内存层专家在 CPU 上计算，同时 GPU 经 PCIe 加载其余的，分工按每层平衡让两边同时完成；`0` = 关。`STRATA_GLM_PREFILL_CPU_ROW_MS` 固定其计划使用的一行 CPU 成本（默认：学习得到） |
| `STRATA_GLM_PRESTAGE` | 单卡 160，多卡 0 | 提示：一层的注意力运行时（那时 PCIe 链路空闲）把专家拷到 GPU，放进从池尾借来的、容纳这么多专家的缓冲——下一层路由最多的内存层专家；`0` = 关。`STRATA_GLM_PRESTAGE_ADAPT=1`（实验性，尚未测量）按每层注意力的耗时学习每层的数量 |
| `STRATA_GLM_DISK_READERS` | 关（0） | 提示：这么多读取线程读取只在磁盘上的专家，每个线程一个在途（它的三个部分整块读取），它和排在它前面的都落地后立刻移交，而不是一个读取线程一次按 `STRATA_GLM_DISK_QD` 小块读取、全部落地才移交整批。随机读取喜欢大请求的 NVMe 收益最大：无 DRAM 的 Kingston NV3（74 GB 专家、60 GB 内存的 3090），`--bench` 配 `8`：2K 提示 +50%，8K 提示 +25%；输出相同（对旧读取器的 KL 0.019，在其自身逐次波动 0.018 之内）。自带 DRAM 的 NVMe 反而可能变慢（2x V100 旁的 Samsung PM981a：配 `8` 提示慢 15%），所以要测你自己的 |
| `STRATA_GLM_PREFILL_TRACE` | 关 | `1`：每块提示的每层，注意力耗时，以及预装拷贝、拷贝通道、CPU 通道何时结束（调试） |
| `STRATA_GLM_KQ` | 开 | 提示的 CPU 专家：多 token 的 AVX-512 Q3_K/Q2_K 内核（8 token 一个专家起，约为 ggml 的 2 倍）；`0` = 只用 ggml 的点积 |
| `STRATA_GLM_CPU_MT` | 开 | 多个 token 共享的 CPU 专家（DFlash2 窗口的合并 CPU 趟、提示的 CPU 专家）：每行权重最多为 8 个 token 解码一次——IQ2_XXS/XS/S、IQ3_XXS、IQ3_S、IQ4_XS、Q4_K、Q5_K，在 AVX2 或 AVX-512 上——结果与 ggml 的一对一（pair）点积逐位相同（首次使用时对照校验；对不上的格式保留 ggml 的）。Maya-L 在 Xeon 6152、40 线程、一个专家被 2 / 4 / 8 个 token 共享：455 -> 333 / 889 -> 434 / 1794 -> 864 us。IQ2_XXS 与 IQ3_XXS 的行在寄存器里构建符号（单核、2-8 token：比查表快 23-34%）。单 token：`STRATA_GLM_CPU_1T`；`0` = 只用 ggml 的点积 |
| `STRATA_GLM_CPU_1T` | 开 | 解码步中使用 IQ3_S 或 IQ3_XXS 权重（Maya-L 的 gate/up）、每专家单 token 的 CPU 专家：ggml-cpu 的 AVX2 点积配更廉价的解码——IQ3_XXS 的符号在寄存器里构建、IQ3_S 的网格索引在标量寄存器里——与其逐位一致（首次使用时对照校验）。Maya-L 在 Xeon 6152，一个专家的 gate/up + down 单 token，1 / 12 / 40 线程：3677 -> 3274 / 370 -> 333 / 250 -> 247 us——CPU 通道核越少，收益越大。IQ2_XXS（Maya-S / Maya-S24 的 gate/up）在带 AVX-512 VBMI 与 BMI2 的 CPU（Zen 4 及以后、Ice Lake 及以后）：网格作为 2-bit 码放进寄存器（每个网格字节都是 0x08 / 0x19 / 0x2b）、符号用 pdep 构建，也逐位一致——带 RTX 3090 的 Ryzen 7 9700X、Maya-S24、8 线程：一个专家 0.220 -> 0.175-0.196 ms，解码 +6-8%。`0` = ggml 的点积；`STRATA_NO_IQ512=1` = 仅 AVX-2 档 |
| `STRATA_GLM_PREFILL_SUB` | NVIDIA：至多 4096；AMD：256 | 提示：注意力层、稠密 FFN 与共享专家运行的子批大小（16-8192）。NVIDIA 上未设置时，注意力层与稠密 FFN 的是 256 翻倍、只要装得下该块 MoE 缓冲已占的内存（更少更大的 GEMM，每个权重每个子批只反量化一次），共享专家的保持 256；`256` = 旧的子批 |
| `STRATA_GLM_KV_INT8` | 关 | `1`（或配置 `"args"` 里的 `--kv int8`）：注意力的隐式缓存用 INT8——每 32 个值 512 个码加一个 FP16 scale，每 token 每层 544 字节而不是 1024。单块 V100、128K 上下文：状态 1.78 -> 1.13 GB，多出 84 个专家槽；质量无可测变化（对 FP16 的 KL 在 FP16 自身逐次波动之内）；提示读取速度约为 FP16 的 2% 以内（两块 V100，8K-32K） |
| `--kv-streaming on\|off`、`--kv-resident N` | 开（安装器的配置）、32768 | KV 流式传输，与 Strata 相同：安装器把 `"--kv-streaming", "on"` 写进每个配置的 `"args"`——那里的 `off` 关闭它，旁边的 `--kv-resident N` 设窗口（`STRATA_GLM_KV_RESIDENT` 从环境同样设置），重新安装会保留你的设置：每个注意力层的隐式缓存住在锁定的内存里，其 N 个位置（至少 20480）在显存里，其余原本占用的显存装专家。indexer 选定一个 token 的 2,048 个位置后，它点名的 4 个一组、不在显存里的块经 PCIe 拷入（一次时钟扫描决定谁离开）；提示从一份 staging 副本读取整层缓存，该副本在前面几层计算时经 PCIe 填充（`STRATA_GLM_KV_PREFETCH=0`：在计算流上、该层之前）。不超过 N 个位置的上下文不流式传输；多个会话同时（`--parallel`）时每个保留自己的主机副本。在缓存相对显卡很大时划算：1-3 块 GPU 上的长上下文与多会话（带 DFlash2 起草器的 Maya-L，一张 RTX 3090 与若干 RTX 5060 Ti，183 GB 内存）。3090 上 512K 时缓存要占 6.4 GB 显存：流式传输后有 1,176 个专家槽而不是 714，提示快 4-9 倍（100K token 836 对 92 tok/s——不开时池只借给提示路径 768-token 的块），解码 +5%。3090 上 1M 只有流式传输才能启动（900K token 的提示 489 tok/s）。三块卡 1M 解码快 16%；两块卡 1M 配 `--prefill 32768` 读提示快 26%（不开流式传输时拆分退回 16K 块）；两块卡四个 128K 会话（`--parallel 4`）合计解码快 37%。128K 单会话时变化不大（3090：专家槽 1,218 -> 1,302，解码在逐次波动之内；提示和之前一样快——100K token 856 / 858 tok/s，不开预取 809）。解码给出与不开时相同的 token；提示的 token 可能略有不同，正如任何改变哪些专家能进显存的变动一样 |
| `STRATA_GLM_DSA_SCORE` | 开 | 提示：稀疏注意力的 indexer 用寄存器分块的 FP32 GEMM 打分（一块 3090，位置 10K 处 4096 个 token：78.8 -> 5.6 ms）；`0` = 每池一个 warp 的内核 |
| `STRATA_GLM_DSA_ALL` | 开 | 解码：当不超过 indexer top-k 位置数（2048）的上下文让每个池都入选时，稀疏注意力按位置顺序全部取用、不打分——省掉 indexer 的查询 GEMV、打分与选择（每个 DSA 层约 40 us）；所有解码路径一致，token 在它们之间保持一致。`0` = 任何长度都照常打分与选择 |
| `STRATA_GLM_MLA_MMA` | 开（NVIDIA sm_80+） | 解码：稀疏注意力的隐式打分与上下文用张量核计算（F32 求和的 FP16 部分，乘积保持 F32 精度），一个 block 最多跑 16 个 32 格的块；验证窗口的 token 在几次启动内完成。512-2048 格时比 F32 块快约 4 倍（RTX 5060 Ti 在 512 格时 49 -> 11 us 每 token 每 DSA 层）。`0` = F32 块 |
| `STRATA_GLM_PREFILL_ATTN` | 张量核 | 提示的稀疏注意力内核：Ampere 及更新上是每 block 32 个头的 mma.sync 内核（一块 3090，26.6K token 提示：832 -> 966 tok/s）；`wmma` = Volta 跑的 91 KB 共享内存内核；`tcreg` = Turing 跑的寄存器内核；`f32` = FP32 内核。AMD：gfx12（R9700 / RX 9070：提示 +8-10%）与 Strix Halo（gfx1151：提示 +4%）默认 `wmma2`，其他 gfx11 默认 `f16q`。`STRATA_GLM_PREFILL_ATTN_CHECK=1` 把选定内核与 FP32 内核对照（调试） |
| `STRATA_GLM_EXL3_MMA_ROWS` | 64 | EXL3 包的提示：一块中被路由不超过这么多 token 的专家，用边走边解码其 trellis 的张量核内核跑（RTX 30 及更新）；更忙的专家解码成 FP16 一次、交给 cuBLAS 乘。`0` = 所有专家都走 cuBLAS |
| `STRATA_GLM_EXL3_DECODE` | 按 GPU | EXL3 包的解码内核：RTX 30 及更新在每个 tile 上用张量核解码（`new`）；更早的 NVIDIA GPU（V100、T4、RTX 20）保留 v1.1.8 的内核（`legacy`），它们在 CUDA 核上比新内核解码快约 10%，输出与 v1.1.8 一致。`legacy` 或 `new` 在任何 GPU 上强制其一（A/B）；提示两种情况走同一路径 |
| `STRATA_GLM_EXL3_ROT` | 1 | EXL3 包的提示、走 cuBLAS 的专家（A/B）：它的行承担输入与输出旋转，权重只解码到 FP16（`x W = H(H(x * suh) W') * svh`，与解码相同）；`0` = 改为把权重旋转并缩放到 FP16（每个权重两次 128 点变换：9 块卡上一个 8K 提示慢 5-7%） |
| `STRATA_GLM_MMQ_GLU` | 开（NVIDIA，RTX 20 及更新） | 提示：专家的 SwiGLU 在 MMQ gate/up 乘积的回写中施加（固定版本 MMQ 上的 `third_party/ggml/mmq-fused-swiglu.patch`）：它写出一块的 T x n_ff 个值，而不是要另一个内核读回的 T x 2 n_ff——在普通乘积于单个 block 内求和每个输出的地方逐位相同，一块的大小正是如此（`glm_mmq_glu_test`）。Maya-L 的形状、一个 8K 块的一层：RTX 5060 Ti 43.5 -> 36.0 ms，RTX 3090 26.3 -> 25.4 ms；单块 RTX 3090 读提示和之前一样快（那里它在等 CPU 和 PCIe）。`0` = 单独的内核。patch 打不上的 ggml 检出（`-DSTRATA_GGML_DIR`）构建时不带它 |
| `STRATA_GLM_DROP_CACHE` | 2 个及以上 NUMA 节点时开 | 在锁定内存层之前，丢弃模型文件的干净页缓存（引擎用 O_DIRECT 读专家）：一个缓存的 GGUF 占满一个节点会让交错铺开的层 78% 落在另一个节点上，CPU 通道于是读着一个插槽的内存；`0` = 保留 |
| `STRATA_GLM_PROFILE_WEIGHT` | 1 | 配合 `STRATA_GLM_USAGE`：包的路由画像（`expert_counts.txt`）在你的使用文件之上的权重，决定启动时专家层的顺序；`0` = 只用你的使用 |
| `STRATA_GLM_LOOP_THINK`、`STRATA_GLM_LOOP_ANSWER` | 256、1024 | 死循环守卫：当最近生成的这么多 token 是一个至多 32 个 token 模式的精确重复（量化模型可能陷入 `0 0 0 ...` 直到 `max_tokens`）时，立即关闭思考；在回答中则关闭模型当前的工具调用参数（或整个轮次）让 agent 继续；同一回答里的第二次死循环会结束它。`0` = 关 |
| `STRATA_GLM_WIN_PIPE` | 三块 GPU 及以上 | 拆分上的 DFlash2：`1` = 流水线验证窗口（各 GPU 同时处理相邻 token），`0` = 批量窗口（整个窗口一次一块 GPU）。两块卡用批量的，三块及以上用流水线的 |
| `STRATA_GLM_WIN_THREADS`、`STRATA_GLM_WIN_SKIP` | 开 | 流水线窗口（A/B）：`0` = 一个主机线程发出所有 GPU 的工作（九块卡：每 token 3-5 ms，比任何 GPU 的份额都长）/ 错误草稿之后的 token 照常跑它们的工作 |
| `STRATA_GLM_HEAD_SPLIT`、`STRATA_GLM_HEAD_SPLIT_CHECK` | 开（3 块及以上 GPU 上带 DFlash2 的 EXL3 包）、关 | 流水线窗口的取值把输出头的词表前半在最后一块 GPU 上读、后半在一块助手 GPU 上（第一块与最后一块之间层数最少的那块，它保留自己的副本：约 240 MB 专家池）同时读，然后合并——整行的取值；`STRATA_GLM_HEAD_SPLIT=0` = 整个头在最后一块 GPU 上。`STRATA_GLM_HEAD_SPLIT_CHECK=1`（调试、更慢、贪心）：整行也照读，取值对照——退出时引擎日志打印取值次数与不一致次数 |
| `STRATA_GLM_DFLASH_HEAD_SPLIT` | 开（随头拆分） | DFlash2 草稿的头同时读同样的两半——词表前半在最后一块 GPU、后半在助手，各取自己切片的 top-k 候选（起草器运行时助手空闲）；每轮起草省约 0.5 ms。`0` = 整个头在最后一块 GPU 上 |
| `STRATA_GLM_DFLASH_AHEAD` | 开（流水线窗口） | 下一轮的上下文注入与草稿在该轮最后一个取值一到手就从自己的线程排队，与提交和下一个锚点的开始并行（它们约 1.3 ms 的启动原本排在这些之后，期间起草器的 GPU 空闲）。`0` = 在下一轮开头 |
| `STRATA_GLM_DFLASH_INJ_OVERLAP` | 开 | 起草器的上下文注入（保留位置的 K / V 进它的 ring，RTX 5060 Ti 上约 0.5 ms）在自己的流上与下一次草稿并行：草稿的第一层同时运行，每层的注意力只等自己那层的行。`0` = 在草稿之前、草稿的流上 |
| `STRATA_GLM_WIN_COMMIT_WAIT` | 关 | `1` = 流水线窗口的提交会等所有 GPU 的恢复与重放都排队后才开始下一个锚点（默认每块 GPU 的下一次开始排在它自己的提交之后，且状态在被任何其他东西读取前会被等待） |
| `STRATA_GLM_WIN_REPLAY` | 开 | DFlash 窗口的提交把保留的 token 一次性重放到一块 GPU 全部 KDA 层的循环状态备份上（每 token 两次启动），再把备份拷到活动状态——下一个流水线窗口因此跳过它的备份。`0` = 先恢复活动状态，再逐层重放 |
| `STRATA_GLM_WIN_LANES`、`STRATA_GLM_WIN_CPU`、`STRATA_GLM_WIN_STAGGER`、`STRATA_GLM_WIN_ALLCPU` | 2、开、开、开 | 批量窗口（单卡，或两卡）：它的 CPU 通道用两个池——单卡：另一个 NUMA 节点上的第二个池（`2`：两个都不绑，`1`：各绑一个节点，`0`：一个池）；两卡：每块 GPU 的窗口层也在另一块 GPU 的池上——每层去重后的专家读一次、在两个池之间分（`STRATA_GLM_WIN_CPU=0`：一池一个 token）；窗口的 token 分两组、层两两交错，CPU 回答一组时 GPU 跑另一组的注意力（`STRATA_GLM_WIN_STAGGER=0`：一次一层全部）；以及窗口的所有内存层专家都给 CPU（`STRATA_GLM_WIN_ALLCPU=0`：校准过的 CPU / PCIe 计划）。单个 token 仍用一个池 |
| `STRATA_GLM_WIN_BATCH` | NVIDIA：开，AMD：关 | 批量窗口的路由专家：每个去重专家为窗口全部 token 跑一次（`1`），或每个 token 的专家走 token 路径的内核（`0`；AMD 的默认——它的 token 路径专家内核为 Ryzen AI Max APU 调过优，窗口的 token 于是也一次一层）。两种方式 token 相同 |
| `STRATA_GLM_WIN_MULTI` | 开 | `STRATA_GLM_WIN_BATCH` 关时（AMD）：窗口各 token 的路由专家一次启动全部处理（每个 token 的算术与单独跑相同；两个 token 路由到的同一专家只从内存读一次——Radeon 8065S 的回答 +10%）；`0` = 一个 token 接一个。两种方式 token 相同 |
| `STRATA_GLM_WIN_X3`、`STRATA_GLM_WIN_X3_POOLS` | 开、2 | EXL3 包、批量窗口（A/B）：`STRATA_GLM_WIN_X3=0` = 每个 token 的路由专家在 GPU 上一个 token 接一个跑，而不是每个去重专家为它的全部 token 跑一次（两种方式比特相同）；`STRATA_GLM_WIN_X3_POOLS=1` = 合并窗口层的 CPU 专家只在第一个池上跑，而不是分给两个（单卡时两个不绑的池共享同样的核心；一张 RTX 3090 上带 DFlash2 的 EXL3-4.05bpw：两种方式速度相同） |
| `STRATA_GLM_DFLASH_STATS`、`STRATA_GLM_DFLASH_PROF` | 关 | DFlash2：每轮一行写进该文件（检查与保留的草稿、草稿与窗口的 ms、本轮 CPU 通道专家与显存失手、草稿的置信度）/ `1` = 每个请求的轮次计时进引擎日志 |
| `STRATA_GLM_DFLASH_TL` | 关 | DFlash2 的流水线窗口：每轮一行写进该文件——从本轮第一个 token 开始的 ms：草稿就绪与推送、主机的开始步骤、每个取值从最后一块 GPU 出来时，然后是提交完成的 `c` 行 |
| `STRATA_GLM_DFLASH_PRIO` | 开 | 起草器的流用设备的最高优先级，使草稿不被排在最后一块 GPU 已排队的工作之后（9 块卡、EXL3-4.05bpw：一次草稿 6.4 -> 5.7 ms）；`0` = 默认优先级 |
| `STRATA_GLM_TIMING`、`STRATA_GLM_POOL_STATS` | 关 | `1` = 计时与缓存统计进引擎日志 |
| `STRATA_GLM_ROUTE_LOG` | 关 | `<prefix>`：每条解码路由一行写进 `<prefix>.<GPU>`——层、它的专家、哪些被取回、失手或在 CPU 上算，然后 `|` 之后是 16 个接近未选中的（同一得分下的下一批 rank）——供缓存研究：`tools/glm_tier_replay.py` 按显存层规则及其替代方案的模型重放一份日志 |
| `STRATA_GLM_SCORE_PREFILL` | 关 | 配合 `STRATA_GLM_SCORE=<c>`（第一个 c 之后对 `--tokens` 打分）：`1` 让那 c 个 token 走提示路径而不是逐个，长上下文几分钟就能打完分（调试：长上下文下比较 KV 缓存格式） |
| `STRATA_GLM_MULTI_BENCH` | 关 | `<n>,<prompt>,<tokens>`（`STRATA_GLM_SEQS` >= n、`--glm-pack`、不 `--serve`）：n 个提示，然后各贪心生成那么多 token——序列 0 单独走 token 路径，然后全部 n 个走流水线解码；正的 `STRATA_GLM_SPLIT_ROWS` 让每个部分最多 k 个序列一组并报告 `split-rows`；打印两个速率以及序列 0 有多少 token 一致（`STRATA_GLM_CPU_LANE=0` 时全部一致） |
| `STRATA_GLM_THINK_RESERVE` | 1024 | 服务器：思考块要提前关闭到给回答至少留下这么多 token，或请求 `max_tokens` 的四分之一（取更大者；配置的 `"thinking_budget"` 按请求下调以适配）——否则 `max_tokens` 在模型还在思考时用尽的请求会以完全没有回答结束 |
| `STRATA_ENGINE_STALL_S` | 90 | 服务器：引擎静默这么久、且期间没用过 CPU、磁盘或 GPU，就是卡住了——结束它，请求收到错误，下一个请求重新启动它（正在干活的静默引擎永远不会被结束）；`0` = 关 |
| `STRATA_WATCHDOG_S` | 180 | 引擎：一个请求这么久没完成任何提示层和任何 token 就是挂起了，即使有线程在自旋（#40）。引擎把卡住的位置写进它的日志（Windows 上还写 `maya-stall-<pid>.dmp`，含所有线程的栈：提 issue 时两个都附上）并结束；请求收到错误，下一个请求重新启动它。`0` = 关 |
| `STRATA_HTTP_BACKLOG`、`STRATA_MAX_BODY_MIB` | 256、256 | 服务器：允许等待接受的连接数，以及最大请求体（MiB，更大的在读取前收到 413） |
| `STRATA_ENGINE_QUIT_S` | 50 | 服务器：停止后给引擎多久来结束（并在 `STRATA_GLM_SLOT_KEEP` 下保留它的会话）——SIGTERM 按 Ctrl+C 处理，第二个被忽略；守护进程自己的宽限（llama-swap 的 `unloadTimeout`）应超过它 |
| `STRATA_ENGINE_WRAP_S` | 40 | 服务器：停止时，进行中的请求会被收尾而不是丢弃——思考立即关闭，它最多再回答这么久（然后截断），客户端得到回答、这一轮结束；还在排队的请求被拒绝。把它加上引擎的结束时间保持在守护进程宽限之内（llama-swap 的 `unloadTimeout`）；`0` = 立即截断 |
| `STRATA_FAIR_SLICE_S` | 0（关） | 服务器：当另一个请求在等待时，已解码这么多秒的回答把引擎让给它、之后从停下的地方继续（引擎随它写入的内容保留状态：一个 token 会被重读）。并发聊天与 agent 的旁路请求不再等整个回答（4x RTX 4090，3000-token 回答期间的短问题：首 token 34 s -> 15-20 时 13-17 s）。每次切换把一个会话放到磁盘上（SATA SSD 阵列上 100K token 0.1-1 s），慢磁盘需要更长的片 |

**专家画像**（`data/expert-profile-glm.bin`）。启动时引擎用画像排在最前的专家填充显存。该画像给每个
（层，专家）对排序，它的顺序还决定每层分到多少显存槽——拆分上的 MTP 草稿层除外，它在每个起草的 token
上运行，因此至少保留与其显卡上其他层相等的份额。安装器把随包发布的那份 `data/expert-profile-glm.bin`
传给每个模型（配置 `"args"` 里的 `--expert-profile`；在它之前写的配置会在下次启动时获得它，因此更新
无需重装）。它来自 router 的路由，所以适用于 GLM-5.3-Flash 的每种量化：一份 Maya-L 带 MTP 草稿模块
运行的路由 trace，覆盖 66 个请求（代码、带思考的数学、聊天、八种语言、工具调用、长文档）。Maya-L 在
RTX 3090 + 8x RTX 5060 Ti 上冷启动：解码平均 40.7 tok/s，无画像时 35.4；起初 96% 的路由由显存承接，
无画像时 89%。除非你要求，否则跨启动不学习任何东西：运行期间引擎仍会把你在用的专家搬进显存，但下次
启动又从画像重新开始。这些引擎参数写进配置的 `"args"`；重新安装会保留它们：

| 参数 | 作用 |
| --- | --- |
| `--expert-profile FILE` | 从这份画像启动，而不是随包发布的那份 |
| `--expert-profile-save FILE` | 把专家层学到的东西保存为画像：显存里的专家，然后是计数到的路由，最后是它启动时用的画像。在退出时以及每 `--expert-profile-save-every MIN` 分钟（默认 10；`0` = 仅退出时）在请求之间写入。把 `--expert-profile` 指向它即可从那里启动（你自己的文件，不是随包发布的那份：重新安装会保留它） |
| `--dump-routing FILE` | 把每条路由（提示与回答）写成路由 trace，作为 `tools/make_profile.py` 的输入 |

为你的用法构建一份画像：用 `--dump-routing /tmp/trace.bin` 启动模型一次，发送你典型用法的请求
（代码、文档、聊天、语言），停掉它，然后运行
`python tools/make_profile.py --glm --no-base /tmp/trace.bin --out my-profile.bin`。该工具也读取引擎的
文本记录——一个使用文件（`STRATA_GLM_USAGE`）、一个包的 `expert_counts.txt`——`--equal` 让多个输入
权重相同。完全没有画像时，引擎读取包自己的 `expert_counts.txt` / `expert_prior.txt`
（`tools/glm_expert_prior.py`），否则按专家顺序填充。

**提示分块**（`--prefill`）。引擎分块读取提示，每块路由到的每个专家每块拷贝进显存一次，所以更大的块
让长提示读得更快；一块的缓冲在提示运行期间从专家池借用。安装器把 `--prefill auto` 写进配置的
`"args"`（重新安装会保留你在那里的改动）：

| 参数 | 作用 |
| --- | --- |
| `--prefill auto` | 每块卡的专家池能借出缓冲的最大块，以 256 token 为步长——至多其槽位的 90%（当不足 90% 的专家字节在显存或锁定的内存层时为 85%），始终保留一条路由的专家与备用槽——单卡至多 32768 token、拆分至多 8192——各自在更快的地方（一块 V100，26K token 提示：32768 时 559 tok/s，8192 时 367；两块 V100：8192 时 709，32768 时 551）。拆分上每块卡读同样的块，所以由最小的卡定，日志会点名卡在比第一块卡更低的位置 |
| `--prefill N` | N 个 token 一块（比如 `32768`），不断减半直到每块卡的池能借出——只需保留一条路由的专家与备用槽；`0` = 逐 token。拆分上内存 96 GB 及以上时，安装器指向 `32768` |

一块卡每块几乎要把每个专家经 PCIe 流一遍，所以更少更大的块在长提示上划算。拆分上，装得下一个块的
提示让各卡依次运行，而两个或更多块让两边都忙。

## 专家池的大小

每块 GPU 的显存在模型的稠密权重、上下文状态与起草器之外剩下的部分给专家（专家池），再减去给驱动、
采样器和之后才分配的东西（验证窗口的 graph，首次使用时捕获）留的预留。安装器把 `--expert-cache auto`
写进配置的 `"args"`；你自己的数字和下面的预留也写在那里，重新安装会保留它们
（`STRATA_GLM_EXPERT_CACHE`、`STRATA_GLM_RESERVE_MB` 和 `STRATA_GLM_RESERVE_LATER_MB` 从环境同样设置；
配置的 args 优先）：

| 参数 | 作用 |
| --- | --- |
| `--expert-cache auto` | 默认：池取空闲显存减去预留。这个预算写定后，空闲显存会被再读一次：Windows 下分配只有被触碰后才计数，之前读到的空闲值可能高约 1 GB，而超过它填满的卡会把 GPU 内存换页到系统内存而不是失败（解码慢几倍；请求可能永久卡住）。预留不够时，池归还差额并重新检查；驱动拒绝的分配（Windows 还把显存计入提交量，内存 + 页面文件）以驱动给到的为准结束。不检查：APU（Ryzen AI Max：它的 GPU 内存就是系统的），以及远超池能容纳总量的预算（全部专家加备用槽和提示的空间——它填不满卡）。`STRATA_GLM_POOL_CHECK=0` 跳过检查 |
| `--expert-cache N` | 你自己的大小：第一块 GPU 的池至多装 N 个专家（N 个它的最大专家槽）；拆分的后续 GPU 保持 `auto`。写定后不再检查：被填得过满的卡只在启动时报告 |
| `--vram-reserve-mib N` | 池之外留出的空闲显存：默认 700 MiB（Linux 下的 Ryzen AI Max APU 为 1024）。在小卡上若这会让池在每层装不下一条路由的专家和备用槽，`auto` 会把它降到低至 300 MiB 并说明；这里设的数字会保留。安装器也接受它（`--vram-reserve-mib N`） |
| `--vram-reserve-later-mib N` | 拆分后续 GPU 上的同样预留（默认：`--vram-reserve-mib` 的值）——不驱动显示器的卡需要更少 |

启动时引擎日志会说明每块 GPU 在全部加载后还剩多少显存（`maya serve: CUDA0: ... MiB of VRAM free with
everything loaded`）。低于 256 MiB（拆分后续 GPU 为 512）时加上 `LOW` 和能留出空间的预留值。引擎只是
说明：调大 `--vram-reserve-mib`（`--vram-reserve-later-mib`）、调低 `--max-context`，或关闭其他用 GPU 的
程序。起草器的验证窗口在启动之后的首次使用时分配：一块带 DFlash2、32K 的 RTX 3090 启动时剩 431 MiB，
第一个请求后剩 77 MiB——在 Linux 上没问题（满卡会让分配失败而不是换页），但在 Windows 上是调大预留的
理由。

## 可重复的答案

贪心请求（`temperature` 0）只有在每个专家每次都由同一条路径计算时，才会在重启后返回相同的 token。
CPU 通道开着时做不到：一个内存层专家去通道还是经 PCIe 去 GPU 取决于它住在哪，而两条路径的舍入不同——
GPU 的内核按 32 个值一块量化激活，ggml-cpu 的点积（通道）按 256 个值一块，一个专家的输出相差约 2%
（相对 L2，Maya-S 的 42 层上 150 次通道调用的中位数；对照双精度参考，通道偏 1.7%，GPU 偏 1.1%）。
router 的 top-8 会放大它：一旦通道算了一个之前由 GPU 算的专家，几个 token 之内就有四分之一或更多的层
选出不同的 8 个专家，贪心 token 在几十个 token 内分叉。让通道在不同启动之间计算不同专家的东西：

| 什么 | 为什么不同 | 怎么固定 |
| --- | --- | --- |
| 显存池 | `--expert-cache auto` 取空闲显存减预留，所以别的程序占用的几百 MiB 就是少几百个专家（启动的 `tiers warm: N experts` 和 `only ... MiB free once the expert pool ... GB less` 行；同一台电脑五次启动 9,453-9,488 个专家） | 配置 `"args"` 里的 `--expert-cache N`（优先于 `STRATA_GLM_EXPERT_CACHE`），N 比 `auto` 给出的略小 |
| 通道的分工 | 启动时计时：当一个专家在 CPU 与经 PCIe 成本接近（0.137 与 0.141 ms）时，启动会落在相邻的分工上（`... 3 3 3` 与 `... 3 3 4`） | `STRATA_GLM_PCIE_SHARE` 或 `STRATA_GLM_CPU_PLAN` |
| 重新计时的分工 | `STRATA_GLM_CPU_RETIME` 用解码自己的计时重新规划它 | `STRATA_GLM_CPU_RETIME=0` |
| 层之间的后台搬移 | 它们在拷贝完成时生效 | `STRATA_GLM_PROMOTE=0` |
| 内存层 | 除非设了 `STRATA_GLM_RAM_GB`，按空闲内存定大小 | 设它 |

把这些固定后（RTX 5090、Maya-S-v2-IQ2_XXS、42 GB 内存层、`--expert-cache 3250`、
`STRATA_GLM_PROMOTE=0`、`STRATA_GLM_CPU_RETIME=0`、`STRATA_GLM_CPU_PLAN=001122333`；六个 256 token 的
提示，贪心，五次引擎启动）全部 70 对 token 流都相同，启动时少 300 MiB 空闲显存也相同（对前五次启动
105 对全同）——解码 18.31 tok/s，默认设置时 18.26（五次中位数）。同样少 300 MiB 而用
`--expert-cache auto`，105 对没有一对匹配。默认设置下，70 对启动之间 17 对匹配。

`STRATA_GLM_CPU_LANE=0` 让问题消失：每个专家无论住在哪都由 GPU 的内核计算，所以 token 在每次重启后、
任何内存层下都相同（这里 26 与 42 GB 完全一致），代价是解码速度低 4%（26 GB）到 10%（42 GB）。
两种方式都不能让两个不同的设置互相一致：不同的池、内存层或拆分会给出不同的、同样有效的 token，
另一个构建或驱动也一样。在一台电脑上测量；DFlash2、MTP、多卡、Linux 与 AMD 未尝试。

## 多个请求同时

`--parallel N`（1-8，默认 1）。`./maya.sh --parallel 4`（Windows：`START-MAYA.bat --parallel 4`）会写进
配置的 `"args"`。Maya 随后同时回答至多 N 个请求，其后的排队等待。控制台显示正在生成几个回答，
`/v1/status` 把 N 报告为其并发数。
- **单卡：** 每个请求的一个 token 一起穿过模型。每份稠密权重（注意力投影、共享专家、稠密 FFN 层与
  输出头）为所有请求只读一次，而注意力与路由专家按请求运行。每个请求得到的 token 与它单独运行时
  完全相同（贪心、CPU 通道关：2 个与 4 个同时都验证过）。在从内存流式读取 Maya-S 大部分专家的
  V100 32 GB 上的真实聊天问题：两个同时合计 21.2 tok/s，逐个问 15.0（x1.41），四个 x1.35。每个对话
  带来自己的专家，所以在只装得下一部分的卡上它们共享显存；显存装下的模型越多，竞争越小。全部专家
  都在 GPU 内存里时（带 Maya-S 的 Strix Halo）：两个同时 x1.67，四个 x2.02。
- **两块 GPU 及以上：** 默认每块卡处理一个不同的请求，一个接一个。配 `STRATA_GLM_SPLIT_ROWS` 时，
  每个部分改为批量请求组，多个组在拆分上流水线化。
- **单独一个请求：** 在测试过的四卡 Maya-M 设置上开启 split rows 后，解码与预填充保持在旧路径的
  ±5% 以内。支持推测解码的地方，现有的单请求推测分支优先。

同一主机四个请求时，`k=1` 在逐次波动之内接近旧路径，而 `k=2` 与 `k=4` 降低了吞吐。这些短提示没有
填满配置的 256K 上下文，所以这里测的是解码吞吐而不是满上下文容量。

每个额外请求在显存里保留自己的上下文状态（注意力缓存、循环状态与一份快照），启动时从专家池拿，
所以更大的 N 让显存里的专家更少；日志的 `glm multi:` 行给出大小。N 大于 1 时引擎解码不带
[DFlash2 起草器](#使用-dflash2-起草)（日志会说明）。

## 使用 DFlash2 起草

`./maya.sh --setup --draft dflash2` 下载起草器（0.7 GB，
[在 Hugging Face 上](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF)；其许可证为
CC BY-NC-ND 4.0：仅限非商用）并以 `--dflash GGUF` 写进配置的 `"args"`；NVIDIA GPU，一次一个请求
（`--parallel 1`）。安装器在 RTX 30 / A100 及更新（sm_80）上提供它：在 Tesla V100 上它解码比 MTP 模块
慢，单卡与两卡都是。收益多少也取决于电脑——大部分专家来自内存时，检查草稿可能比省下的更贵
（RTX 3090 + 3060 带 Maya-M，#127）——所以安装器会在那里实测
（[速度校准](#速度校准)）。

为 GLM-5.3-Flash 制作的 DFlash2 起草器——一个小型模块起草器，读取模型五个层的隐状态，一趟起草至多
7 个 token，用模型自己的 embedding 与输出头——取代 MTP 模块：有它时引擎不加载 MTP 模块。每轮它起草
一个模块，一个验证窗口检查草稿，最长的正确前缀连同模型自己的下一个 token 一起保留——与逐个解码
得到相同的 token。

三块 GPU 及以上时窗口是一条流水线：各 GPU 同时处理窗口中相邻的 token（每块 GPU 一个主机线程发出
工作），起草器在窗口的第一个 token 已在运行时起草，每个 token 的取值一出最后一块 GPU 就被读取，
错误草稿之后的 token 跳过它们的工作（它们的专家也不取回、不在 CPU 上算），同时下一轮在其后开始。
最慢的 GPU 给每次起草定节奏，所以拆分搜索在那里平衡的是各 GPU 的时间，而不只是显存
（`STRATA_GLM_SPLIT`，见上）。

单卡上，以及两卡上，窗口改为一次批量趟，为 CPU 通道定节奏的解码而建：它的 token 分两组、层两两
交错（CPU 回答一组的专家时 GPU 跑另一组的注意力），每层的专家为共享它们的 token 读一次、分给两个
CPU 池——单卡时第二个在另一个插槽，两卡时是另一块 GPU 的池——窗口的内存层专家全部在 CPU 上。
两卡每轮至少检查一个草稿，并在置信度乘积保持 0.9 以上时最多检查四个（那里一个两 token 的窗口只花
约 1.25 个单独 token）；单卡检查置信度 0.95 以上的草稿，其下的一个都不查（约 1.55 个单独 token）。

| 参数 | 作用 |
| --- | --- |
| `--dflash GGUF` | 起草器（`STRATA_GLM_DFLASH` 从环境同样设置）；它放在有模型输出头的 GPU 上 |
| `--spec T` | 一轮的窗口至多 T 个 token：该 token 加至多 T - 1 个草稿（默认：起草器的模块大小 8；两卡四个草稿） |
| `--spec-min-p P` | 草稿在第一个选择器置信度低于 P 时停止（三块及以上 GPU 默认 0.5，那里错误草稿代价小；单卡 0.95；两卡改为置信度的乘积，见下） |

请求可以自带窗口与截断（为调参，无需重启）：`"strata_tune": {"spec": 6, "spec_min_p": 0.5}`（还有
`"spec_n_min"`：不论置信度保留草稿；`"spec_cum"`：按置信度乘积截断——两卡默认 1 和 0.9；
`STRATA_GLM_DFLASH_MIN` / `_CUM` 为所有请求设置它们）；`"spec": 1` 不起草解码。

Maya-L，对五个提示的贪心回答——解释、代码、一个列表、一个短故事、关于一份 20,000 字符文档的问题——
单位 tokens/s，每一对背靠背测量。九块 GPU（一张 RTX 3090 与八张 RTX 5060 Ti，CPU 通道开）：

| | 解释 | 代码 | 列表 | 故事 | 文档 | 全部 |
| --- | --- | --- | --- | --- | --- | --- |
| MTP 模块（两次运行） | 43.7 | 44.7 | 46.1 | 38.3 | 42.0 | 42.6 |
| DFlash2（三次运行） | 60.1 | 76.2 | 90.8 | 48.0 | 51.8 | **61.4 (+44%)** |

单张 RTX 3090（单卡不加载 MTP 模块：普通解码），以及 3090 加一张 5060 Ti：

| | 解释 | 代码 | 列表 | 故事 | 文档 | 全部 |
| --- | --- | --- | --- | --- | --- | --- |
| 3090，普通解码（两次运行） | 16.4 | 15.7 | 15.6 | 16.4 | 14.0 | 15.8 |
| 3090，DFlash2（两次运行） | 19.4 | 20.6 | 20.2 | 17.1 | 16.4 | **18.8 (+19%)** |
| 3090 + 5060 Ti，MTP 模块（三次运行） | 20.9 | 21.3 | 21.9 | 16.7 | 19.4 | 19.7 |
| 3090 + 5060 Ti，DFlash2（三次运行） | 23.2 | 24.2 | 25.8 | 19.4 | 19.2 | **22.1 (+12%)** |

### 速度校准

当安装器用 DFlash2 起草时——以及更新后首次启动时，对已经用它起草的配置一次——Maya 在你的电脑上
对照 MTP 模块测量它（单卡：对照不起草解码）并保留更快的。它加载模型两次，同一配置分别不带与带
`--dflash`。每次它读取一份长文档（约 8,000 token，真实对话的深度）并回答一次预热——不计入——然后
回答关于它的新问题，按控制台的方式采样（两次同样的问题与种子），直到写出约 1,500 个 token。
持续速度——所有这些 token 除以写它们的全部时间——决定去留：DFlash2 只有快 5% 以上才保留。需要
5-20 分钟，并逐步显示。结果保存在配置里（`installer.speed_calibration`）；`./maya.sh
--speed-calibration` 重新测量（新 GPU、更多内存），`--draft dflash2 --no-speed-calibration` 不测量
直接保留 DFlash2。
