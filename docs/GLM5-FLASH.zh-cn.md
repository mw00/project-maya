# 在 Strata 上运行 GLM-5.3-Flash——研究

<p align="center"><a href="GLM5-FLASH.md">English</a> | 简体中文</p>

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) 要跑在这个引擎上需要什么，逐组件
分析，附上决策需要的每个数字。写于 2026-09-29，针对引擎 0.1.27（`main`）；所有 `file:line` 引用都指
那棵树。

Strata 不是通用推理引擎：它是单一架构——Qwen3.8-Flash-Next（"qwen4exp"）——的原生实现，其内存分层
（显存专家缓存 / 锁定内存 / 映射固态硬盘）与推测解码栈都围绕这一个形状构建。GLM-5.3-Flash 是同
*一类*模型（3:1 线性注意力/稀疏注意力混合 MoE，带超连接与一个 MTP 层），这就是移植值得研究而不是
直接否定的原因。引擎约三分之二的机制原样适用；剩下的三分之一是混合器（KDA 对 GDN）、全注意力混合器
（MLA 对普通 QKV）、router 打分和打包。

---

## 1. 模型的确切规格

来自发布的 `config.json`（`zai-org/GLM-5.3-Flash`、`Glm5NextForConditionalGeneration`、
`model_type: glm5-next`、MIT、发布于 2026-08-25）：

| | |
| --- | --- |
| 总参数 / 激活参数 | 320 B（321,323,031,390）/ 每 token 18 B |
| 原生权重 | FP8 E4M3，块 128×128，动态激活 scale；norm、gate、router、embedding、`lm_head`、视觉塔与 indexer 保持高精度（BF16/F32） |
| 层 | 45（`hidden_size` 4096）；层模式为 3 个 `linear_attention` 后接 1 个 `deepseek_sparse_attention`，所以全注意力层是 3、7、……、43——**共 11 个**，与 Strata 的 `qsa_interval` 同为 4 间隔 |
| 全注意力（11 层） | MLA，NoPE：64 头，`q_lora_rank` 1536，`kv_lora_rank` 512，`qk_nope_head_dim` 256，`qk_rope_head_dim` 0，`v_head_dim` 256，`mla_use_nope` true。外加一个 DeepSeek 风格的稀疏注意力 indexer：`index_topk` 2048，`index_n_heads` 32，`index_head_dim` 128，`index_kpool` 4（压缩的 index-key 池） |
| 线性注意力（34 层） | Kimi Delta Attention：64 头 × head_dim 128，`short_conv_kernel_size` 4，`gate_lower_bound` -5.0 |
| 超连接 | `mhc: true`，`hc_mult` 4，`hc_sinkhorn_iters` 20——流形约束变体（mHC） |
| MoE | 288 个路由专家 + 1 个共享，每 token top-8，`moe_intermediate_size` 2048，sigmoid 打分，`noaux_tc` 路由，`routed_scaling_factor` 2.5，router 为 float32；**层 0-2 稠密，3-44 稀疏（42 个 MoE 层）** |
| MTP | `num_nextn_predict_layers` 1 |
| 视觉 | 24 块 ViT，hidden 1024，16 头，448 px，patch 14，空间合并 2，输出 4096，投影中间层 10240 |
| 词表 / 上下文 | 154,880 / 1,048,576 |
| 路由专家参数占比 | 288 × 42 × 3 × 4096 × 2048 ≈ **304.6 B 参数 = 模型的 95%** |

GGUF 已经存在：[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
（动态量化；他们的 3-bit 约 120 GB，4-bit 约 180 GB）。llama.cpp 已有这个架构需要的构件（KDA 分块 +
循环内核、NoPE-MLA、来自 DeepSeek V3.2 工作的 DSA lightning indexer、nextn/MTP），这一点意义有两重：
它证明内核是可移植的，而 `third_party/ggml` 正是 Strata 构建其原生路径与 `strata-vision` 所依赖的
固定版本依赖。

## 2. 引擎现在是什么

供参考，已实现的几何（`include/strata/core/layout.hpp:27-59`）：n_embd 2560，48 层，每第 4 层全注意力
（12 个 QSA），GDN 线性层（16 个 k 头 / 48 个 v 头，状态 128，conv 4），QSA 全注意力（24 头，2 个 KV
头，head_dim 256，indexer 4 个 q 头共用一个 128 维共享 key），带 hc = 4 流、320 宽瓶颈的门控残差，
MoE 512 专家 × top-10 × 中间层 640，一个由 28.8 GB n-gram 表支撑的第 1 层 PLE 模块，以及一个 MTP 草稿
层。每层：

    gr_read (attn) -> GDN | QSA -> gr_write (attn) -> gr_read (ffn) -> MoE -> gr_write (ffn)

引擎的单元是一张捕获的每层 CUDA graph，由一个主机循环重放，在门铃（doorbell）之后重叠一个 CPU 专家池
与一个显存专家缓存（`include/strata/core/session.hpp`）。专家住在 `experts.bin`（定长 blob，
`layer * n_expert + expert`），默认由锁定 arena 供给，低内存模式下由 mmap 供给（`--mmap-experts`，
`include/strata/core/expert_source.hpp:303-360`）。稠密权重从 pack 加载进显存。一条"原生 pack"路径直接
从 GGUF 分片运行、带每层运行时几何（`include/strata/kernels/cpu/native_expert.hpp`）——Coder（混合
i-quant 类型，每层 256 专家）就是这么工作的，而 GLM 的 pack 也将走这条路。
## 3. 逐组件差距分析

### 3.1 原样适用

| 部分 | 原因 |
| --- | --- |
| 3:1 层模式 | GLM 的全注意力层是 `layer % 4 == 3`，与 `is_qsa_layer` 相同（`layout.hpp:63`）。45 层 → 11 个 QSA + 34 个线性，且 `n_qsa_layers() = 45/4 = 11` 本来就是正确的算术。 |
| 超连接残差的管线 | `hc_mult` 4 = 已实现的 `hc = 4`。流栈、每半边的读/写结构、BF16 权重契约与融合 GR 快速路径全部沿用。（改变的是*哪些权重*以及一处归一化——§3.4。） |
| 专家内存层机制 | 显存专家缓存、专家画像、锁定、PCIe 重叠、多 GPU 专家拆分、mmap 源——全部在运行时按 `(n_layers, n_expert, blob_bytes)` 参数化。 |
| KV 缓存格式与流式传输 | INT8 / Q4_0 / K8V4、超过 64K 的 KV 流式传输、检查点——全部按 `ModelGeometry` 定尺寸。 |
| MTP 推测协议 | 验证窗口、草稿策略、prompt-lookup 起草器。起草器的*权重*是模型自己的 MTP 层（§3.6）。 |
| 预填充流水线 | 分块 MMQ 预填充、`--prefill auto`——形状来自 pack。 |
| 服务 | OpenAI/Anthropic API、MCP、monitor、免 tokenizer 的提示缓存——本来就是模型无关的。 |
| PLE | GLM 没有 PLE 模块。`PleRun` 已经是可选的（`layer.hpp:545`，"PLE 未接线时为 null"），没有 `blk.1.ple_*` 的 pack 会跳过它——GLM 正该这么做。没有分片 2，没有 29 GB n-gram 表。 |

### 3.2 线性注意力混合器：GDN → KDA（中等内核工作量）

已实现的 GDN（`ref/gdn.py::gdn_decode`，`layer.hpp:8-17`）用每头标量：`alpha`、`beta`、`gate` 是每个
v 头一个 float（`layer.hpp:73-75`），gate 是 `softplus(walpha@x + dt) * ssm_a`。

KDA 在两处有实质差异：

1. **细粒度（每通道）衰减门。** 衰减是每头一个 head_dim 的*向量*（64 × 128），按元素作用于状态，
   而不是每头一个标量。在 llama.cpp 里这是 `linear_attn_config` 的两种衰减门形式，由
   `gate_lower_bound` 选择；GLM 设 -5.0，所以要实现的是 KDA 的下界细粒度门形式。循环状态本身的形状
   （64 头 × 128 × 128 ≈ 每层 1M float，接近已实现的 128 × 128 × 48 ≈ 786K）与短卷积（kernel 4 =
   `d_conv` 4）属于 `gdn.cu` / `native_gdn.cu` / `fused_gdn.cu` 内核已经处理的同一形状家族。
2. **门的下界。** gate logit 上的 -5.0 钳制*在门变成向量之后*是一行改动，但它改变了数值契约，所以
   CUDA 与 `ref/gdn.py` 预言机必须一起改并重新通过 `gdn_parity`。

工作：把分块（预填充）与循环（解码）GDN 内核里的门缓冲与衰减施加扩展到每通道门；加下界钳制；新的
BF16 `ssm_alpha` 类权重（形状从 (48) 变为每 token (64×128)）；更新 `check_layer` 断言。激活契约逻辑
（哪些权重用 Q8_K、哪些用 BF16 激活）是按张量的、本来就是数据驱动的。

### 3.3 全注意力混合器：QSA → NoPE-MLA + DSA（最大的一块）

已实现的 QSA 层是带 RoPE 的普通 QKV 注意力：24 头，2 个 KV 头 × 256，cos/sin 表
（`layer.hpp:239-241`），按头分页的 KV 池（`k_pool` `[page][kv_head][page][dim]`），一个块池化的
indexer（4 个 q 头，一个共享 128 维 key）驱动 top-k 格选择，以及一个每头输出门。

GLM 的 11 个全注意力层需要：

1. **带吸收的 MLA，NoPE。** `qk_rope_head_dim = 0` 意味着这些层里*完全没有 RoPE*——rope 表、`pos_dev`
   旋转与用于注意力的 M-RoPE 位置 id 从路径中消失（M-RoPE 只在视觉路径需要时作为视觉位置编码保留）。
   KV 缓存变成压缩的隐式 `c_kv`（每 token 每层 512 float）——这比已实现的 KV *更便宜*：11 层 × 512 ×
   2 B ≈ **每上下文 token FP16 下 11.3 KB**，约为已实现的 2 × 256 × 2 头 × 2 B = 24.6 KB 的一半，
   128K 上下文时 1.4 GB。两种实现选项：
   * *吸收式：*预计算 `q_hat = W_uk^T q`（每头一个 512 向量）并直接对隐式打分。注意力内核随后读
     一个 512 宽的池而不是每头各池——比今天更简单更快的访存模式，代价是每 token 一个 64 × 512 GEMV
     来构建 `q_hat`。
   * *展开式：*写入时把每头 K（256）与 V（256）物化进现有池布局（隐式字节的 3.2 倍，每 token
     36 KB），原样复用现有解码/验证内核。吸收式是正确的目标（MLA 存在的原因），但展开式是诚实的
     第一个里程碑——它把正确性与内核新颖性隔离开。
2. **indexer。** GLM 的 indexer 是 32 个 q 头 × 128（已实现的是 4 × 128 共享 key），top-k 2048 格
   （已实现的选择宽度本来就近似 2K——`kTopkMaxCells`，"8 queries x 2,051 cells"，`layer.hpp:275`），
   外加 `index_kpool` 4——一个压缩的 index-key 池，其确切形式需要转换后的 GGUF（或 llama.cpp 的 glm5
   转换）作参考。QSA indexer 机制（`native_qsa_indexer`、块打分、池化 key）是正确的骨架；头数与池
   压缩是差异点。
3. **每头输出门**（`v_head_dim` 256，门控注意力输出）——已实现的 `q_full` q|gate 拆分
   （`layer.hpp:367-371`）已有这个形状；只是维度变了。

工作：新的吸收式 MLA 注意力内核（解码 + 预填充 + 验证窗口），或展开式的过渡路径；indexer 重新参数化；
不带 rope 表的 `QsaState`；按类型重写 `check_layer`。这是单个最大的内核工作流，也是数值对等投入集中
的地方。

### 3.4 超连接：HC → mHC（小，但要解决一个问题）

已实现的 GR（`ref/gr.py`，从 llama.cpp 的 `qwen4exp` `build_hc_mix/combine` 转写，`gr.hpp:110-120`）
用每半边权重 `w_norm`（hc·n_embd）、低秩瓶颈 `w_down`/`w_up`（hc_lr 320）与 `w_inject`；写入门是
`2*sigmoid(inject/hc)`。

GLM 的 mHC 加上流形约束：混合矩阵做 Sinkhorn 归一化（`hc_sinkhorn_iters` 20，`hc_mult` 4）。要从第一份
转换的 GGUF 解决的问题是 Sinkhorn 迭代住在*哪里*：

* 如果发布的权重已经归一化（约束在训练期间施加），mHC 就是**打包时的权重预处理步骤**，推理内核与
  今天的 GR 逐位相同。
* 如果转换期望引擎每次前向归一化（config 暴露迭代次数正是因为推理需要它），引擎就需要每层每 token
  在 4×4 矩阵上做 20 次 Sinkhorn——便宜得微不足道（16 个数），但它必须进捕获的 graph，且预言机
  （`ref/gr.py`）必须长出同一步。

无论哪种，这都是*最便宜*的架构差异；风险只在对等纪律。

### 3.5 MoE：形状 + 打分（引擎工作小，有一堵硬墙）

* **形状。** 288 专家 × top-8 × 中间层 2048，对比 512 × 10 × 640。`n_expert`、`k` 与 `n_ff` 全部在
  运行时穿过 `ModelGeometry`，router 的 top-k 早就是参数（`router_top10.hpp:17`，"k 必须 <= 64"），
  门铃/专家分发也显式携带 `k`。
* **稠密层 0-2。** 已实现的栈假设每层都是 MoE。GLM 的前三层是稠密 MLP。`core/native_dense.hpp` 已存在
  （稠密 GEMV 机制）——`block_layer` 需要一个稠密 FFN 层类型给层 0-2，是一个小的结构改动加权重接线。
* **打分。** 已实现的 router 是对全部专家 softmax + 重归一化（`router_top10.hpp:3-5`，
  `ref/moe.py::router`）。GLM 是 **sigmoid 打分、`noaux_tc`、对选中的 8 个重归一化、× 2.5 路由缩放、
  float32 router 权重**。router 内核的一个打分函数变体 + 缩放因子——小，但这是一次*选择语义*的改变：
  对等必须在 router 输出处检查（BF16-GEMV → logits → top-8 边界），因为一次翻转的选择是另一个专家，
  不是舍入误差（仓库自己的注释，`layer.hpp:131-139`）。
* **硬墙：规范 Q2_0 CPU 专家内核。** `include/strata/kernels/cpu/expert.hpp:34-45` 把 H = 2560、
  FF = 640、NE = 512 与 BLOB = 1,382,400 烤成编译期常量，其契约*拒绝*其他形状（"不同的 n_embd 或不同
  的专家宽度会被拒绝而不是错误索引"，`expert_source.hpp:18-21`）。GLM 不走它：**原生路径**
  （`native_expert.hpp`）是运行时几何的（每层一个 `NativeFmt`，带 `n_embd`、`n_ff`、来自 GGUF 的 ggml
  类型），GLM 必须走这条路——正如 Coder 与 Orca 兼容包所做的那样。对研究的推论：GLM 的 pack 是原生
  pack；规范 Q2_0 pack 路径不在移植范围内。
* **专家 blob 大小。** 3 × 4096 × 2048 = 每专家 25.2M 参数 ≈ 已实现专家的 3.2 倍。blob 大小变为按层
  原生（已支持：`layer_offsets_`/`layer_blob_bytes_`，`expert_source.hpp:345`）。按 Q2_0 档速率每 token
  的路由流量 ≈ 336 次读取 × 约 7 MB ≈ 2.4 GB——已实现的 664 MB/token 的约 3.6 倍
  （`layer.hpp:526-527`），这才是真正的性能故事，§5。

### 3.6 MTP、PLE 类附加件、视觉、tokenizer

* **MTP。** 一个 nextn 层，同样的协议。两处修复：起草器的几何被硬编码为规范默认值
  （`static const strata::core::ModelGeometry draft_geometry{}`，`generate.cpp:1726`），必须跟随目标
  模型；Qwen 草稿词表 embed 大小在一个调用点硬编码（`generate.cpp:1374`，248320），必须来自 pack。
  GLM 的 MTP 张量必须在 GGUF 里（llama.cpp 的 glm5 转换携带 nextn 张量），且 `tools/mtp_pack.py` 要
  认识这些张量名。草稿接受率是每个模型的经验问题。
* **视觉。** `strata-vision` 由固定版本的 llama.cpp `mtmd` 构建（`tools/vision/`）。GLM 的 24 块
  448 px ViT 是标准布局；支持归结为固定版本的 llama.cpp 对 `glm5-next` 视觉支持什么 + 一份转换的
  `mmproj` GGUF，外加 `setup.py` 里的家族接线。如果固定版本的 llama.cpp 早于 glm5 视觉支持，这是一次
  子模块 pin 升级加对等测试重跑。
* **Tokenizer。** 打包器从 GGUF 导出模型自己的 tokenizer（规则：永不共用另一个模型的 tokenizer）。
  GLM 的 BPE（154,880 词表）像 Qwen 一样流过 `gguf-py`；验证 pre-tokenizer 正则是 `gguf-py` 认识的那
  一种。服务器需要一份 GLM 聊天模板（`serve/chat_template.jinja` 是 Qwen 的），以及它的图片/视频
  token id（154854/154855）。
* **元数据。** 引擎读取九个 `qwen4exp.*` 键（§6）。必须识别一套 `glm5-next.*`——最小改动在
  `generate.cpp:1466-1480`，它今天只从模型文件读 `expert_count` 与 `expert_used_count`。

### 3.7 量化与 pack

* 源权重是 FP8 E4M3；打包器经 ggml-cpu 消费标准 GGUF 量化（`ref/load.py` 的十二格式表 + 原生路径）。
  unsloth 的 GGUF（Q2_0/IQ2/IQ3/IQ4 动态）是输入；规范的 Q2_0 pack 流水线不需要（§3.5）。
* `--compat-bf16` 路线（`tools/iq_pack.py`，为 ORCA 构建）正是 GLM pack 的高精度投影所需：GLM 的 BF16
  清单与 Qwen 不同（config 的 `modules_to_not_convert`：稀疏层的注意力投影、norm、gate、router、
  embedding、`lm_head`、视觉塔、indexer），所以 compat 清单必须从架构的张量名推导，而不是照抄。
* 按张量的激活契约（`docs/activation-contract.md`，按源类型分 Q8_0 与 Q8_K）已经是数据驱动的——新的
  量化组合直接搭车。

## 4. mmap 研究（内存池问题）

**核心数字：在每一档有用的量化下，仅路由专家就超过 64 GB 内存。**

| 格式 | bpw（约） | 每专家 | 288 × 42 专家 | + 稠密 ≈ | 文件 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2_0 档 | 2.25 | 7.1 MB | **约 86 GB** | 约 93 GB | 约 100 GB |
| IQ2_XS 档 | 2.06 | 6.5 MB | **约 79 GB** | 约 85 GB | 约 95 GB |
| IQ3_XXS 档 | 2.66 | 8.4 MB | **约 101 GB** | 约 108 GB | 约 120 GB |
| 4-bit 档 | 4.25 | 13.4 MB | **约 162 GB** | 约 170 GB | 约 180 GB |

（对比 Strata 自己的 Qwen 数字：23–50 GB 专家*固定在内存里*——"全部专家驻留内存"这一设计假设对消费级
硬件上的 GLM 已经死了。这正是用户点出的问题。）

解码期每 token 流量：8 专家 × 42 层 = 336 次专家读取 ≈ **Q2_0 档速率下 2.4 GB/token**（Qwen 的
664 MB/token 的 3.6 倍）。这些字节从哪里来决定一切：

* 显存缓存（热点专家，按画像排序）：机制不变，命中率经济学相同，但每次命中现在从内存/固态硬盘路径上
  消掉 3.2 倍的字节。
* 内存（页缓存或锁定）：在 128 GB 机器上，约 86 GB 的 Q2_0 档专家可以和操作系统并存——页缓存热的
  mmap 源此时是 DRAM 速度，当前的低内存模式*就是*主模式。在 64 GB 机器上大约装得下专家集的一半；
  其余从固态硬盘换页，一次缺页严重的 token 约需 1.2 GB NVMe 读取 ≈ 100–150 ms——冷启动约
  5–8 tok/s 可用，降级但能用，而随时间压低缺页率靠的正是专家画像。
* 固态硬盘：NVMe 按 blob 粒度（约 7 MB）的随机读取，恰是现有 mmap 源与 PLE 读取器的无缓冲异步路径
  （`direct_file.hpp`）为之而建的访问模式。

引擎已有的应对：`FileExpertSource`（MapViewOfFile/mmap，blob = base + multiply，
`expert_source.hpp:303-360`）、`pin_cache_complement`（内存允许时锁定未被缓存的补集）、带衰减用量换入
换出的显存层、预填充注意力期间下层专家的 PCIe 流式传输、KV 流式传输，以及无缓冲异步文件层。

代码里已经有一条教训，研究予以采纳：**保留率就是全部，随机访问提示是敌人。**
`FILE_FLAG_RANDOM_ACCESS` 曾在专家映射上试过，实测 **1.93 GB/s——磁盘速度，差 14 倍**——因为它告诉
Windows 缓存管理器尽快再丢掉这些页（`expert_source.cpp:332-343`）。专家的访问模式是每层随机但*每个
token 重复*，所以映射刻意不带任何提示。同样的推理排除了 Linux 上的 `MADV_RANDOM`。一个 100 GB+ 的
映射真正需要的是（本分支已实现，§7）：

1. **在 `begin_layer` 处预取被路由的 blob。** 源文件自己的契约就这么说："从磁盘读的源希望从这里开始
   读取，以便与量化重叠"（`expert_source.hpp:98`）。当某层的 router id 已知时，恰好为那 k 个 blob 发出
   `PrefetchVirtualMemory`（Windows）/ `posix_fadvise(WILLNEED)`（Linux），会在内存池量化激活、GPU
   跑完上一层的同时把它们换入——这是预填充路径已有的 PCIe 专家流式传输在 mmap 路径上的对应物。与
   随机访问标志不同，预取是*保留友好的*：它把 router 真正会碰的页拉进待机列表，对其余页的换出不作任何
   表态。
2. **让映射保持无提示。** 在 GLM 的规模（约 86–162 GB 专家）下，用提示"修复"随机访问的诱惑更大，而
   失效模式是同一个实测过的那个。映射保持不加装饰；页缓存按新近度决定保留，这正是一个远大于热点工作集
   的映射文件所希望的。
3. **绝不误触 map-and-pin。** `pin_cache_complement` 仍是可选的，现在也是 128 GB 内存情形的工具（显存
   缓存没装下的补集若能装下就锁定它）；在 64 GB 上它必须保持关闭，必须允许页缓存回收。

## 5. 预期性能区间（估计值，±30 %）

方法：GPU 部分按带宽与显存内专家份额缩放，CPU 部分按剩余专家字节缩放。假设一块 RTX 3090 档显卡
（24 GB）+ DDR5、Q2_0 档 pack：

* 约 3,000 个显存缓存专家覆盖约 70 % 的读取（画像经济学可迁移；每专家字节大 3.2 倍，所以 24 GB 卡约
  缓存 2.5–3 K 个）。剩余约 30 % = 约 720 MB/token 从约 40 GB/s 的 DRAM 来 ≈ 18 ms → 若 CPU 池与
  重叠都成立，**解码上限 ≈ 50–65 tok/s**。
* 在 12 GB 显存 + 64 GB 内存上：约 1.5 K 个专家被缓存（约 55 % 的读取）；缺页在页缓存与固态硬盘之间
  分摊 → 热的约 **25–40 tok/s**，冷的更低。
* 预填充：8K token 分块上的批量专家 MMQ 是计算/权重受限的；专家字节/token 在一个块上摊薄约 800 倍，
  所以同档显卡上**约 1,000–1,800 tok/s** 的预填充是合理的。
* 长上下文比 Qwen 的情形*更好*：MLA 减半的 KV（fp16 下 11.3 KB/token）意味着 128K ≈ 1.4 GB、
  1M ≈ 11 GB——配合 KV 流式传输，1M 上下文在 24 GB + 128 GB 上可行，而已实现的模型会更早饱和。

这些是研究估计，不是实测；对等测试框架（§6）先于任何调优。

## 6. 执行计划

**Phase 0——参考与元数据（本分支）。**
`generate.cpp` 识别 `general.architecture = glm5-next`，在内核工作落地之前*响亮而具体地*失败，而不是在
`check_all` 里死于四十个形状不匹配；§4 的 mmap 改进（它们对现有模型的低内存模式同样有用）；本文档。

**Phase 1——预言机（已完成，本分支）。** `ref/glm.py`：按需 GGUF → FP32（`ref/load.py` 机制），以发布
config 作为元数据默认值对 glm5-next 做一次完整的 teacher-forced 前向；引擎将需要的每个中间量都作为对等
接缝记录在 llama.cpp 自己的 `cb()` 名下（合成模型上 175 个张量），`tools/test_glm_oracle.py` 断言架构
不变量（MoE 权重重归一化、Sinkhorn 的目标随机性、门的取值范围、有限的接缝）。规格已对照
`ref/upstream-llama.cpp/glm5-next.cpp`（已 vendored，llama.cpp @ `272aad8b`）钉死，它也回答了两个悬而
未决的问题：**mHC Sinkhorn 归一化在每次前向内、graph 内运行**（对目标维 softmax、+eps、然后交替的源/目标
归一化——每层每 token 一个 4×4 矩阵，但它是数据相关的，所以引擎必须逐 token 计算而非打包时计算），以及
**k 池 indexer**：对每 4 个连续 indexer key 的池，用逐 key 维的 softmax（对 gate + 可学习的逐位置加性
嵌入）做加权平均，用对预缩放权重做 ReLU 化的多头点积给池打分，每个 query 选 top_k/kpool 个池，并附加不
完整的尾部（发布 config 里 `kpool_select_tail` = true）。同样钉死的还有：KDA 的门是**低秩分解**的
（`f_a` 把 rank head_dim → `f_b`，输出门 `g_a/g_b` 同理），下界形式为 `lb·sigmoid(−g·A)`，且融合的
delta-net 内核在 KEY 轴上施加门并把 `1/√S` 输出缩放烤进去。剩余已知缺口：llama.cpp 尚未实现 GLM5-Next
的 MTP graph（glm5-next.cpp L190 "not implemented yet"），所以草稿层的确切形式仍是 Phase 3 的问题。

**Phase 2——混合器（已启动）。** 两个内核家族已落地，每个都带仓库负例测试纪律的对等测试框架（每种错误
读法都必须先产生实质差异，正确的读法才能被判合格）：

* **mHC**（`include/strata/kernels/glm_hc.hpp`、`src/kernels/cuda/glm_hc.cu`、`glm_hc_parity`）：读半
  部是三个内核——联合栈的 inverse RMS（对四条流做 ONE 次 norm，不是 gr 的每流各一次）、24 行的 mixes
  GEMV、以及一个串行通道：组成三个门、在 4×4 comb 上跑 Sinkhorn（对目标维 softmax、+eps 加到值上、目标
  归一化，然后 (iters-1)×[源归一化, 目标归一化]），并组成 `mixed = Σ_s pre_s·R_s`。写半部是一个逐元素
  内核：`out_dst = block_out·post_dst + Σ_src comb[dst,src]·residual_src`。负例测试钉住联合 norm 对每流
  norm、源主序的 comb 行、softmax 后加 eps、sum 对 mean、以及门前下限（结构 fixture 里把 hc_eps 放大到
  3e-2，因为真实的 1e-6 低于 f32 噪声）。权重暂时是 F32——BF16 输入变体是 Phase 3 的接线决定，以这份
  F32 参考为准。
* **Router**（`include/strata/kernels/router_sigmoid.hpp`、`src/kernels/cuda/router_sigmoid.cu`、
  `router_sigmoid_parity`）：每 token 一个 block，无归约（sigmoid 有界），并列规则复现 router_top10 的
  rank 公式，bias 只施加于*选择*，权重从未加 bias 的 sigmoid 收集、带 2**-14 钳制重归一化、再缩放。负例
  测试钉住加 bias 的权重（noaux_tc 陷阱）、softmax 对 sigmoid、缺重归一化/缩放、以及通过量化 fixture 的
  并列顺序。

* **KDA**（`include/strata/kernels/glm_kda.hpp`、`src/kernels/cuda/glm_kda.cu`、`glm_kda_parity`）：线性
  注意力层的四件套，端到端用 ggml 布局，Phase 3 接线无需转置——因果卷积 + SiLU，历史滑动是独立内核（早
  先 token 的线程还在读旧列时就地滑动是竞态）、低秩门的下界压缩、**递归**（每头一个 block，token 在
  block 内循环：key 轴的行缩放、`u = S^T k` 归约、delta 修正、外积更新、`o = S^T q` 并像 ggml 的内核一
  样烤进 1/sqrt(dv)）、以及归一化加门的输出。负例测试钉住 GDN 的读法：每头标量门、value 轴的门、*没有*
  delta 修正（纯线性注意力——形状全都吻合）、softplus 形式、缺失的输出缩放、以及反序的卷积抽头。
* **带钳制的 SwiGLU**（`include/strata/kernels/glm_ffn.hpp`、`src/kernels/cuda/glm_ffn.cu`、
  `glm_ffn_parity`）：稠密前导层、路由专家与共享专家的激活——`silu(min(gate, limit)) * clamp(up,
  ±limit)`。负例测试钉住单侧的 gate 钳制与"先钳制后 silu"的顺序，它们的差异活在 silu 被压低的尾部且有界
  （fixture 下约 4.5e-3），所以阈值由这些界设定而非猜测。这个内核还定下一条精度契约：`__expf` 随参数缩放
  的 ulp 误差在 |x| ~ 10 处可见，所以这些内核使用精确的 `expf`。

* **DSA indexer + 吸收式 MLA 注意力**（`include/strata/kernels/glm_dsa.hpp`、
  `src/kernels/cuda/glm_dsa.cu`、`glm_dsa_parity`）：池化（对 gate+ape 的逐 key 维 softmax 加权成员平
  均）、打分（对预缩放权重做 ReLU 化的多头点积）、选择（对*可见*池用 router 的 rank 公式——池 p 对 query
  t 可见仅当其最后一个成员 <= t——并附加不完整的尾部），以及注意力本身：每（query, 头）一个 block 对
  512 宽的隐式缓存，q 经 wk_b 吸收，v 在 softmax 后经 wv_b 吸收，1/sqrt(qk_nope) 缩放，padding 排除在
  softmax 之外。负例测试：整池 softmax、无权重池化、无 relu、无尾部、错误的缩放分母。
* **层组合测试**（`glm_layer_parity`）：两个完整的合成模型层（0：KDA + 稠密 FFN；3：DSA + MoE）跑过整个
  内核家族，每个接缝都对 `ref/glm.py` 的原始 dump（`data/glm5-synth-dumps/`）负责。它抓到三个单内核对等
  测试看不到的真 bug：预言机的 conv einsum 给两个通道轴标了不同的名、算出的是垃圾（已修：
  `"tc,tck->ck"`——内核本来就是对的）；router bias 对 token >= 1 会读出其每层 (n_expert,) 分配的边界
  （已修：broadcast）；以及测试自己的数组 KV 读取（`expert_feed_forward_length` 是数组——数组的
  `MetaValue::u` 是 0）。它还把 glm.py 的 hc flat 修成 ggml 的流主序。

Phase 2 状态：**完成**——全部七个测试套件在 Mercury 上通过（glm_hc、router_sigmoid、glm_kda、glm_ffn、
glm_dsa、file_expert_source、glm_layer_parity）。最后一个组合 bug 是层测试自己对 down-exps 的读取：
ffn_down_exps 是 {n_ff, n_embd, n_expert}，所以 ne0 是*输入*维，down_e[o, v] 位于
v + n_ff*o + n_ff*n_embd*e（gate/up 的转置，后者的 ne0 是 embedding 轴）。

Phase 3 状态：**Glm5Model 流式 runner**（`src/core/glm_model.cu`）已完成并验证——`glm_model_test` 在
一个 12-token 序列的每个位置把它对到预言机（批量对预言机 2e-3、全部 8 个前缀 argmax、流式对批量
1e-5），贪心链从 dump 推导而非写死常量（写死的 Phase-1 链在预言机的 conv einsum 修复后过时了——四次独立
评审都收敛到这个诊断），并检查位置 8–11 的流式续跑。**8/8 套件在 Mercury 上通过，GPU 时间 2.1 秒。**
打包器（`iq_pack.py`）识别 `glm5-next` 并携带按架构的 BF16 保留清单（mHC 的 fn/base/scale 名、glm5 的
indexer 集合、router + noaux_tc bias、KDA 的门），3-D 张量按 (ne0*ne1, ne2) 视图逐字节精确打包；
`strata_tokenizer` 实现了 glm4/glm5 的 pre-tokenizer 方案（对 llama-vocab.cpp 的逐字 CHATGLM4 转写：数
字串每组至多 3 位、字母/标点类中不含 `\p{M}`、`ignore_merges` 的全词表命中发射），并且仍然拒绝它没有转
写过的方案。pack 导出携带逐方案的 pattern 与 ignore_merges 标志。驱动的门现在指向已验证的 runner。
2026-10-01 在全新 Mercury 提取上重新验证：8/8 套件，GPU 时间 2.0 秒。Phase 4 剩余：生产驱动路径的量化
内核接线、安装器的 FAMILIES 条目（刻意推迟——它会宣传一个驱动仍拒绝服务的模型），以及 serve/measure——
负责人已下令**先做 IQ1_M 量化**，端到端跑通后再向上优化；§3.5 的硬墙适用（GLM 的 pack 是原生 pack；规范
Q2_0 专家内核的编译期几何不适合 GLM）。llama.cpp 没有 GLM5-Next 的 MTP graph；草稿层保持推迟。

**Phase 3——打包与安装。** `glm5-next` GGUF → 经 `tools/iq_pack.py` 配 GLM 的 compat-BF16 清单生成原生
pack；MTP pack；tokenizer 导出；`setup.py` 里的 `FAMILIES["glm53"]`（安装器的家族表，
`setup.py:80-104`，就是一个 dict 条目）；GLM 聊天模板；引擎元数据键（`glm5-next.embedding_length`、
`.block_count`、`.expert_count`、`.expert_used_count`、注意力头数）读入 `ModelGeometry`。

**Phase 4——服务与测量。** 对 glm5 mmproj 的视觉辅助；对预言机按残差阶梯（`dump_layers`）的端到端对等；
1K–1M 的针测试；基准矩阵；然后基准表加一节 GLM。

粗略工作量权重：Phase 2 ≈ 60 %，Phase 1 ≈ 15 %，Phase 3 ≈ 15 %，Phase 4 ≈ 10 %。关键路径穿过 MLA 注意
力内核及其对等测试框架。

## 7. 本分支包含的内容

* `docs/GLM5-FLASH.md`——本研究。
* `FileExpertSource` 里的 §4 mmap 工作：从现有 `begin_layer` 钩子按层预取被路由的 blob（Windows 的
  `PrefetchVirtualMemory`；Linux 的 `posix_fadvise(WILLNEED)`），尽力而为且保留友好——映射本身出于上述
  实测原因保持无提示。附测试。
* 驱动里 §6 的 Phase-0 架构门。
* 干净克隆验证发现的一个构建修复：`native_mmvq_multi` 与 `hit_cpu_order_parity` 现在像它们的兄弟一样受
  EXISTS 保护——它们的 bench 源码不在仓库里，而 `STRATA_BUILD_TESTS=ON` 时 generate 步骤会完全拒绝运行。
* Phase 1：`ref/glm.py`（glm5-next 预言机）、`tools/glm_synth_gguf.py`（它与内核对等测试所跑的合成模型）、
  `tools/test_glm_oracle.py`（7 个不变量测试，无需 GPU），以及 `ref/upstream-llama.cpp/`（vendored 的
  llama.cpp 规格，@ `272aad8b`）。
* Phase 2 完成：五个内核家族——mHC（`glm_hc_*`）、sigmoid router（`router_sigmoid_*`）、KDA
  （`glm_kda_*`）、带钳制的 SwiGLU + MoE 尾部（`glm_ffn_*`）、DSA indexer + 吸收式 MLA 注意力
  （`glm_dsa_*`）——每个都带对等测试框架，外加 `glm_layer_parity`（完整层组合）与 `data/` 下的二进制
  fixture。
* Phase 3：`Glm5Model` 流式 runner（`src/core/glm_model.{hpp,cu}`，由 `glm_model_test` 对到预言机）；
  `iq_pack.py` 的 glm5-next 支持（`tools/test_iq_pack_glm.py`）与 `strata_tokenizer` 里的 glm4/glm5
  pre-tokenizer（`tools/test_tokenizer_glm.py`）。

关于在 2x V100 机器（Mercury）上跑内核对等测试的说明：它的 GPU 是 V100（sm_70），*低于*引擎的 sm_75
最低要求，所以预构建的 arch 产物是 V100 无法启动的二进制（"no kernel image"）。内核源码本身是 arch 可移植
的（sm_80 特性都有 fp32 回退），所以本地测试树把 configure 守卫放宽一行
（`sed -i 's/_base LESS 75/_base LESS 70/'`）并以 `-DCMAKE_CUDA_ARCHITECTURES=70` 配置——这是一处不提交、
仅存在于构建树内的改动，只为让 V100 能执行对等套件。

### 8.1 偏移宽度审计（第二意见第 7 项，2026-10-03）

glm5 内核与 runner 计算的每一个索引乘积都对照*真实*维度（n_embd 4096、n_ff 12288/2048x288、1M 上下文）
检查过。结论，没有一个是 bug：

* 宿主侧 arena 寻址端到端是 int64：区域向量（`kda_S_`、`dsa_lat_`……）、每个 scratch 偏移、以及专家 slab
  基址（`ffn_*_exps + slab*e`，2.4e9 个元素超过 INT32_MAX）全部用 int64/size_t。
* 设备侧索引凡乘积会增长处都是 `(size_t)`（隐式缓存行 `kv_lora*cell` 在 1M 上下文时 = 5.4e8；q_absorb；
  gemv 行）。任何内核在 32 位 `int` 局部量里形成的最大乘积受单次调用的激活大小限制——runner 是单 token、
  对等测试框架是 8 token——最多约 2e7，比 2**31 低两个数量级。
* 每层 22 亿元素的专家张量从不作为内核内部偏移出现：内核只看到一个专家的 slab（840 万个元素），层基址由
  宿主以 int64 加上。

审计的边界：引擎自己的 ggml 侧内核（native_expert、稠密 GEMV 路径）不在此处范围内——它们已经在跑
qwen4exp 的生产模型。

### 8.2 大合成模型与 llama.cpp 交叉验证（第二意见第 7 项，2026-10-03）

`tools/glm_synth_gguf.py --big`（seed 777）构建第二个 fixture：45 层按发布的节奏（DSA 在 il%4==3，层
3..43），n_embd/q_lora/kv_lora/n_ff_exp 都是 256 的倍数（IQ 的行宽），32 选 top-8，kpool 4 配 top_k
256（一个 512-token 序列跨过 128 个池并丢掉一半），sinkhorn 迭代 5，热点 FFN 缩放（SwiGLU 钳制触发）与
+8 的 dt 尖峰（KDA 的门触到下界）。tools/cover_audit.py 在它上面通过——第一个 fixture 留零的每个分支在这
里都会触发。

交叉验证：llama.cpp（master，CPU）流式跑同样的 512 个 token；f64 预言机对它们做 teacher-force。结果，
全部 512 列：

  * 每列最大 |logit 差|：p50 9.0e-05，p99 2.5e-03（logit 标准差 0.8）；
  * softmax 上的 KL(oracle || llama)：p50 2.0e-09，p99 1.7e-07；
  * argmax 一致 510/512；*两处*翻转都能用 MoE 选择的近并列解释——rank-8 与 rank-9 的加 bias 得分差为
    1.2e-5（第 108 列）与 1.7e-6（第 250 列），在 f32 对 f64 的噪声之内，所以两个实现各自合法地选了不同
    的第 8 个专家。这正是第二意见第 2 项的机制原样重现，也是为什么 Phase-4 的验收门是 top-1 一致率 +
    均值/p99 KL，而不是逐 logit 的 2e-3 容差。

在这对上标定的 Phase-4 验收阈值：KL p99 < 1e-5，top-1 一致率 > 99%，且每处分歧都能追溯到约 1e-4 以下的
选择边界差。272aad8b 这个 pin 不能用于交叉验证（它的 graph_reserve 会因 scatter 死槽 bug 中止，master 已
修）；Mercury 上的 /tmp/llama-ref 存有 master 构建 + tools/llm_logits_dump.cpp。

### 8.2.1 真实模型交叉验证（UD-IQ1_S 对 llama.cpp，2026-10-04）与量化验收门

§8.2 的门是在 F32 对 F32（llama 跑 F32 fixture）上标定的。真实模型交叉验证让 llama.cpp（上游 master，
CPU）对 pack（设备 MMVQ 稠密 + CPU IQ 专家）在*同一份量化 GGUF* 上对跑——两边都是量化的，内核不同。在
token 1..8 上的结果（用 GLM_CB_DIR 对 llama 的 LLM_CB_DIR 做接缝级二分，完整配方在 NEXT-STEPS.md）：

  * 第 0 层：每个接缝都在 3e-3 内（fp 噪声）；router 链（logits → 加 bias 概率 → top-8 → 重归一化权重）
    在被探测的层上*相等*；MoE 选择从不翻转；
  * 首个分歧：L3 ffn_out（第一个路由专家层），平滑增长；L18 的 ×68 状态爆炸（真实模型行为，两边都有）把
    它放大；
  * 每专家的 h/down 对一个精确反量化 + fp32 参考：我们 3.0–4.3e-2，llama 1.5–4.5e-2——*同样忠实*；差距
    是各自的激活量化误差（我们在设备稠密路径上 q8_1，llama 在 CPU 上 Q8_K），在一个本就略有差异的流的每
    次重量化上重新掷骰；
  * 汇总（8 个 token）：top-1 5/8，p50|Δlogit| 0.31–1.29，cos 相似度 0.70–0.97——形状跟随目标分布的尖锐
    度（第 5–7 位回升到 0.95），而不是累积性 bug。

结论（负责人 2026-10-04 接受）：没有结构性分歧；差异是累积的量化内核噪声。**量化模式验收门**（用于同一
份量化 GGUF 上 pack 对 llama 的比较——上面的 F32 门*不适用*）：

  * 每个 argmax/logit 有限、同一词表、穿过全部主干层无 NaN；
  * 每个 teacher-forced 位置上 p50|Δlogit| < 1.5 且 cosine(logits) > 0.65；
  * MoE 选择（top-8 id）在每个被探测层上相等；专家计算与精确反量化 + fp32 参考的距离在 llama 的 2 倍以内；
  * 无位置漂移：分歧*不得*随位置单调增长（状态/缓存 bug 会；噪声会在分布变尖时恢复）。

改善种子属于 M2 邻近项（负责人决定：并入 M2）：把设备稠密路径切到 Q8_K 风格的激活量化能压低对 llama 的
分歧种子，代价是重做每个 MMVQ 调用点——留到该路径反正要重建的地方再定（IQ1_S 内核、专家分层）。

### 8.3 tokenizer 差异（第二意见第 5 项，2026-10-03 关闭）

真实的 GLM-5.3-Flash tokenizer.json（154,856 个 BPE token + 36 个 special，321,649 条 merge；CHATGLM4
的 pre-regex 逐字出现，含 `(?i:'s|'t|...)`）现在是一个被接受的 fixture：`tools/vocab_gguf.py` 把它写成
仅词表的 GGUF（用 llama.cpp 自己的 gguf-py GGUFWriter 写——我手写的 writer 在大字符串数组上会和 llama 的
reader 失步），`tools/tokenizer_diff.py` 把 `llama-tokenize` 与 `strata_tokenizer` 逐个 id 对比。

一个真 bug：ignore_merges 的全词表查找跑在*原始* pre-token 片段上而不是字节*映射后*的片段——对 ASCII 是
恒等（此前每个测试都过），对任何非 ASCII 的精确词表片段是错的。已修成与 llama-vocab.cpp L634 一致（先映
射，再查找）。

结果：在边界语料（504 个 id）及其 60 倍重复（30,360 个 id）上完全一致——数字、组合符号、CRLF（测试框架
必须以 newline="" 读）、CJK、emoji、缩写、special、字节边界。转写在 id 级别得到验证，而不只是结构上。

### 8.4 runner 吃下大 fixture（2026-10-03）

`glm_model_big_test`（一个 ctest fixture：tools/glm_big_fixture.py 把 1.2 GB 的 seed-777 GGUF 生成进构
建树；f64 期望值提交在 data/glm5-big-dumps/ 下）把 llama.cpp 交叉验证用过的同样 512 个 token 流式跑过
CUDA runner：

  * 流式对批量在前缀 100/250/511：逐位 0；
  * 状态确定性（A、reset、B、reset、A）：逐位一致；
  * 对 f64 预言机跨全部 512 列：**KL p50 2.0e-9、p99 有 5.1e-11 的上界、最大 6.8e-9，argmax 512/512，
    中位列差 1.6e-5，没有一列超过 KL 1e-4**——也就是说，在真实节奏的几何下、丢弃/钳制/门饱和都在 runner
    内触发的情况下，runner 比 llama.cpp 更贴合预言机（llama 的 2 处翻转是它的 f32 舍入，runner 的 f32 顺
    序恰好与 f64 预言机的每个 argmax 相符）；
  * `glm_model_test` 补上顾问指出的两个缺失门：状态确定性与扰动敏感性（文件中段 4 KB 的权重损坏必须让
    logits 移动 > 1e-2，证明门 1 抓得住一个错的模型）。

套件：10/10（此前 8 个 + fixture 生成器 + 大测试）。一条流程注记：提交时的期望文件一度被*转置*（双重转
置）——门立刻抓到，这正是系统按设计在工作。

## 9. Phase 4——源文件与接线计划（2026-10-03 开启）

源：unsloth/GLM-5.3-Flash-GGUF 的 **UD-IQ1_S**（3 个分片，93.0 GB；sha256 已记录），按负责人指示下载到
系统 NVMe（HDD 与第二块 NVMe 的流式速度不足以支撑 mmap 服务；第二块 NVMe 只做暂存）。负责人把 Qwen 的
IQ3_S 量化（79 GB）移到第二块 NVMe 腾出空间。

### 9.1 该文件的实际配方（从分片 2 的张量目录解析）

* token_embd / output：**Q4_K**；稠密前导层 ffn_gate/up **Q5_K**，ffn_down **Q6_K**；attn_output
  **Q5_K**（blk.11 为 Q6_K），attn_q_a Q5_K/Q6_K；
* 吸收式 MLA 投影 + 整个 indexer 注意力集合：**Q8_0**；
* ffn_gate_inp、indexer.proj、indexer_compressor_ape：**F32**（恰是打包器的 F32 集合）；
* 路由专家，*逐层动态*：每层的 gate 与 up 总是共用一个类型（IQ1_S 或 IQ2_XXS），down 是 IQ3_XXS（blk.11/12
  为 IQ4_XS）——所以 pack 每层单一 gu_type 的设计成立；类型是 **IQ1_S、IQ2_XXS、IQ3_XXS、IQ4_XS**——
  *不是*研究 §3.5 假设的类型（gate-up 为 IQ1_M/IQ2_XS/IQ2_S/IQ3_S，down 为 Q2_0/IQ4_NL）。顾问的"把原生
  路径扩展到源的类型"因此得到了最好的可能答案：`native_fmt` 由 ggml-cpu trait 驱动，所以四种类型开箱可用
  （ggml-cpu 对每一种 i-量化都有 vec_dot）；不需要新的反量化器。

### 9.2 IQ1_S-first 里程碑：pack 支撑的 Glm5Model

保留已验证的 runner 数学（mHC/KDA/DSA/hc 内核），教会它服务 pack：

1. 加载 index.txt + native_experts.txt；在加载时把稠密张量反量化进现有 F32 设备 arena（仅
   token_embd+output 就是 2×2.5 GB 的 F32——与约 24 GB 未动的 V100 显存并存放得下）；
2. 让 MoE 循环走引擎的 CPU 原生专家路径（FileExpertSource mmap + 预取、native_gu_rows/native_down_rows），
   而不是设备 gemv 的 F32 slab；
3. 修正 kNativeActBytes/kNativeHBytes：它们按 qwen4exp 的 2560/640 宽激活定尺寸；GLM 需要 4096/2048
   （Q8_K 约 4.2 KB / 约 2.1 KB）——不改就是硬溢出；
4. 上述能服务之后，在驱动的门里解除对 glm5-next 的拒绝；FAMILIES 条目放最后。

完整的生产驱动 graph 路径（捕获的 graph、服务器、layout.cpp 里的 glm5-next WeightTable）仍是后续里程碑；
layout 表从张量名开始就是 qwen4exp 专用的。
