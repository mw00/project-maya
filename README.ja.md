<p align="center"><a href="README.md">English</a> | 日本語 | <a href="README.zh-CN.md">简体中文</a> | <a href="README.pt-BR.md">Português</a></p>

<h1 align="center">Project Maya</h1>

<p align="center"><b>GLM-5.3-Flash (321 B)、DeepSeek-V4.1-Flash-1017 (748 B)、MiniMax-M3.1-Flash (428 B) を自分の GPU で</b><br>
プライベートで高速 · NVIDIA GPU 1 枚から最大 16 枚まで · AMD と Windows (実験的) · チャット、画像、OpenAI / Anthropic 互換 API · 複数リクエストの同時処理</p>

<p align="center">
<a href="https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/glm.svg" alt="GLM-5.3-Flash Maya downloads"></a>
<a href="https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/glm-m-derisked.svg" alt="GLM-5.3-Flash Maya-M-Derisked downloads"></a>
<a href="https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/deepseek.svg" alt="DeepSeek-V4.1-Flash Maya downloads"></a>
<a href="https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/minimax.svg" alt="MiniMax-M3.1-Flash Maya downloads"></a>
</p>

<p align="center">作成・主導: <b>Peasant Smith</b> (<a href="https://github.com/mw00">@mw00</a>)、共同メンテナー: <b><a href="https://github.com/needmorevram">@needmorevram</a></b></p>

<p align="center"><a href="https://github.com/sponsors/mw00"><img src="https://img.shields.io/badge/Sponsor-mw00-ea4aaa?logo=githubsponsors&logoColor=white&style=for-the-badge" alt="Sponsor on GitHub" height="50"></a>
<a href="https://buymeacoffee.com/peasantsmith"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Project Maya は無料です。お使いの PC で快適に動くなら、スポンサーになっていただくと、テスト用ハードウェアと開発の費用になります。</sub></p>

この規模のモデルは、普通ならデータセンターが必要です。Maya は、すでにお持ちのハードウェアでその 3 つを動かします。
**GLM-5.3-Flash**、**DeepSeek-V4.1-Flash-1017**、**MiniMax-M3.1-Flash** です。モデルごとに専用のエンジンがあります。会話でよく使われるエキスパートを
GPU に、次に使われるものを RAM に、残りを NVMe SSD に置き、チャットの進行に合わせて入れ替えます。GPU、CPU、RAM、SSD を測定して自動でチューニングします。
データがお使いのマシンの外に出ることはありません。

- **高速:** RTX 5090 2 枚 + RTX 4090 で最大 149 トークン/秒、H100 2 枚で 124、RTX 4090 4 枚で 118、RTX PRO 6000 1 枚で 81、RTX 5090 1 枚でも 30 以上。[実際のユーザーによる実測値](#how-fast-is-it)。
- **元のモデルに近い品質:** Maya 独自の GLM-5.3-Flash 量子化は FP8 モデルのゼロショット精度の 97.7-99.2% を保ちます。DeepSeek V4.1 · Maya-S は元のモデルと同じ次トークンを 92.6% の確率で選び、MiniMax M3.1 · Maya-S は 87.6% (カテゴリ平均) です。
- **お使いのツールにそのままつながる:** ツール呼び出しに対応した OpenAI / Anthropic 互換 API で、コーディングエージェントにすぐ使えます。
- **複数リクエストの同時処理:** GLM-5.3-Flash、DeepSeek-V4.1-Flash、MiniMax-M3.1-Flash で最大 8 件をまとめて処理します。複数セッションを動かすエージェントや、1 台の PC を複数人で使う場合に向いています。
- **すべて同梱:** ブラウザ上のチャットとライブモニター、すべてのモデルで画像、思考レベル、モデルのダウンロード・読み込み・削除ができる Models ページ、Turbo ボタン、ワンコマンドのセットアップ。

## <a name="quick-start"></a>クイックスタート

```sh
git clone https://github.com/mw00/project-maya.git && cd project-maya
./setup.sh
```

セットアップはどのモデルを使うかを尋ね、ダウンロードし、すべてを一度ビルドして、**モデルを起動します**。その後 `http://127.0.0.1:8080` を開いてください。Windows では `./setup.sh` の代わりに `START-MAYA.bat` を使います。詳しくは[インストール](#install)を参照してください。

## <a name="the-models"></a><a name="deepseek-v41-flash-1017-new"></a><a name="minimax-m31-flash-new"></a>モデルを選ぶ

Maya の量子化はすべて、モデル名のあとに階層 (XS、S、M、L。エキスパートが何ビットを保つかで分かれます) が付いた名前です。セットアップが PC に合ったものを勧めます。自分で選ぶ場合は、セットアップ ID を指定します: `./setup.sh --model glm-maya-s24`。

| モデル | Maya の量子化 · セットアップ ID | サイズ | 想定する用途 |
| --- | --- | ---: | --- |
| **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**<br><sub>zai-org · MIT · 321 B、アクティブ 18 B · 最大 1M コンテキスト</sub> | **Maya-S** (デフォルト) · `glm-maya-s` | 96.5 GB | メモリ (RAM と VRAM の合計) が小さめの PC |
| | Maya-S24 · `glm-maya-s24` | 94.7 GB | 24 GB 以下のカード: カードの約 1.5 GB 分をエキスパートに多く回せ、デコード +14% |
| | Maya-XS · `glm-maya-xs` | 73.4 GB | RAM 64 GB と 24 GB のカードの PC、128K コンテキスト |
| | Maya-M · `glm-maya-m` | 116 GB | より大きなメモリ; ツール呼び出しとフロントエンドのコード向けにキャリブレーション |
| | Maya-M-Derisked · `glm-maya-m-derisked` | 116 GB | [Blackfrost_AI](https://x.com/Blackfrost_AI) による重み編集で、一律の拒否を減らした Maya-M; 実験的 |
| | Maya-L · `glm-maya-l` | 156.3 GB | 最大クラスのメモリ; 元のモデルに最も近い |
| **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**<br><sub>DeepSeek · MIT · 748 B</sub> | **Maya-S** (デフォルト) · `deepseek-maya-s` | 340 GB | メモリ量を問わず: VRAM と RAM に収まらない分は SSD から読み込む |
| | Maya-XS (v2) · `deepseek-maya-xs` | 277 GB | 128 GB の PC のメモリにモデル全体を載せる (Engram テーブルは SSD に置く); v2 は v1 より元のモデルに約 12% 近く、書き出しが約 20% 速い - セットアップを再実行すると更新を案内します |
| **[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)**<br><sub>MiniMax · [MiniMax Community License](LICENSE.minimax-m31) · 428 B、アクティブ 23 B · 最大 1M コンテキスト</sub> | Maya-XS · `minimax-maya-xs` | 93 GB | 128 GB の PC のメモリにモデル全体を載せる |
| | Maya-S · `minimax-maya-s` | 122 GB | 元のモデルにより近い |

いずれも NVIDIA の V100 / RTX 20 以降、および AMD と Windows (実験的; [必要なもの](#what-you-need)を参照) で動きます。DeepSeek のサイズには Engram テーブルが含まれます。セットアップでは、unsloth の GGUF と LibertAIDAI の DeepSeek 用 EXL3 量子化、orcarouter の abliterated 版 DeepSeek、Mia-AiLab の MiniMax 用 EXL3 量子化、turboderp の GLM 用 EXL3 量子化、そして画像対応の Q5_K_M GGUF としての Qwen-3.8-Flash-Next (CYBER-FROST) ([その設定](docs/SETTINGS.md#qwen-38-flash-next-cyber-frost)) も選べます。**各モデルのファイル、ドラフター、画像、オプション: [docs/MODELS.md](docs/MODELS.md)。**

## <a name="quality"></a>品質

Maya の量子化は、各モデルの公開された重みから作られ、元のモデルと比較して測定されています。

- **GLM-5.3 · Maya の量子化**は FP8 モデルのゼロショット精度の 97.7-99.2% を保ち、次トークンを 83-90% の確率で一致させます (Maya-L: 99.2%、90%)。作り方と測定方法: [MAYA-S](bench/results/MAYA-S.md)、[MAYA-S24](bench/results/MAYA-S24.md)、[MAYA-M](bench/results/MAYA-M.md)、[MAYA-L](bench/results/MAYA-L.md)。
- **DeepSeek V4.1 · Maya-S** は元のモデルと同じ次トークンを 92.6% の確率で選びます。
- **MiniMax M3.1 · Maya-S** は 87.6% です (カテゴリ平均)。

各モデルの詳細は [Hugging Face](https://huggingface.co/peasantsmith) のモデルカードにあります。

## <a name="how-fast-is-it"></a>どれくらい速い?

ユーザーが実機で測定した結果です。多くは `./maya.sh --bench` によるものです (デコード: 貪欲法で 256 トークンの回答を 3 回; プリフィル: 行に記載がなければ 8K トークンのプロンプト)。ご自身の結果は `--bench` のあと `--report` を実行して、[GitHub の issue](https://github.com/mw00/project-maya/issues) で共有してください。

<details><summary><b>GLM-5.3-Flash の速度ベンチマーク</b></summary>

| マシン | モデル | デコード (回答の書き出し) | プリフィル (プロンプトの読み込み) | 報告者 |
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

<details><summary><b>DeepSeek-V4.1-Flash の速度ベンチマーク</b></summary>

| マシン | モデル | デコード (回答の書き出し) | プリフィル (プロンプトの読み込み) | 報告者 |
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

<details><summary><b>MiniMax-M3.1-Flash の速度ベンチマーク</b></summary>

| マシン | モデル | デコード (回答の書き出し) | プリフィル (プロンプトの読み込み) | 報告者 |
| --- | --- | ---: | ---: | --- |
| **RTX PRO 6000 Blackwell 96 GB** (PCIe 5.0 x16), Ryzen 9 9950X3D, 253 GB RAM, Windows 11 | Maya-S, 256K context, DSpark | **48.6 tokens/s** | 757 tokens/s | @Andrey0710 (v1.1.43, #738) |
| **RTX 3090 24 GB + 8x RTX 5060 Ti 16 GB** (PCIe 3.0 x8 each), 183 GB RAM | EXL3 3.5 bpw, tensor parallel 4 + DSpark, 8K context | 38.2 tokens/s (code), 32.4 (prose) | - | @needmorevram (#477) |
| **1x L40S 48 GB** (vGPU), Xeon Platinum 8580 (a VM, 32 vCPUs), 504 GB RAM | Maya-S / EXL3 3.55 bpw, 256K context | 23.4 / 16.7 tokens/s | 332 / 273 tokens/s (158K) | @talisp (v1.1.32, #577) |
| **2x RTX 3090 24 GB** (power-capped at 230 W), EPYC 7543 (a VM, 32 vCPUs), 224 GB RAM | Maya-S, 200K context, DSpark | 22.0 tokens/s (code), 18.4 (prose) | 163 tokens/s (10K) | @noonghunna (v1.1.30, #549) |
| **AMD Radeon 8065S** (Ryzen AI Max+ PRO 495, 192 GB, 160 GB of it for the GPU), Windows 11 | Maya-S, 128K context, DSpark | 21.0 tokens/s | 139 tokens/s | @tobitege (v1.1.37, #663) |

</details>

1 トークンはおよそ単語の ¾ です。会話が長くなっても速度は保たれます。起動直後の最初の回答は、使うものでキャッシュが埋まるまで最も遅くなります。

## <a name="using-it"></a>使い方

- **ブラウザで:** チャット (ブラウザに保存され、検索、ピン留め、フォルダ、チャットごとの指示と思考レベル、編集と再生成、HTML プレビューに対応)、モデルとハードウェアのライブ**モニター** (各 GPU のメモリの内訳まで)、設定で変えられるコンテキストサイズとポート、そして **Models**: このマシンにセットアップ済みのすべてのモデル、別のモデルへの切り替え (Maya がそのモデルで再起動します)、各モデルのエンジン設定 (コンテキスト、GPU、ドラフト、画像、メモリ)。設定は `maya-<model>.json` に保存されます。
- **Models ページ:** カタログに Maya のすべてのモデルがサイズ付きで並びます。進捗バー付きでダウンロードし、読み込み、実行中のモデルをアンロードし、削除できます。コマンドラインは不要です。
- **Turbo:** モデルのパラメータの横にあるボタン 1 つで、そのモデルの最速設定になります (貪欲サンプリング、思考なし、エキスパートのスキップ、最速のドラフト)。品質を少し犠牲にして速度を取る設定で、オフにすると元の設定にそのまま戻ります。
- **保存:** サイドバーの Saved の行には、同じトークン数をモデル自身の API (DeepSeek、Z.ai、MiniMax) で処理した場合の料金が合計されます。Maya がキャッシュから再利用したプロンプトは API のキャッシュ価格で、DeepSeek のピーク時間帯はピーク価格で計算されます。モデルごとに `token-savings.json` に保存され、価格は **Settings › Savings** で変更できます。
- **アプリとコーディングエージェント:** `http://127.0.0.1:8080/v1` に OpenAI 互換 API、`http://127.0.0.1:8080/v1/messages` に Anthropic の API があります (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`)。ツール呼び出しはモデルが書くそばからストリーミングされ、長いプロンプトの読み込みの進捗もリクエストで指定すれば届きます (`"return_progress": true`)。別のアドレスで配信されている Web ページも、そのアドレスを許可すれば呼び出せます (`"cors_origins"`、[docs/SETUP.md](docs/SETUP.md#web-pages-on-another-address))。
- **Responses API:** `/v1/responses` は、テキストと画像の会話、ストリーミング、関数ツール呼び出し、スキーマ検証付き出力、キャンセルと再生に対応したバックグラウンドタスク、任意の共有インメモリ履歴 (`store:true`、`previous_response_id`) に対応しています。レスポンスはデフォルトでは保存されません。オプションと制限: [docs/RESPONSES.md](docs/RESPONSES.md)。
- **モデル API:** `/api/tags`、`/api/ps`、`/api/show`、`/api/chat`、`/api/generate`、`/api/pull`、`/api/delete` で、あらゆるアプリやスクリプトからモデルの一覧、読み込み、ダウンロード、削除ができます。読み込まれていないモデルへのリクエストはそのモデルに切り替わり、`keep_alive` を指定すると、アイドル時にアンロードされます。`/models/load`、`/models/unload`、`/models/sse` もデフォルトで有効です (`"model_management": false` で無効にできます)。
- **思考:** Off、Low、High (デフォルト)、Max を、設定またはリクエストごとに指定できます。さらに**思考予算**の欄で、モデルが回答の前に思考に使えるトークン数を設定できます。
- **ループガード:** デフォルトで有効です。同じツール呼び出しを同じ結果で繰り返すエージェントを、その繰り返しから外へ誘導します。
- **画像:** チャットに添付するか、どちらの API でも画像パートを送ります。すべてのモデルが画像を読めます (モデルごとのセットアップの詳細: [docs/MODELS.md](docs/MODELS.md))。
- **ほかのデバイスから:** `./maya.sh --setup --host 0.0.0.0 --api-key <secret>`。キーは必ず設定してください。
- **チューニング:** `./maya.sh --calibrate` が 10-15 分でエンジンをお使いのハードウェアに合わせます。すべての設定: [docs/SETTINGS.md](docs/SETTINGS.md)。
- <a name="deepseek-engram-prompt-read-ahead"></a><a name="deepseek-conversations-kept"></a><a name="deepseek-loading-the-experts"></a>**各モデル固有のオプション** - GLM-5.3-Flash の複数リクエスト同時処理 (`--parallel`)、長いコンテキスト、DFlash2 ドラフター; DeepSeek の会話の保持、エキスパートの読み込み、Engram の先読み; MiniMax の思考レベル: [docs/MODELS.md](docs/MODELS.md)。

## <a name="what-you-need"></a>必要なもの

| | |
| --- | --- |
| **GPU** | NVIDIA、compute capability 7.0 以上 (V100、RTX 20 以降): 1 枚、またはモデルを分け合う最大 16 枚。VRAM が多いほど速くなります。Pascal (Tesla P40 / P100、GTX 10) は実験的で、まだ検証されていません。**AMD (実験的):** RX 7900 XT / XTX、Radeon AI PRO R9700 / RX 9070、Strix Halo と Gorgon Halo (Radeon 8060S / 8065S); RX 6800 / 6900 と Radeon PRO V620 / W6800 はまだ検証されていません; GLM-5.3-Flash と DeepSeek-V4.1-Flash-1017 は AMD でも画像を読めます - [docs/AMD_MAYA.md](docs/AMD_MAYA.md) |
| **RAM** | 32 GB で動きます。RAM が多いほど多くのエキスパートを近くに置け、速くなります |
| **ディスク** | モデルが入る空きのある**高速な NVMe SSD** (GLM-5.3 · Maya-XS で約 75 GB、GLM-5.3 · Maya-S で約 100 GB、GLM-5.3 · Maya-L で約 160 GB、DeepSeek V4.1 · Maya-S で約 350 GB、その unsloth の Q8 GGUF で約 470 GB、MiniMax M3.1 · Maya-XS で約 95 GB、MiniMax-M3.1-Flash の EXL3 量子化で約 200 GB) と、画像エンコーダー (1.1 GB)。ハードディスクは不可 |
| **システム** | Linux (x86-64; AVX2 対応 CPU が最適)、NVIDIA ドライバー、CUDA 12.x (CUDA 13 は RTX 20 以降で動作し、V100 と Pascal では動作しません)、g++、Python 3.10+。Windows 10/11 (実験的): [docs/WINDOWS.md](docs/WINDOWS.md)。WSL2 は不可。AMD: ROCm 7 |

セットアップがこれらをすべて確認し、足りないものには正確なコマンドを表示します。システム全体には何もインストールしません。

## <a name="install"></a>インストール

[クイックスタート](#quick-start)の 2 つのコマンドを実行し、いくつかの質問に答えてください (Enter で推奨の選択になります)。セットアップは、ダウンロードを始める前に毎回内容を表示し、すべてを一度ビルドして (ダウンロードを除き 20-40 分)、**モデルを起動します**。その後 `http://127.0.0.1:8080` を開いてください。Ctrl+C で停止し、`./setup.sh` ですぐに再起動できます。

- **Windows:** `./setup.sh` の代わりに `START-MAYA.bat` - [docs/WINDOWS.md](docs/WINDOWS.md) を参照。
- **AMD:** [docs/AMD_MAYA.md](docs/AMD_MAYA.md)。
- **Apple Silicon の Mac (実験的、GLM-5.3-Flash のみ):** `./maya.sh --backend metal` (Mac では `./maya.sh` が自動で Metal を選びます) - [docs/METAL.md](docs/METAL.md) を参照。GitHub の Mac 上でビルドとユニットテストを行っていますが、実際のモデルでの動作確認はまだです (Mac を持っていないため)。お試しいただけたら、結果を issue で教えてください。
- **Intel Arc:** Maya のエンジンは非対応です。llama.cpp の SYCL バックエンドで Maya-XS の DeepSeek GGUF を動かしたコミュニティの報告: [docs/INTEL_ARC.md](docs/INTEL_ARC.md)。
- **更新:** ページの **Settings › About and updates**、または `git pull` のあと `./setup.sh`。モデルは再ダウンロードされません。
- **別の方法でダウンロードしたモデルファイル** (ダウンロードマネージャー、別の PC、Hugging Face キャッシュ): `./maya.sh --import-model FOLDER` が、FOLDER の中から Maya のファイルをサイズとチェックサムで見つけ (名前は問いません)、所定の場所に移動します (FOLDER が別のドライブにあるときは `--copy`。元のファイルは残ります)。一致しないファイルには触れません。
- **すべてのオプション** (GPU、コンテキスト、すでにお持ちのモデルフォルダ、リモートアクセスなど): [docs/SETUP.md](docs/SETUP.md)。

> **起動には数分かかります:** エンジンはエキスパート階層のために空き RAM のほとんど (約 6 GB を残して全部) を固定するため、Maya の実行中はほかのプログラムが使える RAM が少なくなります。

## <a name="something-went-wrong"></a>うまく動かないときは

遅い回答やエラーの直後に **`./maya.sh --report`** (Windows: `START-MAYA.bat --report`) を実行し、書き出された `maya-report.txt` を [GitHub の issue](https://github.com/mw00/project-maya/issues) に添付してください。ハードウェア、セットアップ、エンジンの速度ログが含まれ、自動で送信されることはありません。エンジン自身のログは Maya フォルダの `maya-<model>.log` です。

- **ビルドが止まる** ("nvcc ... cannot build for these GPUs"、"unsupported GNU version"、`cuda_profiler_api.h` / `nvtx3/nvToolsExt.h` が見つからない): それぞれの対処法は [docs/SETUP.md](docs/SETUP.md#if-the-build-fails) にあります。
- **遅く、SSD がずっとビジー:** エキスパート用の空き RAM が足りません。プログラムを閉じるか、RAM を増やしてください。
- **Windows: 回答が突然数倍遅くなる:** カードがいっぱいで、Windows が GPU メモリを RAM に移しています。もっと空けておいてください: `START-MAYA.bat --setup --vram-reserve-mib 2048` ([エキスパートプールのサイズ](docs/SETTINGS.md#the-expert-pools-size))。
- **Windows の Strix Halo / Gorgon Halo: ほかのプログラムのメモリが足りなくなる、またはセットアップがページファイルについて警告する:** セットアップが表示するサイズにページファイルを設定してください ([docs/AMD_MAYA.md](docs/AMD_MAYA.md))。
- **ダウンロードが止まった:** もう一度 `./maya.sh` を実行すると、続きから再開します。
- **ポート 8080 が使用中:** Maya がすでに動いているか、`--port 8081` でもう 1 つ起動できます。Settings > Port で Maya のポートを恒久的に変更できます (そのポートで再起動します)。

## <a name="support"></a>サポート

Project Maya は無料のオープンソースです。すべてのリリースは、公開前に私たち自身の NVIDIA / AMD マシンでテストしています (Mac 対応はこれまでのところ GitHub の仮想 Mac のみ)。スポンサーシップは、そのテスト用ハードウェア (現在の目標: 実機の Mac と新しい GPU のために月額スポンサー 50 人。Maya をより多くのハードウェアで快適に動かすため)、新しい量子化とモデルの制作、そして修正の迅速化に使われます。Maya がクラウドの請求を減らしてくれているなら、その一部をスポンサーとして還元することを検討してください。ありがとうございます!

<p align="center"><a href="https://github.com/sponsors/mw00"><img src="https://img.shields.io/badge/Sponsor-mw00-ea4aaa?logo=githubsponsors&logoColor=white&style=for-the-badge" alt="Sponsor on GitHub" height="50"></a>
<a href="https://buymeacoffee.com/peasantsmith"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>

### スポンサー

Project Maya をスポンサーしてくださっているすべての方に感謝します。月額 $15 以上のスポンサーの方々がここに掲載されます - [最初のスポンサーになる](https://github.com/sponsors/mw00)。

速度レポート、バグ報告、プルリクエストを歓迎します。何が最も助けになるかは [CONTRIBUTING.md](CONTRIBUTING.md) をご覧ください。

## <a name="credits-and-license"></a>クレジットとライセンス

- **人々:** Project Maya は Peasant Smith ([@mw00](https://github.com/mw00)、[X](https://x.com/PeasantSmith)) が作成・主導しており、プロジェクト、その名前、Maya のモデルを所有しています。[@needmorevram](https://github.com/needmorevram) ([X](https://x.com/needmorevram)) は共同メンテナーで、DFlash2 ドラフト、EXL3 サポート、KV ストリーミング、会話チェックポイント、レイヤー分割探索、エキスパートプロファイル、CPU エキスパートカーネル、そして `maya-ds4` の 3.5 ビット EXL3 オプション、4 / 8 GPU のテンソル並列、スレッド化されたエキスパートキャッシュのスワップ (#276) といった主要機能に貢献しました。コードや速度レポートを送ってくださったすべての方にも感謝します ([コントリビューター](https://github.com/mw00/project-maya/graphs/contributors))。
- **[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) をベースにしています** (MIT License、Copyright (c) 2023-2026 The ggml authors): 量子化フォーマット、CPU のドット積、プロンプト処理の MMQ カーネルは、固定したコミットからビルドされています (`third_party/ggml/LICENSE`)。
- **EXL3 サポート**は [exllamav3](https://github.com/turboderp-org/exllamav3) のフォーマット、コードブック、CPU エキスパートカーネル、タイルのテンソルコアデコードに準拠しています (MIT License、Copyright (c) 2025 Turboderp; `third_party/exllamav3/LICENSE`)。**DFlash2 ドラフターのフォワード**は [beellama.cpp](https://github.com/Anbeeld/beellama.cpp) のものに準拠しています (MIT License、Copyright (c) 2023-2026 The ggml authors、Anbeeld; `third_party/beellama.cpp/LICENSE`)。
- モデル: [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) は zai-org 製 (MIT); Maya の量子化と画像エンコーダーのファイルは zai-org の公開された重みから作られ、そのライセンスを引き継ぎます。[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) は DeepSeek 製 (MIT); Maya の量子化、Engram テーブル、ドラフターは DeepSeek の公開された重みから作られ、そのライセンスを引き継ぎます; ビジョンタワーは DeepSeek のチェックポイント由来で、GGUF オプションは [unsloth の GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)、EXL3 オプションは LibertAIDAI のもの ([2.9](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)、[3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)) です。オプションの DFlash2 ドラフター ([GLM-5.3-Flash-DFlash2-GGUF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF)、Anbeeld 製) は CC BY-NC-ND 4.0 で、非商用利用のみ、選んだときにだけダウンロードされます。
- ダッシュボードのフォント: Outfit (SIL Open Font License 1.1、`serve/web/fonts/OFL.txt`); アイコン: [Lucide](https://lucide.dev) (ISC License、`serve/web/LICENSE.lucide.txt`); UI ライブラリ: [Preact](https://preactjs.com) (MIT License) と [htm](https://github.com/developit/htm) (Apache License 2.0)。いずれも告知付きで `serve/web/preact.js` に同梱されています。
- Maya は [MIT License](LICENSE) のオープンソースです。[LICENSE](LICENSE) と `third_party/` の告知は、すべてのコピーに付属します。

---

*この翻訳は参考用です。英語版の [README.md](README.md) が正本で、こちらより新しい場合があります。*
