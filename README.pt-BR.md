<p align="center"><a href="README.md">English</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh-CN.md">简体中文</a> | Português</p>

<p align="center"><img src="assets/maya-logo.png" width="128" alt="Maya logo"></p>

<h1 align="center">Project Maya</h1>

<p align="center"><b>GLM-5.3 (744 B), GLM-5.3-Flash (321 B), DeepSeek-V4.1-Flash-1017 (748 B) e MiniMax-M3.1-Flash (428 B) na sua própria GPU</b><br>
privado e rápido · de uma a 16 GPUs NVIDIA · AMD e Windows · chat, imagens, API compatível com OpenAI e Anthropic · várias requisições ao mesmo tempo</p>

<p align="center">
<a href="https://huggingface.co/peasantsmith/GLM-5.3-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/glm53.svg" alt="GLM-5.3 Maya downloads"></a>
<a href="https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/glm.svg" alt="GLM-5.3-Flash Maya downloads"></a>
<a href="https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/glm-m-derisked.svg" alt="GLM-5.3-Flash Maya-M-Derisked downloads"></a>
<a href="https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/deepseek.svg" alt="DeepSeek-V4.1-Flash Maya downloads"></a>
<a href="https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF"><img src="https://raw.githubusercontent.com/mw00/project-maya/badges/minimax.svg" alt="MiniMax-M3.1-Flash Maya downloads"></a>
</p>

<p align="center">Criado e liderado por <b>Peasant Smith</b> (<a href="https://github.com/mw00">@mw00</a>), com a manutenção compartilhada de <b><a href="https://github.com/needmorevram">@needmorevram</a></b></p>

<p align="center"><a href="https://github.com/sponsors/mw00"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-ea4aaa?logo=githubsponsors&logoColor=white&style=for-the-badge" alt="Sponsor on GitHub" height="40"></a>
<a href="https://buymeacoffee.com/peasantsmith"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=black&style=for-the-badge" alt="Buy Me A Coffee" height="32"></a><br>
<sub>O Project Maya é gratuito. Se ele roda bem no seu PC, patrocinar ajuda a pagar o hardware de testes e o trabalho no projeto.</sub></p>

Modelos desse tamanho costumam precisar de um datacenter. O Maya roda quatro deles no hardware que você já tem:
**GLM-5.3**, **GLM-5.3-Flash**, **DeepSeek-V4.1-Flash-1017** e **MiniMax-M3.1-Flash**. Cada modelo tem um motor feito para ele. O motor mantém na GPU os especialistas (experts) que a sua conversa mais usa, os próximos na RAM e o resto no seu SSD NVMe, e os move conforme você conversa. Ele mede suas GPUs, CPU, RAM e SSD e se ajusta a eles. Nada sai da sua máquina.

- **Rápido:** até 149 tokens/s com duas RTX 5090 e uma RTX 4090, 124 com duas H100, 118 com quatro RTX 4090, 81 com uma RTX PRO 6000, mais de 30 com uma única RTX 5090. [Números reais de usuários reais](#how-fast-is-it).
- **Perto do original:** as quantizações próprias do Maya do GLM-5.3-Flash mantêm 97,7-99,2% da precisão zero-shot do modelo FP8; o DeepSeek V4.1 · Maya-S escolhe o mesmo próximo token que o original em 92,6% das vezes, e o MiniMax M3.1 · Maya-S em 87,6% (a média das suas categorias).
- **Se conecta às suas ferramentas:** API compatível com OpenAI e Anthropic, com chamadas de ferramentas, pronta para agentes de programação.
- **Várias requisições ao mesmo tempo:** até 8 respondidas juntas com GLM-5.3-Flash, DeepSeek-V4.1-Flash e MiniMax-M3.1-Flash - para agentes que rodam várias sessões, ou várias pessoas em um só PC.
- **Tudo incluído:** um chat e um monitor ao vivo no navegador, imagens em todos os modelos, níveis de raciocínio, uma página Models que baixa, carrega e apaga modelos, um botão Turbo e uma instalação com um só comando.

## <a name="quick-start"></a>Início rápido

```sh
git clone https://github.com/mw00/project-maya.git && cd project-maya
./setup.sh
```

A instalação pergunta qual modelo você quer, baixa, compila tudo uma vez e **inicia o modelo**. Depois abra `http://127.0.0.1:8080`. No Windows, use `START-MAYA.bat` em vez de `./setup.sh`. Mais em [Instalação](#install).

Toda a documentação: [docs/README.md](docs/README.md)

## <a name="the-models"></a><a name="deepseek-v41-flash-1017-new"></a><a name="minimax-m31-flash-new"></a>Escolha seu modelo

Toda quantização do Maya tem o nome do modelo seguido do nível - XS, S, M ou L, conforme quantos bits seus especialistas mantêm. A instalação recomenda uma para o seu PC; para escolher uma você mesmo, informe o id dela: `./setup.sh --model glm-maya-s24`.

| Modelo | Quantização Maya · id da instalação | Tamanho | Feito para |
| --- | --- | ---: | --- |
| **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**<br><sub>zai-org · MIT · 321 B, 18 B ativos · até 1M de contexto</sub> | **Maya-S** (padrão) · `glm-maya-s` | 96,5 GB | PCs com um pool de memória (RAM e VRAM juntas) menor |
| | Maya-S24 · `glm-maya-s24` | 94,7 GB | placas de 24 GB ou menos: ~1,5 GB a mais da placa para especialistas, decodificação +14% |
| | Maya-XS · `glm-maya-xs` | 73,4 GB | PCs com 64 GB de RAM e uma placa de 24 GB, com contexto de 128K |
| | Maya-M · `glm-maya-m` | 116 GB | um pool de memória maior; calibrado para chamadas de ferramentas e código de front-end |
| | Maya-M-Derisked · `glm-maya-m-derisked` | 116 GB | Maya-M com uma edição de pesos de [Blackfrost_AI](https://x.com/Blackfrost_AI) que reduz as recusas generalizadas; experimental |
| | Maya-L · `glm-maya-l` | 156,3 GB | os maiores pools de memória; o mais próximo do original |
| **[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)**<br><sub>zai-org · [licença própria](https://huggingface.co/zai-org/GLM-5.3) · 744 B, 40 B ativos · até 1M de contexto · só texto</sub> | **Maya-XS** · `glm-5.3-maya-xs` | 170,0 GB | o modelo inteiro num pool de memória de 192 GB (VRAM e RAM juntas); NVIDIA a partir da V100 e AMD ([detalhes](docs/GLM53.md)) |
| | UD-IQ2_M (unsloth) · `GLM-5.3-UD-IQ2_M` | 238,6 GB | os maiores pools de memória; mais perto do original |
| **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**<br><sub>DeepSeek · MIT · 748 B</sub> | **Maya-S** (padrão) · `deepseek-maya-s` | 340 GB | qualquer quantidade de memória: o que a VRAM e a RAM não comportam é lido do SSD |
| | Maya-XS (v2) · `deepseek-maya-xs` | 277 GB | o modelo inteiro na memória de um PC de 128 GB (as tabelas Engram ficam no SSD); a v2 é cerca de 12% mais próxima do original e cerca de 20% mais rápida para escrever que a v1 - rodar a instalação de novo oferece a atualização |
| **[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)**<br><sub>MiniMax · [MiniMax Community License](LICENSE.minimax-m31) · 428 B, 23 B ativos · até 1M de contexto</sub> | Maya-XS · `minimax-maya-xs` | 93 GB | o modelo inteiro na memória de um PC de 128 GB |
| | Maya-S · `minimax-maya-s` | 122 GB | mais próximo do original |

**Qual modelo cabe no seu PC?** Veja em [docs/MEMORY.md](docs/MEMORY.md) a memória recomendada de cada modelo (VRAM + RAM somadas).

Todos rodam em NVIDIA a partir da V100 / RTX 20, e em AMD e Windows (veja [O que você precisa](#what-you-need)). Os tamanhos do DeepSeek incluem as tabelas Engram. A instalação também oferece os GGUFs da unsloth e as quantizações EXL3 da LibertAIDAI do DeepSeek, o DeepSeek abliterated da orcarouter, a quantização EXL3 do MiniMax da Mia-AiLab, as quantizações EXL3 do GLM da turboderp o Qwen-3.8-Flash-Next (CYBER-FROST) como GGUF Q5_K_M e o próprio Qwen-3.8-Flash-Next como GGUF UD-IQ2_XS da unsloth, ambos com imagens ([suas configurações](docs/SETTINGS.md#qwen-38-flash-next-cyber-frost)). **Arquivos, drafters, imagens e opções de cada modelo: [docs/MODELS.md](docs/MODELS.md).**

## <a name="quality"></a>Qualidade

As quantizações do Maya são feitas a partir dos pesos divulgados de cada modelo e medidas contra o original:

- **GLM-5.3-Flash · quantizações Maya** mantêm 97,7-99,2% da precisão zero-shot do modelo FP8 e acertam o próximo token dele em 83-90% das vezes (Maya-L: 99,2%, 90%). Como são feitas e medidas: [MAYA-S](bench/results/MAYA-S.md), [MAYA-S24](bench/results/MAYA-S24.md), [MAYA-M](bench/results/MAYA-M.md), [MAYA-L](bench/results/MAYA-L.md).
- **DeepSeek V4.1 · Maya-S** escolhe o mesmo próximo token que o original em 92,6% das vezes.
- **MiniMax M3.1 · Maya-S** em 87,6% das vezes (a média das suas categorias).

Mais na página de cada modelo no [Hugging Face](https://huggingface.co/peasantsmith).

## <a name="how-fast-is-it"></a>Qual é a velocidade?

Máquinas reais medidas por nossos usuários, a maioria com `./maya.sh --bench` (decodificação: três respostas gulosas de 256 tokens; prefill: um prompt de 8K tokens, salvo indicação na linha). Compartilhe a sua: `--bench`, depois `--report`, em uma [issue no GitHub](https://github.com/mw00/project-maya/issues).

<details><summary><b>Benchmarks de velocidade do GLM-5.3-Flash</b></summary>

| Máquina | Modelo | Decodificação (escrevendo a resposta) | Prefill (lendo o seu prompt) | Por |
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
| **8x AMD Instinct MI50 32 GB** (gfx906, PCIe), EPYC 7K62, 112 GB RAM | Maya-L, 128K context | 43.3 tokens/s | 288 tokens/s | @abluesheep (v1.1.52, #830) |
| **RTX PRO 6000 Blackwell 96 GB** (PCIe 5.0 x16), Ryzen 9 9950X3D, 253 GB RAM, Windows 11 | Maya-L, a coding-agent session grown from 265K to 482K tokens | 42.4 tokens/s (median of 51 answers) | - | @Andrey0710 (v1.1.53, #873) |
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

<details><summary><b>Benchmarks de velocidade do DeepSeek-V4.1-Flash</b></summary>

| Máquina | Modelo | Decodificação (escrevendo a resposta) | Prefill (lendo o seu prompt) | Por |
| --- | --- | ---: | ---: | --- |
| **8x RTX 5090 32 GB**, Xeon Gold 6530, 1 TB RAM | Maya-S, 512K context, `--parallel 4` | **184.2 tokens/s** | 2510 tokens/s | @etevrfed34-commits (v1.1.34, #585) |
| **10x RTX 3090 24 GB** (tensor parallel 2, PCIe 4.0 x4 each, 200 W), Xeon (24 threads), 64 GB RAM | EXL3 2.9 bpw, 1M context | **97.0 tokens/s** | 2390 tokens/s (128K) | @DoktorMincs (v1.1.28, #494) |
| **RTX 6000D 85 GB** (PCIe 5.0 x16), Threadripper PRO 9965WX, 251 GB RAM | EXL3 2.9 bpw, 512K context | **89.3 tokens/s** | 1687 tokens/s (2836 at 33K) | @Ross-Yang-94 (v1.1.25, #423) |
| **2x A100-SXM4 80 GB**, Xeon Platinum 8360Y, 1 TB RAM | Maya-S, 1M context | **87.4 tokens/s** | 2084 tokens/s | @SunYong0821 (v1.1.34, #445, #594) |
| **2x RX 7900 XTX 24 GB** (tensor parallel 2, PCIe 4.0 x4 each) + **Radeon 8060S** helper (Ryzen AI Max+ 395, 128 GB), the model on a USB4 NVMe | Maya-XS (v2), 512K context, `DS4_INDEX_TC=1` | **82.5 tokens/s** | 1587 tokens/s (130K; 663 at 8K) | @pepuscz (v1.1.50 + #801, #797) |
| **RTX PRO 6000 Blackwell 96 GB** (PCIe 5.0 x16), Ryzen 9 9950X3D, 253 GB RAM, Windows 11 | Maya-S, 256K context | 72.3 tokens/s | 1757 tokens/s | @Andrey0710 (v1.1.43, #737) |
| **2x RTX PRO 5000 Blackwell 72 GB**, Ryzen 9 9950X3D, 249 GB DDR5 | Maya-S, 512K context | 68-71 tokens/s (prose), up to 162 (code), tuned | ~3700 tokens/s (42K) | @realsquall1980-art (v1.1.28, #424) |
| **4x RTX 4090 24 GB** (PCIe 4.0 x16), 2x Xeon Platinum 8380, 251 GB RAM, SATA SSD | Maya-S, 256K context | **67.5 tokens/s** | 1087 tokens/s (400 at 2K) | @oscarlius (v1.1.51) |
| **1x A100-SXM4 80 GB**, Xeon Platinum 8360Y, 1 TB RAM | Maya-S, 1M context | 60.2 tokens/s | 241 tokens/s | @SunYong0821 (v1.1.34, #594) |
| **4x AMD Instinct MI100 32 GB** (gfx908, PCIe 4.0 x16 each), EPYC 7402, 252 GB RAM | Maya-XS, 32K context | 51.8 tokens/s | 172 tokens/s | @nsaxploit (v1.1.22, #396) |
| **4x RTX 3090 24 GB** (PCIe 3.0, SM clocks capped at 1500 MHz), Threadripper 1950X, 126 GB RAM | Maya-S, 256K context | 50-53 tokens/s (code), 40-43 (prose) | 947 tokens/s (10K), 1954 (90K) | @mgabor3141 (v1.1.26, #448) |
| **2x NVIDIA A40 48 GB** (PCIe 4.0 x16), 2x Xeon Gold 6330, 256 GB RAM, Windows Server 2022 | Maya-S, 256K context | 48.9 tokens/s (tuned) | ~660 tokens/s (3K) | @yjyz1011 (v1.1.28, #508) |
| **2x RTX PRO 6000 Blackwell 96 GB** (tensor parallel 2, PCIe 5.0), Core Ultra 7 270K Plus, 188 GB DDR5 | UD-Q8_K_XL, 1M context, DSpark | 48.3 tokens/s | 1245 tokens/s (3360 at 48K) | v1.1.51, #819 |
| **RTX 4090 24 GB** (PCIe 4.0 x8), Ryzen 9 9950X3D, 192 GB DDR5 | Maya-XS, 64K context | 44.1 tokens/s | 539 tokens/s | @naeq (v1.1.27, #503) |
| **4x Tesla V100 32 GB** (PCIe 3.0 x16), 2x Xeon E5-2630 v4, 504 GB RAM | Maya-S / UD-Q4_K_XL / UD-Q8_K_XL, 64K context | 44.0 / 28.6 / 29.7 tokens/s | 592 / 236 / 233 tokens/s | @HansStrupat (v1.1.49, #785) |
| **1x L40S 48 GB** (vGPU), Xeon Platinum 8580 (a VM, 32 vCPUs), 504 GB RAM | UD-Q8_K_XL, 256K context | 43.2 tokens/s | 1862 tokens/s (174K) | @talisp (v1.1.32, #577) |
| **AMD Radeon 8065S** (Ryzen AI Max+ PRO 495, 192 GB, 160 GB of it for the GPU), Windows 11 | orcarouter abliterated Mix-128G, 256K context, DSpark | 43.2 tokens/s | 453 tokens/s | @tobitege (v1.1.37, #663) |
| **2x RX 7900 XTX 24 GB** (tensor parallel 2, PCIe 4.0 x4 each) + **Radeon 8060S** helper (Ryzen AI Max+ 395, 128 GB), the model on a USB4 NVMe | Maya-S, 512K context, `DS4_INDEX_TC=1` | 41.7 tokens/s (49.4 after a 130K-token prompt, cached) | 1378 tokens/s (130K; 478 at 8K) | @pepuscz (v1.1.48 + #801, #797) |
| **4x RTX 3090 24 GB** (PCIe 4.0 x16), Threadripper PRO 3975WX, 502 GB RAM | UD-Q8_K_XL, 256K context | 37.1 tokens/s | 596 tokens/s | @bandito1 (v1.1.50, #766) |
| **4x RX 7900 XT 20 GB** (gfx1100, `--tp 4`, tuned), 2x Xeon E5-2699C v4 (a Dell T7910), 1 TB DDR4-2400 | UD-Q4_K_XL | 35.0 tokens/s | - | @Cass67 (v1.1.32, #566) |
| **2x RTX 3090 24 GB** (PCIe 4.0 x8, 270 W), Core Ultra 7 265K, 192 GB DDR5 | Maya-S, 1M context, tuned | 33.3 tokens/s | 467 tokens/s | @Jellepepe (v1.1.35, #614) |
| **2x Tesla V100 16 GB** (PCIe 3.0 x16), 2x Xeon E5-2696 v4, 128 GB DDR3-1866 | Maya-XS (v2), 256K context, DSpark | 32.0 tokens/s | 420 tokens/s | @c200762620-max (v1.1.51, #815) |
| **3x Tesla V100 32 GB** (PCIe 3.0 x16), 2x Xeon E5-2696 v3, 128 GB DDR3L, Windows 10 | Maya-S, 64K context | 23-26 tokens/s (long answers) | 377 tokens/s | @noahark (v1.1.26, #450) |
| **2x RTX 2080 Ti 22 GB** (modded, PCIe 3.0 x16, power-capped at 250 W), 2x Xeon E5-2696 v4, 94 GB RAM | Maya-XS, 128K context | 23.1 tokens/s | 317 tokens/s | @NorwinYu (v1.1.36, #659) |
| **2x RTX 4090 24 GB** (PCIe 3.0 + 4.0 x16), 2x Xeon Silver 4309Y, 188 GB RAM | Maya-XS, 64K context | 21.5 tokens/s | 573 tokens/s | @fuad00 (v1.1.22, #393) |
| **RTX 4000 Ada 20 GB**, Xeon w7-3455, 255 GB DDR5, Windows 11 | Maya-XS, 256K context | 21.1 tokens/s | 518 tokens/s (20K) | @pietroMastro92 (v1.1.29, #534) |
| **2x Tesla V100 32 GB** (PCIe 3.0 x8 each), Ryzen 9 5900XT, 128 GB DDR4-3200 | Maya-S, 64K context | 20.8 tokens/s | 344 tokens/s | @pirosap (v1.1.34, #586) |
| **RX 7900 XTX 24 GB** (gfx1100, PCIe 4.0 x16), Core i7-14700KF, 63 GB RAM, ROCm 10.2 | Maya-XS, 64K context, DSpark | 18.4 tokens/s | 395 tokens/s (146 at 2K) | @Cayaf89 (v1.1.46, [#757](https://github.com/mw00/project-maya/issues/757)) |

</details>

<details><summary><b>Benchmarks de velocidade do MiniMax-M3.1-Flash</b></summary>

| Máquina | Modelo | Decodificação (escrevendo a resposta) | Prefill (lendo o seu prompt) | Por |
| --- | --- | ---: | ---: | --- |
| **RTX PRO 6000 Blackwell 96 GB** (PCIe 5.0 x16), Ryzen 9 9950X3D, 253 GB RAM, Windows 11 | Maya-S, 256K context, DSpark | **48.6 tokens/s** | 757 tokens/s | @Andrey0710 (v1.1.43, #738) |
| **RTX 3090 24 GB + 8x RTX 5060 Ti 16 GB** (PCIe 3.0 x8 each), 183 GB RAM | EXL3 3.5 bpw, tensor parallel 4 + DSpark, 8K context | 38.2 tokens/s (code), 32.4 (prose) | - | @needmorevram (#477) |
| **1x L40S 48 GB** (vGPU), Xeon Platinum 8580 (a VM, 32 vCPUs), 504 GB RAM | Maya-S / EXL3 3.55 bpw, 256K context | 23.4 / 16.7 tokens/s | 332 / 273 tokens/s (158K) | @talisp (v1.1.32, #577) |
| **2x RTX 3090 24 GB** (power-capped at 230 W), EPYC 7543 (a VM, 32 vCPUs), 224 GB RAM | Maya-S, 200K context, DSpark | 22.0 tokens/s (code), 18.4 (prose) | 163 tokens/s (10K) | @noonghunna (v1.1.30, #549) |
| **AMD Radeon 8065S** (Ryzen AI Max+ PRO 495, 192 GB, 160 GB of it for the GPU), Windows 11 | Maya-S, 128K context, DSpark | 21.0 tokens/s | 139 tokens/s | @tobitege (v1.1.37, #663) |
| **2x Tesla V100 16 GB** (PCIe 3.0 x16), 2x Xeon E5-2696 v4, 128 GB DDR3-1866 | Maya-XS, 256K context, DSpark | 14.5 tokens/s | 254 tokens/s | @c200762620-max (v1.1.51, #816) |

</details>

Um token equivale a cerca de ¾ de uma palavra. A velocidade se mantém conforme a conversa cresce, e as primeiras respostas depois de iniciar são as mais lentas, enquanto os caches se enchem com o que você usa.

## <a name="using-it"></a>Como usar

- **No navegador:** chat (guardado no seu navegador, com busca, fixar e pastas, instruções e nível de raciocínio por chat, editar e regenerar, prévias de HTML), um **Monitor** ao vivo do modelo e do seu hardware (até o que ocupa a memória de cada GPU), um tamanho de contexto e uma porta que você pode mudar em Settings, e **Models**: todos os modelos instalados nesta máquina, a troca para outro (o Maya reinicia com ele) e as configurações do motor de cada um (contexto, GPUs, drafting, imagens, memória), salvas no `maya-<model>.json` dele.
- **Página Models:** o catálogo lista todos os modelos Maya com o tamanho. Baixe um com barra de progresso, carregue, descarregue o que está rodando ou apague um - sem linha de comando.
- **Turbo:** um botão ao lado dos parâmetros do modelo aplica as configurações mais rápidas para ele (amostragem gulosa, sem raciocínio, pular especialistas, o drafting mais rápido). Troca um pouco de qualidade por velocidade; ao desligar, suas configurações voltam exatamente como estavam.
- **Economizado:** a linha Saved da barra lateral soma o que a API do próprio modelo (a do DeepSeek, da Z.ai ou da MiniMax) teria cobrado pelos mesmos tokens - o prompt que o Maya reaproveita do cache pelo preço de cache da API, e o horário de pico do DeepSeek pelo preço de pico - guardado por modelo em `token-savings.json`; os preços podem ser mudados em **Settings › Savings**.
- **Seus apps e agentes de programação:** uma API compatível com OpenAI em `http://127.0.0.1:8080/v1` e a da Anthropic em `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`). As chamadas de ferramentas são enviadas em streaming conforme o modelo as escreve, e o progresso da leitura de um prompt longo também, quando a requisição pede (`"return_progress": true`). Uma página web servida de outro endereço também pode chamá-la, depois que você permitir esse endereço (`"cors_origins"`, [docs/SETUP.md](docs/SETUP.md#web-pages-on-another-address)).
- **Responses API:** `/v1/responses` oferece conversas com texto e imagem, streaming, chamadas de ferramentas de função, saída validada por esquema, tarefas em segundo plano com cancelamento e replay, e histórico opcional compartilhado em memória (`store:true`, `previous_response_id`). As respostas não são guardadas por padrão. Opções e limites: [docs/RESPONSES.md](docs/RESPONSES.md).
- **API de modelos:** `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull` e `/api/delete` listam, carregam, baixam e apagam modelos a partir de qualquer app ou script. Uma requisição a um modelo que não está carregado troca para ele, e `keep_alive` o descarrega quando fica ocioso. `/models/load`, `/models/unload` e `/models/sse` também vêm ligados por padrão (`"model_management": false` os desliga).
- **Raciocínio:** Off, Low, High (o padrão) ou Max, em Settings ou por requisição, e um campo de **orçamento de raciocínio** que define quantos tokens o modelo pode usar pensando antes de responder.
- **Guarda contra loops:** ligada por padrão - um agente que repete a mesma chamada de ferramenta com o mesmo resultado é conduzido para fora disso.
- **Imagens:** anexe uma no chat, ou envie partes de imagem por qualquer uma das APIs. Todos os modelos leem imagens (detalhes da instalação por modelo: [docs/MODELS.md](docs/MODELS.md)).
- **Ferramentas MCP:** liste servidores MCP na configuração (`"mcp_servers"`; o bloco `mcpServers` do Claude Desktop funciona como está) e o modelo do chat na web pode chamar as ferramentas deles. Um interruptor no menu **+** do chat liga e desliga. O que ele faz: [serve/mcp.py](serve/mcp.py).
- **Docker:** não há imagem oficial. [docs/DOCKER.md](docs/DOCKER.md) é uma receita para montar uma a partir de uma versão.
- **De outro dispositivo:** `./maya.sh --setup --host 0.0.0.0 --api-key <secret>`. Sempre defina uma chave.
- **Ajuste fino:** `./maya.sh --calibrate` ajusta o motor ao seu hardware em 10-15 minutos. Todas as configurações: [docs/SETTINGS.md](docs/SETTINGS.md).
- <a name="deepseek-engram-prompt-read-ahead"></a><a name="deepseek-conversations-kept"></a><a name="deepseek-loading-the-experts"></a>**As opções próprias de cada modelo** - várias requisições ao mesmo tempo no GLM-5.3-Flash (`--parallel`), contextos longos e o drafter DFlash2; as conversas mantidas, o carregamento de especialistas e a leitura antecipada do Engram no DeepSeek; os níveis de raciocínio do MiniMax: [docs/MODELS.md](docs/MODELS.md).

## <a name="what-you-need"></a>O que você precisa

| | |
| --- | --- |
| **GPU** | NVIDIA, compute capability 7.0 ou superior (V100, RTX 20 e mais novas): uma, ou até 16 que dividem o modelo. Mais VRAM é mais rápido. Pascal (Tesla P40 / P100, GTX 10) é experimental, ainda não validada. **AMD:** RX 7900 XT / XTX, Radeon AI PRO R9700 / RX 9070, Strix Halo e Gorgon Halo (Radeon 8060S / 8065S); RX 6800 / 6900 e Radeon PRO V620 / W6800 ainda não validadas; GLM-5.3-Flash e DeepSeek-V4.1-Flash-1017 leem imagens lá também - [docs/AMD_MAYA.md](docs/AMD_MAYA.md) |
| **RAM** | 32 GB já roda. Mais RAM mantém mais especialistas por perto e é mais rápido |
| **Disco** | **Um SSD NVMe rápido** com espaço para o modelo (~75 GB para GLM-5.3-Flash · Maya-XS, ~100 GB para GLM-5.3-Flash · Maya-S, ~160 GB para GLM-5.3-Flash · Maya-L, ~350 GB para DeepSeek V4.1 · Maya-S, ~470 GB para o GGUF Q8 da unsloth dele, ~95 GB para MiniMax M3.1 · Maya-XS, ~200 GB para a quantização EXL3 do MiniMax-M3.1-Flash) e o codificador de imagens (1,1 GB). Não um HD |
| **Sistema** | Linux (x86-64; uma CPU com AVX2 é o ideal), o driver NVIDIA, CUDA 12.x (o CUDA 13 funciona para RTX 20 e mais novas, não para a V100 nem Pascal), g++, Python 3.10+. Windows 10/11: [docs/WINDOWS.md](docs/WINDOWS.md). Não WSL2. AMD: ROCm 7 |

A instalação verifica tudo isso, mostra o comando exato para o que faltar e não instala nada no sistema inteiro.

## <a name="install"></a>Instalação

Execute os dois comandos do [Início rápido](#quick-start) e responda a algumas perguntas (Enter aceita a opção recomendada). A instalação mostra cada download antes de começar, compila tudo uma vez (20-40 minutos mais o download) e **inicia o modelo**. Depois abra `http://127.0.0.1:8080`. Ctrl+C o para, e `./setup.sh` o inicia de novo na hora.

- **Windows:** `START-MAYA.bat` em vez de `./setup.sh` - veja [docs/WINDOWS.md](docs/WINDOWS.md).
- **AMD:** [docs/AMD_MAYA.md](docs/AMD_MAYA.md).
- **Mac com Apple Silicon (experimental, só GLM-5.3-Flash):** `./maya.sh --backend metal` (em um Mac, `./maya.sh` escolhe o Metal sozinho) - veja [docs/METAL.md](docs/METAL.md). Compilado e testado por unidade nos Macs do GitHub, ainda não testado por nós com o modelo real (não temos um Mac): se você rodar, conte numa issue como foi.
- **Intel Arc:** não é suportada pelo motor do Maya; um relato da comunidade com os GGUFs do DeepSeek Maya-XS no backend SYCL do llama.cpp: [docs/INTEL_ARC.md](docs/INTEL_ARC.md).
- **Atualização:** em **Settings › About and updates** na página, ou `git pull` e depois `./setup.sh`. O modelo não é baixado de novo.
- **Arquivos do modelo baixados de outro jeito** (um gerenciador de downloads, outro PC, um cache do Hugging Face): `./maya.sh --import-model FOLDER` encontra os arquivos do Maya em FOLDER pelo tamanho e pela soma de verificação, não importa o nome, e os move para o lugar certo (`--copy` quando FOLDER está em outro disco; os originais ficam). Arquivos que não batem são deixados em paz.
- **Todas as opções** (GPUs, contexto, uma pasta de modelo que você já tem, acesso remoto, ...): [docs/SETUP.md](docs/SETUP.md).

> **A inicialização leva alguns minutos:** o motor reserva a maior parte da RAM livre (tudo menos cerca de 6 GB) para a camada de especialistas, então os outros programas ficam com pouca RAM enquanto o Maya roda.

## <a name="something-went-wrong"></a>Algo deu errado?

Execute **`./maya.sh --report`** (Windows: `START-MAYA.bat --report`) logo depois de uma resposta lenta ou de um erro, e anexe o `maya-report.txt` que ele gera a uma [issue no GitHub](https://github.com/mw00/project-maya/issues). Ele contém seu hardware, sua instalação e as linhas de velocidade do motor; nada é enviado por conta própria. O log do próprio motor é `maya-<model>.log` na pasta do Maya.

- **A compilação para** ("nvcc ... cannot build for these GPUs", "unsupported GNU version", ou faltam `cuda_profiler_api.h` / `nvtx3/nvToolsExt.h`): [docs/SETUP.md](docs/SETUP.md#if-the-build-fails) tem a solução para cada caso.
- **Lento, com o SSD ocupado o tempo todo:** não há RAM livre suficiente para os especialistas. Feche programas ou adicione RAM.
- **Windows: respostas de repente várias vezes mais lentas:** a placa está cheia demais e o Windows move memória da GPU para a RAM. Deixe mais espaço livre nela: `START-MAYA.bat --setup --vram-reserve-mib 2048` ([o tamanho do pool de especialistas](docs/SETTINGS.md#the-expert-pools-size)).
- **Windows em um Strix Halo / Gorgon Halo: outros programas ficam sem memória, ou a instalação avisa sobre o arquivo de paginação:** ajuste o arquivo de paginação para o tamanho que a instalação mostra ([docs/AMD_MAYA.md](docs/AMD_MAYA.md)).
- **O download parou:** execute `./maya.sh` de novo; ele continua de onde parou.
- **A porta 8080 está em uso:** o Maya já está rodando, ou `--port 8081` inicia outro. Settings > Port muda o Maya para outra porta de vez (ele reinicia lá).

## <a name="support"></a>Apoio

O Project Maya é gratuito e de código aberto. Cada versão é testada em nossas próprias máquinas NVIDIA e AMD antes de sair (o suporte a Mac, até agora, só nos Macs virtuais do GitHub), e o patrocínio paga esse hardware de testes (a meta atual: 50 patrocinadores mensais para um Mac de verdade e novas GPUs, para o Maya rodar bem em mais hardware), a criação de novas quantizações e modelos e correções mais rápidas. Se o Maya poupa você de uma conta de nuvem, considere devolver uma pequena parte dela como patrocínio - obrigado!

<p align="center"><a href="https://github.com/sponsors/mw00"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-ea4aaa?logo=githubsponsors&logoColor=white&style=for-the-badge" alt="Sponsor on GitHub" height="40"></a>
<a href="https://buymeacoffee.com/peasantsmith"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=black&style=for-the-badge" alt="Buy Me A Coffee" height="32"></a><br>
<b>O Maya é gratuito e aberto para todos. Tornar-se patrocinador ajuda a mantê-lo assim e paga o hardware em que cada versão é testada. Obrigado!</b></p>

Relatos de velocidade, relatos de bugs e pull requests são bem-vindos: o [CONTRIBUTING.md](CONTRIBUTING.md) diz o que mais ajuda.

## <a name="credits-and-license"></a>Créditos e licença

- **Pessoas:** o Project Maya é criado e liderado por Peasant Smith ([@mw00](https://github.com/mw00), [X](https://x.com/PeasantSmith)), dono do projeto, do nome e dos modelos Maya. [@needmorevram](https://github.com/needmorevram) ([X](https://x.com/needmorevram)) é co-mantenedor e contribuiu com grandes funcionalidades: o drafting DFlash2, o suporte a EXL3, o streaming de KV, os checkpoints de conversa, a busca de divisão de camadas, o perfil de especialistas e os kernels de especialistas na CPU, e, para o `maya-ds4`, a opção EXL3 de 3,5 bits, o tensor parallel em 4 / 8 GPUs e as trocas do cache de especialistas em threads (#276). Obrigado também a todos que enviaram código e relatos de velocidade ([contribuidores](https://github.com/mw00/project-maya/graphs/contributors)).
- **Construído sobre o [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT License, Copyright (c) 2023-2026 The ggml authors): os formatos de quantização, os produtos escalares na CPU e os kernels MMQ do caminho de prompts, compilados a partir de um commit fixo (`third_party/ggml/LICENSE`).
- **O suporte a EXL3** segue o formato, os codebooks, o kernel de especialistas na CPU e a decodificação de um bloco em tensor cores do [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT License, Copyright (c) 2025 Turboderp; `third_party/exllamav3/LICENSE`). **O forward do drafter DFlash2** segue o do [beellama.cpp](https://github.com/Anbeeld/beellama.cpp) (MIT License, Copyright (c) 2023-2026 The ggml authors, Anbeeld; `third_party/beellama.cpp/LICENSE`).
- Os modelos: [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) da zai-org (MIT); as quantizações do Maya e o arquivo do codificador de imagens são feitos a partir dos pesos divulgados pela zai-org e mantêm a licença dela. [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) da DeepSeek (MIT); a quantização do Maya, suas tabelas Engram e o drafter são feitos a partir dos pesos divulgados pela DeepSeek e mantêm a licença dela; a torre de visão vem do checkpoint da DeepSeek, a opção GGUF vem dos [GGUFs da unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) e as opções EXL3 são da LibertAIDAI ([2.9](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) e [3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)). O drafter DFlash2 opcional ([GLM-5.3-Flash-DFlash2-GGUF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF), de Anbeeld) é CC BY-NC-ND 4.0, só para uso não comercial, e só é baixado quando você o escolhe.
- A fonte do painel: Outfit (SIL Open Font License 1.1, `serve/web/fonts/OFL.txt`); seus ícones: [Lucide](https://lucide.dev) (ISC License, `serve/web/LICENSE.lucide.txt`); sua biblioteca de UI: [Preact](https://preactjs.com) (MIT License) com [htm](https://github.com/developit/htm) (Apache License 2.0), embutidos em `serve/web/preact.js` com seus avisos.
- O Maya é de código aberto sob a [MIT License](LICENSE); os avisos em [LICENSE](LICENSE) e `third_party/` acompanham cada cópia.

---

*Esta tradução é uma referência. O [README.md](README.md) em inglês é a versão de referência e pode estar mais recente.*
