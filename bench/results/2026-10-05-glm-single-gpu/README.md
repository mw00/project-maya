# GLM-5.3-Flash UD-IQ1_S on one GPU at several VRAM budgets (2026-10-05)

Build: glm5-flash after commit 54f6fc5 + `MAYA_GLM_VRAM_GB` (caps the engine's total use of the card, so the
expert pool gets what a card of that size would leave it). Host: the 2x V100 PC, using one Tesla V100-PCIE-32GB (PCIe 3.0 x16,
900 GB/s HBM2), Xeon E5-2690 v4, **30 GB RAM** (the pinned RAM tier gets ~23 GB), NVMe at ~2.2-2.4 GB/s.
Protocol: `CUDA_VISIBLE_DEVICES=0 MAYA_GLM_SPLIT=0`, fresh process (tiers warmed at load from expert_prior.txt),
three requests of 128 tokens on different prompts (chat 40 tokens, code 61, math 61), context 4096.

| VRAM | in use before experts | expert slots (of 12,096) | decode tok/s, req 1 / 2 / 3 | VRAM hit | RAM fetches /tok | disk reads /tok (ms/tok) | prompt ms/tok |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| 2x V100 (64 GB, split) | 2 x ~4 GB | 8,458 | ~30-35 (warm) | 98% | ~6 | 0.2-0.5 (1-2) | 37-50 |
| 32 GB (1x V100) | 6.7 GB | 3,738 | 5.8 / 6.7 / 7.4 | 74-82% | 37-53 | 25-33 (75-97) | 250-283 |
| 24 GB | 6.7 GB | 2,562 | 4.0 / 4.8 / 9.1 | 61-84% | 34-78 | 18-51 (54-150) | 322-381 |
| 16 GB | 6.7 GB | 1,302 | 2.8 / 3.6 / 5.3 | 44-65% | 84-117 | 32-77 (95-225) | 418-483 |
| 12 GB | 6.7 GB | 714 | 2.3 / 2.9 / 4.5 | 29-50% | 133-141 | 34-96 (100-276) | 456-564 |

Reading it:
- One GPU holds all 45 layers' dense weights: 6.7 GB is gone before the first expert (a 12 GB card keeps 4.6 GB
  for experts = 17 per layer).
- On every single-GPU budget the DISK is the wall: VRAM + this host's 23 GB RAM tier hold 4,300-7,300 of the
  12,096 experts, so 25-96 experts per token come from the NVMe (~3 ms each). A typical 64 GB-RAM PC would hold
  ~2x as many in RAM, which removes most of those reads; then PCIe fetches dominate (consumer cards have PCIe 4.0,
  ~2x this PC's gen3). Not measured here - this PC has 30 GB RAM.
- The numbers climb across requests: the LFU adapts from the prior to the live routing (short 128-token runs).
- Nothing is tuned for these budgets yet (no CPU expert path, no smaller dense footprint, no pruning).
