#!/bin/bash
# Uranus: Maya-S v2 from the v1 calibration statistics - quantize, pack, then KL against the FP8 reference.
#   setsid nohup bash run_v2.sh > /mnt/nvme/maya/quant/run_v2.log 2>&1 &
set -u
Q=/mnt/nvme/maya/quant
SRC=/mnt/nvme/maya/src
PY=$HOME/.venvs/gguf-convert/bin/python
OUT=$Q/maya-s-v2
N=GLM-5.3-Flash-Maya-S-v2-IQ2_XXS
REF=/mnt/nvme/maya/models/glm53-iq1s/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf   # tensor names and order only
cd $SRC/tools/maya_quant
if [ ! -f $OUT/quantized.ok ]; then
  echo "$(date +%T) quantizing"
  mkdir -p $OUT
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u quantize.py --fp8 /mnt/nvme/maya/glm53-fp8 --stats $Q/stats \
      --ref $REF --recipe recipes/maya-s-v2.json --out $OUT/$N.gguf --gptq \
      --name "GLM-5.3-Flash Maya-S v2" --quantized-by peasantsmith || { echo "$(date +%T) quantize failed"; exit 1; }
  touch $OUT/quantized.ok
fi
if [ ! -f $OUT/pack/index.txt ]; then
  echo "$(date +%T) packing"
  cd $SRC
  MAYA_GGUF_PY=$SRC/build/_deps/strata_llamacpp-src/gguf-py $PY tools/iq_pack.py --gguf $(ls $OUT/$N-00001-of-*.gguf) \
      --out $OUT/pack --compat-bf16 || { echo "$(date +%T) pack failed"; exit 1; }
  cp /mnt/nvme/maya/models/glm53-iq1s/pack/expert_prior.txt /mnt/nvme/maya/models/glm53-iq1s/pack/expert_counts.txt $OUT/pack/
fi
du -sh $OUT
cd $SRC/tools/maya_quant
echo "$(date +%T) KL vs FP8"
$PY -u kl_eval.py --pack $OUT/pack --eval $Q/eval.npy --ref $Q/stats/ref_topk.npz --seqs 0,3,6,9,12,15,18,21 --len 1024
echo "$(date +%T) DONE"
