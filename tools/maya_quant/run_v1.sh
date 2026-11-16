#!/bin/bash
# Uranus: Maya-S v1 end to end once the calibration pass is done - quantize (error-feedback gate/up on the GPU),
# pack for the engine, then eval_v1.sh (KL against the FP8 reference, the loop test).
#   setsid nohup bash run_v1.sh > /mnt/nvme/maya/quant/run_v1.log 2>&1 &
set -u
Q=/mnt/nvme/maya/quant
SRC=/mnt/nvme/maya/src
PY=$HOME/.venvs/gguf-convert/bin/python
OUT=$Q/maya-s-v1
REF=/mnt/nvme/maya/models/glm53-iq1s/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf   # names and metadata only
cd $SRC/tools/maya_quant
echo "$(date +%T) waiting for the calibration pass"
while pgrep -f "calib_drive.py run" > /dev/null; do sleep 60; done
[ -f $Q/stats/ref_topk.npz ] || { echo "$(date +%T) no ref_topk.npz: the calibration pass did not finish"; exit 1; }
if [ ! -f $OUT/quantized.ok ]; then
  echo "$(date +%T) calibration done; quantizing"
  mkdir -p $OUT
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u quantize.py --fp8 /mnt/nvme/maya/glm53-fp8 --stats $Q/stats \
      --ref $REF --recipe recipes/maya-s-v1.json --out $OUT/GLM-5.3-Flash-Maya-S-IQ2_XXS.gguf --gptq \
      --name "GLM-5.3-Flash Maya-S" --quantized-by peasantsmith || { echo "$(date +%T) quantize failed"; exit 1; }
  touch $OUT/quantized.ok
fi
G1=$(ls $OUT/GLM-5.3-Flash-Maya-S-IQ2_XXS*00001-of-*.gguf 2>/dev/null || ls $OUT/GLM-5.3-Flash-Maya-S-IQ2_XXS.gguf)
if [ ! -f $OUT/pack/index.txt ]; then
  echo "$(date +%T) packing $G1"
  cd $SRC
  MAYA_GGUF_PY=$SRC/build/_deps/strata_llamacpp-src/gguf-py $PY tools/iq_pack.py --gguf $G1 --out $OUT/pack --compat-bf16 \
      || { echo "$(date +%T) pack failed"; exit 1; }
  for f in expert_prior.txt expert_counts.txt; do   # this machine's routing profile (any quant of the model)
    cp /mnt/nvme/maya/models/glm53-iq1s/pack/$f $OUT/pack/ 2>/dev/null
  done
fi
du -sh $OUT
bash $SRC/tools/maya_quant/eval_v1.sh $OUT/pack
echo "$(date +%T) DONE"
