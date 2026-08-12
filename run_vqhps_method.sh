#!/usr/bin/env bash
# End-to-end run of the VQ-HPS-method experiment (both stages), unattended.
#
#   stage 1  train the coarse VQ-HPS-scale FSQ tokenizer   (32 tokens x 512 codes, d=3)
#   gate     analyze_token_stability.py — refuse to spend ~43h of stage 2 on an unstable target
#   stage 2  train TokenHMR with MODEL.VQHPS_METHOD=True   (600k steps)
#   eval     EMDB + 3DPW-TEST (hard decode; under VQHPS_METHOD soft==hard by construction)
#
# The tokenizer checkpoint path is discovered from stage 1's output dir and passed to stage 2
# as a Hydra override, so the FILL_AFTER_STAGE1 placeholder in the experiment yaml is bypassed
# and no file needs editing between the stages.
#
# Resumable: each phase is skipped if its output already exists. Deliberately no `set -e` —
# a failing phase must report rather than silently cancel the rest.
#
# Usage:  bash run_vqhps_method.sh            # full pipeline
#         SKIP_STAGE1=1 bash run_vqhps_method.sh   # reuse an existing tokenizer of that name
#
# The stage-1 candidate is selected by env vars (defaults = candidate A after the 32x512
# gate failure of 31-07-2026: keep 160 tokens, shrink the codebook to 512):
#   STAGE1_CFG=configs/tokenizer_amass_moyo_fsq_160x512_masked.yaml \
#   TOKENIZER_NAME=tokenization_transformer_fsq_160x512_masked bash run_vqhps_method.sh
# Stage 2's TOKEN_NUM / TOKEN_CLASS_NUM / TOKEN_CODE_DIM are derived from the trained
# checkpoint's hparams, so they always match the gated tokenizer.

set -uo pipefail
cd /home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr

PY=/home/marco/miniconda3/envs/thesis-HMR/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/vqhps_method_${STAMP}
mkdir -p "${LOGDIR}"
QUEUE_LOG="${LOGDIR}/queue.log"

STAGE1_CFG=${STAGE1_CFG:-configs/tokenizer_amass_moyo_fsq_160x512.yaml}
TOKENIZER_NAME=${TOKENIZER_NAME:-tokenization_transformer_fsq_160x512}
STAGE2_RUN=logs/tokenhmr_fsq_vqhps/runs/fsq_vqhps_0

say () { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"; }

find_tokenizer () {
  ls -d tokenization/output/${TOKENIZER_NAME}/${TOKENIZER_NAME}_ID00_*/${TOKENIZER_NAME}/best_net.pth 2>/dev/null | tail -1
}

say "pipeline starting; logs in ${LOGDIR}"

# ---------------------------------------------------------------- stage 1
if [ "${SKIP_STAGE1:-0}" = "1" ] && [ -n "$(find_tokenizer)" ]; then
  say "stage 1: skipped (SKIP_STAGE1=1, reusing $(find_tokenizer))"
else
  say "stage 1: training ${TOKENIZER_NAME} (${STAGE1_CFG}, 400k iters)"
  ( cd tokenization && ${PY} train_poseVQ.py \
      --cfg "${STAGE1_CFG}" --cfg_id 0 ) \
    > "${LOGDIR}/stage1_tokenizer.log" 2>&1 \
    || say "  !! stage 1 training failed (see ${LOGDIR}/stage1_tokenizer.log)"
fi

TOKENIZER_CKPT=$(find_tokenizer)
if [ -z "${TOKENIZER_CKPT}" ]; then
  say "FATAL: no fsq_vqhps tokenizer checkpoint found — cannot continue to stage 2"
  exit 1
fi
TOKENIZER_CKPT=$(realpath "${TOKENIZER_CKPT}")
say "stage 1 done; tokenizer = ${TOKENIZER_CKPT}"

# ---------------------------------------------------------------- gate
# stab@1° is the CE-learnability bar: an image model cannot resolve a label the encoder itself
# will not hold fixed under sub-annotation-noise perturbation. The 160x1920 FSQ sat at 63%,
# which put top-1 below the dataset prior. Target here is >= 85%.
say "gate: token stability of the new tokenizer (vs the old 160x1920 FSQ for reference)"
# realpath BEFORE the subshell cd — inside `( cd tokenization ... )` this relative path
# no longer resolves, which is exactly the bug that crashed the 2026-07-31 gate run.
OLD_FSQ=$(realpath tokenization/output/tokenization_transformer_fsq/tokenization_transformer_fsq_ID00_13-05-2026_00-29-18/tokenization_transformer_fsq/best_net.pth)
( cd tokenization && ${PY} analyze_token_stability.py \
    --ckpt "${TOKENIZER_CKPT}" --label "fsq vqhps" \
    --ckpt "${OLD_FSQ}" --label "fsq d4 160x1920" \
    --num-poses 2048 --seed 0 --latex \
    --jitter 0.5 --jitter 1 --jitter 2 --jitter 5 \
    --out output/token_stability_vqhps ) \
  > "${LOGDIR}/gate_stability.log" 2>&1 \
  || { say "FATAL: stability analysis failed (see ${LOGDIR}/gate_stability.log) — refusing to start stage 2 ungated"; exit 1; }

if [ -f tokenization/output/token_stability_vqhps/table.md ]; then
  say "stability table:"
  tee -a "${QUEUE_LOG}" < tokenization/output/token_stability_vqhps/table.md
fi

# Hard gate: parse stab@1 deg of the new tokenizer from summary.json and refuse stage 2 below
# the threshold. Override with GATE_MIN_STAB1= (percent) or FORCE_STAGE2=1 to bypass.
GATE_MIN_STAB1=${GATE_MIN_STAB1:-85}
STAB1=$(${PY} -c "
import json
d = json.load(open('tokenization/output/token_stability_vqhps/summary.json'))
r = [x for x in d['runs'] if x['label'] == 'fsq vqhps'][0]
print(f\"{r['stability']['1']['ids_unchanged_pct']:.1f}\")")
say "gate: new tokenizer stab@1 deg = ${STAB1}% (threshold ${GATE_MIN_STAB1}%)"
if [ "${FORCE_STAGE2:-0}" != "1" ] && \
   ${PY} -c "import sys; sys.exit(0 if float('${STAB1}') < float('${GATE_MIN_STAB1}') else 1)"; then
  say "GATE FAILED: stab@1 deg ${STAB1}% < ${GATE_MIN_STAB1}% — NOT starting stage 2."
  say "The 32x512 run showed FEWER tokens make stability WORSE (redundancy is what creates"
  say "stable IDs). Try the masked candidate (STAGE1_CFG=configs/tokenizer_amass_moyo_fsq_160x512_masked.yaml"
  say "TOKENIZER_NAME=tokenization_transformer_fsq_160x512_masked), or accept a lower bar via"
  say "GATE_MIN_STAB1=<pct>, or rerun with FORCE_STAGE2=1."
  exit 1
fi

# ---------------------------------------------------------------- stage 2
# Token geometry comes from the checkpoint itself (single source of truth); the values in
# the experiment yaml are only documentation. TokenClassfier re-asserts the match at load.
read -r TOKEN_NUM TOKEN_CLASS TOKEN_DIM <<< "$(${PY} -c "
import torch, numpy as np
a = torch.load('${TOKENIZER_CKPT}', map_location='cpu', weights_only=False)['hparams'].ARCH
lv = list(a.FSQ_LEVELS[0]) if isinstance(a.FSQ_LEVELS[0], (list, tuple)) else list(a.FSQ_LEVELS)
print(a.NUM_TOKENS, int(np.prod(lv)), len(lv))")"
say "stage 2 token geometry from ckpt: TOKEN_NUM=${TOKEN_NUM} TOKEN_CLASS_NUM=${TOKEN_CLASS} TOKEN_CODE_DIM=${TOKEN_DIM}"

if [ -f "${STAGE2_RUN}/checkpoints/epoch=9-step=600000.ckpt" ]; then
  say "stage 2: skipped (600k checkpoint already exists)"
else
  say "stage 2: training tokenhmr_fsq_vqhps (600k steps, ~43h)"
  ${PY} tokenhmr/train.py datasets=mix_all experiment=tokenhmr_fsq_vqhps \
      MODEL.TOKENIZER_CHECKPOINT_PATH="${TOKENIZER_CKPT}" \
      MODEL.SMPL_HEAD.TOKENIZER.TOKEN_NUM="${TOKEN_NUM}" \
      MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CLASS_NUM="${TOKEN_CLASS}" \
      MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CODE_DIM="${TOKEN_DIM}" \
    > "${LOGDIR}/stage2_train.log" 2>&1 \
    || say "  !! stage 2 training failed (see ${LOGDIR}/stage2_train.log)"
fi

# ---------------------------------------------------------------- eval
CKPT=$(ls "${STAGE2_RUN}"/checkpoints/*step=600000.ckpt 2>/dev/null | head -1)
[ -z "${CKPT}" ] && CKPT="${STAGE2_RUN}/checkpoints/last.ckpt"
if [ ! -f "${CKPT}" ]; then
  say "!! no stage-2 checkpoint to evaluate"
  exit 1
fi

# Both modes are recorded for table symmetry with the earlier runs, but under VQHPS_METHOD the
# decode is always hard argmax — the two rows should be identical, which is itself the check
# that the soft/hard gap (baseline: +140mm) is gone.
for mode in hard soft; do
  if [ -f "results/release/fsq_vqhps_${mode}/eval_regression.csv" ]; then
    say "  skip eval ${mode} (already have results; eval.py appends)"
    continue
  fi
  say "eval ${mode} using $(basename "${CKPT}")"
  ${PY} tokenhmr/eval.py \
    --dataset EMDB,3DPW-TEST --batch_size 32 --log_freq 200 \
    --dataset_dir dataset_dir/evaluation_data \
    --checkpoint "${CKPT}" --model_config "${STAGE2_RUN}/model_config.yaml" \
    --exp_name "fsq_vqhps_${mode}" --decode_mode "${mode}" \
    > "${LOGDIR}/eval_${mode}.log" 2>&1 \
    || say "  !! eval failed: ${mode} (see ${LOGDIR}/eval_${mode}.log)"
done

say "PIPELINE COMPLETE"
say "results:  grep -h . results/release/fsq_vqhps_*/eval_regression.csv"
say "compare against baseline 57.06 EMDB PA-MPJPE (results/release/fsq_baseline_full_*/) and"
say "ce_soft 73.68 (results/release/*ce_soft*/). Success = at or below 57.06 with top-1 well"
say "above the new tokenizer's marginal prior."
