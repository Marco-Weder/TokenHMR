#!/usr/bin/env bash
# CONTINUOUS-REGRESSION BASELINE (2026-08-15) -- fills the blank row of Table 4.6
# (tab:downstream-baseline): HMR2.0 + TALS with NO tokenizer, so the tokenized rows
# have a reference point to beat.
#
# experiment=tokenhmr_continuous_hmr2 sets MODEL.SMPL_HEAD.TYPE=transformer_decoder,
# which builds the HMR2.0 head instead of the token head (heads/__init__.py), and
# MODEL.LOOSE_SUP=True keeps TokenHMR's TALS on. Nothing about the tokenizer is used.
#
# SCHEDULE CAVEAT: 400k steps, against the 600k of the eight tokenized rows it sits
# beside. That understates the continuous model, so it must be stated in the caption.
# The config's own LR (5e-7, cosine to its floor) is left untouched.
#
# ~14.1 h per 200k on this card, so ~28 h for 400k, plus ~10 min eval.

set -uo pipefail
cd /home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr

PY=/home/marco/miniconda3/envs/thesis-HMR/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/continuous_${STAMP}
mkdir -p "${LOGDIR}"
QLOG="${LOGDIR}/queue.log"
say () { echo "[$(date '+%F %T')] $*" | tee -a "${QLOG}"; }

TASK=tokenhmr_continuous
EXP=continuous_400k
DIR=logs/${TASK}/runs/${EXP}

while pgrep -f 'train\.py datasets=' >/dev/null 2>&1; do
  say "another training holds the GPU; waiting"; sleep 120
done
say "continuous baseline starting (400k). logs in ${LOGDIR}"
say "  free disk: $(df -h /home/marco | awk 'NR==2{print $4}')   GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

if [ -f "${DIR}/checkpoints/last.ckpt" ]; then
  say "checkpoint already present -> skipping training"
else
  ${PY} tokenhmr/train.py \
    datasets=mix_all experiment=tokenhmr_continuous_hmr2 \
    task_name="${TASK}" exp_name="${EXP}" \
    GENERAL.TOTAL_STEPS=400000 \
    trainer.strategy=null \
    > "${LOGDIR}/train_${EXP}.log" 2>&1 \
    || { say "!! TRAINING FAILED; see ${LOGDIR}/train_${EXP}.log"; exit 1; }
fi

say "eval ${EXP}"
${PY} tokenhmr/eval.py \
  --dataset EMDB,3DPW-TEST --batch_size 32 --log_freq 200 \
  --dataset_dir dataset_dir/evaluation_data \
  --checkpoint "${DIR}/checkpoints/last.ckpt" --model_config "${DIR}/model_config.yaml" \
  --exp_name "${EXP}_hard" --decode_mode hard \
  >> "${LOGDIR}/eval_${EXP}.log" 2>&1 \
  || say "!! eval FAILED (see ${LOGDIR}/eval_${EXP}.log)"

${PY} - "${EXP}" <<'EOF' 2>/dev/null | tee -a "${QLOG}"
import csv, sys, os
p = f"results/release/{sys.argv[1]}_hard/eval_regression.csv"
if os.path.exists(p):
    rows = list(csv.DictReader(open(p)))
    for ds in ('EMDB', '3DPW-TEST'):
        g = {r['metric_name']: r['metric_value'] for r in rows if r['dataset'] == ds}
        keep = {k: v for k, v in g.items() if any(s in k for s in ('_re','mpjpe','pve'))}
        if keep: print(f"   {ds}: " + "  ".join(f"{k}={v}" for k, v in keep.items()))
EOF
say "CONTINUOUS BASELINE FINISHED. results in results/release/${EXP}_hard/"
