#!/usr/bin/env bash
# CE-TOKENIZER QUEUE (2026-08-13) -- fills the three projected rows of Table 4.9
# (tab:ce-tokenizer): the gated cross-entropy configuration applied to the three
# learned-codebook tokenizers, so they can be compared against the FSQ row.
#
# WHY: the printed numbers for those rows are currently vqhps_tfl2 / vqhps_cnn /
# vqhps_cosd4 measurements SHIFTED by an offset, because those runs were trained
# under the old four-change VQ-HPS configuration (VQHPS_METHOD=true: per-token
# queries + split LR). This queue retrains them under the SAME two-change
# configuration as chain_gate, so the four rows become directly comparable.
#
#   F  cetok_tfl2    Transformer, EMA (l2),  d=256, K=2048
#   G  cetok_cnn     Convolutional, EMA (l2), d=256, K=2048
#   H  cetok_cosd4   Transformer, EMA (cosine), d=4, K=2048
#
# The FSQ row is chain_gate and is already measured -- it is NOT retrained here.
#
# Everything except the tokenizer is identical to chain_gate: experiment=tokenhmr_additive
# (severed decode, keypoint losses on, VQHPS_METHOD/QUERIES/SPLIT_LR all false,
# one LR of 1e-5), 200k steps, MODEL.VQHPS_CE_GATE=true.
#
# Codebook geometry was read straight off each checkpoint (quantizer.codebook):
#   transformer_with_fnn_blocks -> (2048, 256)
#   cnn_amass_moyo_160tokens    -> (2048, 256)
#   transformer_cosine_dim4     -> (2048, 4)
# This matters because token_classifier.py only ASSERTS that TOKEN_CLASS_NUM matches
# the checkpoint when VQHPS_QUERIES is true, and it is false here. A wrong value would
# silently mispair classes and codes instead of failing.
#
# ~17.6 h training + ~0.1 h eval per run, 3 runs, ~53 h total, sequential.
# Only the hard decode is evaluated: with VQHPS_HARD_DECODE the classifier ignores
# decode_mode at inference, so a soft eval would return identical numbers.
#
# Resumable: a phase whose run dir already holds last.ckpt is skipped.

set -uo pipefail
cd /home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr

PY=/home/marco/miniconda3/envs/thesis-HMR/bin/python
TOKROOT=/home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr/tokenization/output
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/cetok_${STAMP}
mkdir -p "${LOGDIR}"
QLOG="${LOGDIR}/queue.log"

say () { echo "[$(date '+%F %T')] $*" | tee -a "${QLOG}"; }

TOK_TFL2=${TOKROOT}/tokenization_transformer_with_fnn_blocks/tokenization_transformer_with_fnn_blocks_ID00_30-04-2026_15-47-30/tokenization_transformer_with_fnn_blocks/best_net.pth
TOK_CNN=${TOKROOT}/tokenization_cnn_amass_moyo_160tokens/tokenization_cnn_amass_moyo_160tokens_ID00_26-04-2026_00-07-33/tokenization_cnn_amass_moyo_160tokens/best_net.pth
TOK_COS=${TOKROOT}/tokenization_transformer_cosine_dim4/tokenization_transformer_cosine_dim4_ID00_25-06-2026_18-14-16/tokenization_transformer_cosine_dim4/best_net.pth

do_eval () {                     # $1=run dir  $2=results prefix
  local run_dir="$1" prefix="$2" ckpt="$1/checkpoints/last.ckpt"
  if [ ! -f "${ckpt}" ]; then say "  !! no checkpoint in ${run_dir}, skipping eval"; return 1; fi
  say "  eval ${prefix}_hard"
  ${PY} tokenhmr/eval.py \
    --dataset EMDB,3DPW-TEST --batch_size 32 --log_freq 200 \
    --dataset_dir dataset_dir/evaluation_data \
    --checkpoint "${ckpt}" --model_config "${run_dir}/model_config.yaml" \
    --exp_name "${prefix}_hard" --decode_mode hard \
    >> "${LOGDIR}/eval_${prefix}.log" 2>&1 \
    || say "  !! eval FAILED: ${prefix} (see ${LOGDIR}/eval_${prefix}.log)"
  ${PY} - "${prefix}" <<'EOF' 2>/dev/null | tee -a "${QLOG}"
import csv, sys, os
p = f"results/release/{sys.argv[1]}_hard/eval_regression.csv"
if os.path.exists(p):
    rows = list(csv.DictReader(open(p)))
    for ds in ('EMDB', '3DPW-TEST'):
        g = {r['metric_name']: r['metric_value'] for r in rows if r['dataset'] == ds}
        keep = {k: v for k, v in g.items()
                if any(s in k for s in ('_re', 'mpjpe', 'pve', 'top1', 'top5', 'entropy'))}
        if keep:
            print(f"   {ds}: " + "  ".join(f"{k}={v}" for k, v in keep.items()))
EOF
}

# $1=phase  $2=task_name  $3=exp_name  $4=tokenizer ckpt  $5=type  $6=code dim
phase () {
  local letter="$1" task="$2" exp="$3" tok="$4" ttype="$5" cdim="$6"
  local dir="logs/${task}/runs/${exp}"
  if [ ! -f "${tok}" ]; then say "!! phase ${letter}: tokenizer missing: ${tok}"; return 1; fi
  if [ -f "${dir}/checkpoints/last.ckpt" ]; then
    say "phase ${letter} (${exp}) already has a checkpoint -> skipping training"
  else
    say "phase ${letter}: ${exp}  [type=${ttype} d=${cdim} K=2048]"
    say "  free disk: $(df -h /home/marco | awk 'NR==2{print $4}')   GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    ${PY} tokenhmr/train.py \
      datasets=mix_all experiment=tokenhmr_additive \
      task_name="${task}" exp_name="${exp}" \
      GENERAL.TOTAL_STEPS=200000 \
      MODEL.VQHPS_CE_GATE=true \
      MODEL.TOKENIZER_CHECKPOINT_PATH="${tok}" \
      MODEL.SMPL_HEAD.TOKENIZER.TOKENIZER_TYPE="${ttype}" \
      MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CLASS_NUM=2048 \
      MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CODE_DIM="${cdim}" \
      > "${LOGDIR}/train_${exp}.log" 2>&1 \
      || { say "  !! TRAINING FAILED (${exp}); see ${LOGDIR}/train_${exp}.log"; return 1; }
  fi
  do_eval "${dir}" "${exp}"
}

# ---------------------------------------------------------------- preflight
while pgrep -f 'train\.py datasets=' >/dev/null 2>&1; do
  say "another training holds the GPU; waiting"; sleep 120
done
FREE=$(df --output=avail -BG /home/marco | tail -1 | tr -dc '0-9')
say "ce-tokenizer queue starting. logs in ${LOGDIR}. free disk ${FREE}G (need ~50G for 3 runs)"
if [ "${FREE}" -lt 70 ]; then say "!! ABORT: less than 70G free"; exit 1; fi

# ---------------------------------------------------------------- the queue
# Ordered most-informative first, so an interruption still leaves the thesis better off.
# F is the stable l2 transformer (the row that pairs with FSQ to make the "stable" group);
# H is the cosine tokenizer the stability argument predicts will fail.
phase F tokenhmr_cetok_tfl2  cetok_tfl2  "${TOK_TFL2}" transformer 256
phase H tokenhmr_cetok_cosd4 cetok_cosd4 "${TOK_COS}"  transformer 4
phase G tokenhmr_cetok_cnn   cetok_cnn   "${TOK_CNN}"  Vanilla     256

say "CE-TOKENIZER QUEUE FINISHED. results in results/release/cetok_{tfl2,cosd4,cnn}_hard/"
