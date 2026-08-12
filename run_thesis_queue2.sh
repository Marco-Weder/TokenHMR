#!/usr/bin/env bash
# Thesis completion queue (~37h). Three phases, each evaluated as soon as it finishes,
# so partial results are usable at any point.
#
#  Phase 1 (~8h)  resume abl_no_queries_lr1e5 79656 -> 200k. Turns the confounded
#                 "- per-token queries" ablation row into a clean one: at head LR 1e-4 the
#                 shape head diverges, at 1e-5 it does not, so the arm finally measures what
#                 the queries contribute to the TOKEN path rather than an optimiser failure.
#  Phase 2 (~14h) CNN (Vanilla) tokenizer under the full classification recipe, 200k.
#  Phase 3 (~14h) Transformer EMA-L2 tokenizer under the same recipe, 200k.
#
# Phases 2-3 add two points to the label-stability trend of Table 4.8, which currently rests
# on FSQ (63% stability) vs cosine-d4 (22%). CNN is 24% and transformer-L2 is 53%, so together
# they span the range and test whether downstream token accuracy tracks label stability.
#
# Settings mirror the existing vqhps_cosd4_0 comparison run exactly (experiment=tokenhmr_fsq_vqhps,
# 200k, full recipe) so the four tokenizers are directly comparable. Only the tokenizer path,
# its code dim / codebook size, and TOKENIZER_TYPE differ.
#
# Tokenizer identities VERIFIED from the codebook weights, not from filenames:
#   30-04-2026_15-47-30  codebook norms 5.07-62.66 (unnormalised) => EMA L2
#   08-05-2026_16-39-44  codebook norms exactly 1.0 (unit sphere) => cosine  [not used here]
# The tier1 run dir's own model_config.yaml points at the COSINE tokenizer because that dir
# held two trainings, so it must not be used as the source for the L2 path.
#
# Deliberately no `set -e`: one failing phase must not cancel the rest.

set -uo pipefail
cd /home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr

PY=/home/marco/miniconda3/envs/thesis-HMR/bin/python
TOK=/home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr/tokenization/output
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/queue2_${STAMP}
mkdir -p "${LOGDIR}"
QLOG="${LOGDIR}/queue.log"

say () { echo "[$(date '+%F %T')] $*" | tee -a "${QLOG}"; }

# Post-run diagnostic: the validation shape loss is the early-warning signal for the
# divergence that wrecked abl_no_queries at head LR 1e-4. Print its trajectory.
beta_curve () {
  ${PY} - "$1" <<'EOF' 2>/dev/null
import glob, sys, json
from wandb.sdk.internal.datastore import DataStore
import wandb.proto.wandb_internal_pb2 as pb
f = sorted(glob.glob(sys.argv[1] + "/wandb/run-*/run-*.wandb"))
if not f:
    print("   (no wandb history)"); raise SystemExit
ds = DataStore(); ds.open_for_scan(f[-1]); pts = []
while True:
    d = ds.scan_data()
    if d is None: break
    r = pb.Record(); r.ParseFromString(d)
    if r.WhichOneof('record_type') == 'history':
        it = {}
        for i in r.history.item:
            k = i.nested_key[0] if i.nested_key else i.key
            try: it[k] = json.loads(i.value_json)
            except Exception: pass
        if 'val/loss_betas' in it and 'trainer/global_step' in it:
            pts.append((int(it['trainer/global_step']), it['val/loss_betas']))
if not pts:
    print("   (no val/loss_betas logged)"); raise SystemExit
step = max(1, len(pts)//8)
print("   val/loss_betas: " + "  ".join("%dk=%.0f" % (s/1000, v) for s, v in pts[::step]))
print("   final=%.0f  (control abl_full ended at ~690; >2000 means the shape head is diverging)" % pts[-1][1])
EOF
}

# Evaluate one run dir. $1=run dir  $2=results prefix  $3=modes (space separated)
do_eval () {
  local run_dir="$1" prefix="$2" modes="$3" ckpt
  ckpt="${run_dir}/checkpoints/last.ckpt"
  if [ ! -f "${ckpt}" ]; then say "  !! no checkpoint in ${run_dir}, skipping eval"; return 1; fi
  for mode in ${modes}; do
    say "  eval ${prefix}_${mode}"
    ${PY} tokenhmr/eval.py \
      --dataset EMDB,3DPW-TEST --batch_size 32 --log_freq 200 \
      --dataset_dir dataset_dir/evaluation_data \
      --checkpoint "${ckpt}" --model_config "${run_dir}/model_config.yaml" \
      --exp_name "${prefix}_${mode}" --decode_mode "${mode}" \
      >> "${LOGDIR}/eval_${prefix}_${mode}.log" 2>&1 \
      || say "  !! eval FAILED: ${prefix} ${mode} (see ${LOGDIR}/eval_${prefix}_${mode}.log)"
  done
}

say "queue2 starting. logs in ${LOGDIR}"
say "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
say "free disk: $(df -h /home/marco | awk 'NR==2{print $4}')"

# ---------------------------------------------------------------- Phase 1
P1DIR=logs/tokenhmr_fsq_vqhps_abl/runs/abl_no_queries_lr1e5
say "PHASE 1/3  resume abl_no_queries_lr1e5 (79656 -> 200k, head LR 1e-5), ~8h"
${PY} tokenhmr/train.py \
  datasets=mix_all experiment=tokenhmr_fsq_vqhps_abl \
  exp_name=abl_no_queries_lr1e5 \
  MODEL.VQHPS_QUERIES=false MODEL.VQHPS_HEAD_LR=1e-5 \
  +resume_path=${P1DIR}/checkpoints/last.ckpt \
  > "${LOGDIR}/train_p1_no_queries_lr1e5.log" 2>&1 \
  || say "  !! PHASE 1 TRAINING FAILED (see ${LOGDIR}/train_p1_no_queries_lr1e5.log)"
say "  phase 1 training done; shape-loss trajectory:"
beta_curve "${P1DIR}" | tee -a "${QLOG}"
do_eval "${P1DIR}" "abl_no_queries_lr1e5" "hard"
say "PHASE 1 COMPLETE"

# ---------------------------------------------------------------- Phase 2
P2DIR=logs/tokenhmr_vqhps_cnn/runs/vqhps_cnn_0
say "PHASE 2/3  CNN (Vanilla) tokenizer under the classification recipe, 200k, ~14h"
${PY} tokenhmr/train.py \
  datasets=mix_all experiment=tokenhmr_fsq_vqhps \
  task_name=tokenhmr_vqhps_cnn exp_name=vqhps_cnn_0 \
  MODEL.TOKENIZER_CHECKPOINT_PATH=${TOK}/tokenization_cnn_amass_moyo_160tokens/tokenization_cnn_amass_moyo_160tokens_ID00_26-04-2026_00-07-33/tokenization_cnn_amass_moyo_160tokens/best_net.pth \
  MODEL.SMPL_HEAD.TOKENIZER.TOKENIZER_TYPE=Vanilla \
  MODEL.SMPL_HEAD.TOKENIZER.TOKEN_NUM=160 \
  MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CLASS_NUM=2048 \
  MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CODE_DIM=256 \
  GENERAL.TOTAL_STEPS=200000 \
  > "${LOGDIR}/train_p2_cnn.log" 2>&1 \
  || say "  !! PHASE 2 TRAINING FAILED (see ${LOGDIR}/train_p2_cnn.log)"
say "  phase 2 training done; shape-loss trajectory:"
beta_curve "${P2DIR}" | tee -a "${QLOG}"
do_eval "${P2DIR}" "vqhps_cnn" "hard soft"
say "PHASE 2 COMPLETE"

# ---------------------------------------------------------------- Phase 3
P3DIR=logs/tokenhmr_vqhps_tfl2/runs/vqhps_tfl2_0
say "PHASE 3/3  Transformer EMA-L2 tokenizer under the classification recipe, 200k, ~14h"
${PY} tokenhmr/train.py \
  datasets=mix_all experiment=tokenhmr_fsq_vqhps \
  task_name=tokenhmr_vqhps_tfl2 exp_name=vqhps_tfl2_0 \
  MODEL.TOKENIZER_CHECKPOINT_PATH=${TOK}/tokenization_transformer_with_fnn_blocks/tokenization_transformer_with_fnn_blocks_ID00_30-04-2026_15-47-30/tokenization_transformer_with_fnn_blocks/best_net.pth \
  MODEL.SMPL_HEAD.TOKENIZER.TOKENIZER_TYPE=transformer \
  MODEL.SMPL_HEAD.TOKENIZER.TOKEN_NUM=160 \
  MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CLASS_NUM=2048 \
  MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CODE_DIM=256 \
  GENERAL.TOTAL_STEPS=200000 \
  > "${LOGDIR}/train_p3_tfl2.log" 2>&1 \
  || say "  !! PHASE 3 TRAINING FAILED (see ${LOGDIR}/train_p3_tfl2.log)"
say "  phase 3 training done; shape-loss trajectory:"
beta_curve "${P3DIR}" | tee -a "${QLOG}"
do_eval "${P3DIR}" "vqhps_tfl2" "hard soft"
say "PHASE 3 COMPLETE"

say "QUEUE FINISHED. results in results/release/{abl_no_queries_lr1e5_hard,vqhps_cnn_*,vqhps_tfl2_*}/"
