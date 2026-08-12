#!/usr/bin/env bash
# Re-run of queue2's PHASE 2 (CNN / "Vanilla" tokenizer under the classification recipe, 200k),
# which crashed 20s in on 2026-08-08 and left the label-stability trend of Table 4.8 one point short.
#
# The crash was NOT a config problem: `_get_gt_encoder` (token_metrics.py) hardcoded
# TransformerTokenizer, so loading the CNN tokenizer's checkpoint tripped its strict-key guard on
# the first compute_loss. Fixed by dispatching on ARCH.MODEL_NAME, the same key the tokenizer's own
# trainer uses (train_poseVQ.get_model). Fixing that exposed a second, latent bug:
# VanillaTokenizer.encode() never applied the quantizer's NCT -> [N*T, C] reshape (forward() gets it
# free from QuantizeEMAReset.forward), so it had never worked. Both are fixed and verified:
# encode() now reproduces the forward path's codes exactly, and decoding them reproduces forward()'s
# reconstruction to 2e-7.
#
# Training flags are byte-identical to queue2's phase 2 so the CNN row stays comparable with
# vqhps_cosd4_0 / vqhps_fsq / vqhps_tfl2. Tokenizer hparams cross-checked against the checkpoint:
# ARCH.MODEL_NAME=vanilla, NB_CODE=2048, CODE_DIM=256, NUM_TOKENS=160.
#
# Waits for the queue2 driver to finish (phase 3 tfl2 is mid-run) before touching the GPU.

set -uo pipefail
cd /home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr

PY=/home/marco/miniconda3/envs/thesis-HMR/bin/python
TOK=/home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr/tokenization/output
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/queue3_${STAMP}
mkdir -p "${LOGDIR}"
QLOG="${LOGDIR}/queue.log"
WAIT_PID="${1:-}"          # optional: queue2 driver PID to wait on

say () { echo "[$(date '+%F %T')] $*" | tee -a "${QLOG}"; }

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

say "queue3 starting. logs in ${LOGDIR}"

# ------------------------------------------------------- wait for the GPU
# Only wait on the PID if it really is the queue2 driver (guards against PID reuse).
if [ -n "${WAIT_PID}" ] && [ -r "/proc/${WAIT_PID}/cmdline" ] \
   && tr '\0' ' ' < "/proc/${WAIT_PID}/cmdline" | grep -q run_thesis_queue2.sh; then
  say "waiting for queue2 driver (PID ${WAIT_PID}) to finish; it is mid phase-3 (tfl2)"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 120; done
  say "queue2 driver exited"
fi
# Belt and braces: never start while any training still holds the GPU.
while pgrep -f 'tokenhmr/train.py' >/dev/null 2>&1; do sleep 120; done
say "GPU free. free disk: $(df -h /home/marco | awk 'NR==2{print $4}')"

# ------------------------------------------------------- CNN (Vanilla) tokenizer, 200k
P2DIR=logs/tokenhmr_vqhps_cnn/runs/vqhps_cnn_0
# The crashed attempt left configs + an empty wandb run in this dir. Archive it so the run dir and
# its beta_curve/wandb history unambiguously belong to this training.
if [ -d "${P2DIR}" ] && [ ! -d "${P2DIR}/checkpoints" ]; then
  mv "${P2DIR}" "${P2DIR}.crashed_20260808_063554"
  say "archived crashed attempt -> ${P2DIR}.crashed_20260808_063554"
fi

say "CNN (Vanilla) tokenizer under the classification recipe, 200k, ~14h"
${PY} tokenhmr/train.py \
  datasets=mix_all experiment=tokenhmr_fsq_vqhps \
  task_name=tokenhmr_vqhps_cnn exp_name=vqhps_cnn_0 \
  MODEL.TOKENIZER_CHECKPOINT_PATH=${TOK}/tokenization_cnn_amass_moyo_160tokens/tokenization_cnn_amass_moyo_160tokens_ID00_26-04-2026_00-07-33/tokenization_cnn_amass_moyo_160tokens/best_net.pth \
  MODEL.SMPL_HEAD.TOKENIZER.TOKENIZER_TYPE=Vanilla \
  MODEL.SMPL_HEAD.TOKENIZER.TOKEN_NUM=160 \
  MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CLASS_NUM=2048 \
  MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CODE_DIM=256 \
  GENERAL.TOTAL_STEPS=200000 \
  > "${LOGDIR}/train_cnn.log" 2>&1 \
  || say "  !! CNN TRAINING FAILED (see ${LOGDIR}/train_cnn.log)"
say "  training done; shape-loss trajectory:"
beta_curve "${P2DIR}" | tee -a "${QLOG}"
do_eval "${P2DIR}" "vqhps_cnn" "hard soft"

say "QUEUE3 FINISHED. results in results/release/vqhps_cnn_{hard,soft}/"
