#!/usr/bin/env bash
# ADDITIVE CHAIN v2 (2026-08-09) -- rebuilt after v1 phase B reached only 84.05 EMDB PA.
#
# v1 zeroed KEYPOINTS_2D/3D and leaned on CAM_KEYPOINTS_2D, which detaches the 3D joints and
# so cannot supervise the body. See the header of tokenhmr_additive.yaml for the full
# post-mortem. v2 puts the keypoint losses back and moves the severed decode into the base,
# because severing is what lets CE own the token path while the other heads stay supervised.
#
#   B  gate      CE + dataset gate, severed decode     <- run FIRST: make-or-break
#   A  nogate    B without the gate                    <- the control that isolates the gate
#   D  st        B with a straight-through backward
#   E  gumbel    B with a Gumbel-sampled forward index
#
# The soft-decode arms of v1 are gone on purpose: Table 4.7 already reports "Token CE only"
# and "Token CE + pose losses" trained with soft decoding, and Table 4.10 already reports
# both of their hard-decode gaps. Nothing is lost and 28 h are saved.
#
# ~14.3 h training + ~0.3 h eval per run, 4 runs, ~58 h total, sequential (one run fills
# ~24 GB of the 32 GB card). Both decode modes are evaluated after each run. Note that with
# VQHPS_HARD_DECODE the classifier ignores decode_mode at inference (token_classifier.py),
# so the two evals are identical BY CONSTRUCTION, not by anything the model learned. That is
# the same caveat the thesis already states for the classification recipe.
#
# Resumable: a phase whose run dir already holds last.ckpt is skipped, so re-running this
# script after an interruption continues where it stopped.

set -uo pipefail
cd /home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr

PY=/home/marco/miniconda3/envs/thesis-HMR/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/chain2_${STAMP}
mkdir -p "${LOGDIR}"
QLOG="${LOGDIR}/queue.log"

say () { echo "[$(date '+%F %T')] $*" | tee -a "${QLOG}"; }

# --- watchdogs -----------------------------------------------------------------------------
# beta_curve: the split LR is gone, so val/loss_betas should stay flat (abl_full ended ~690,
# the diverging no-queries run hit 12363).
# ce_budget:  v1's failure was invisible in the token metrics and obvious in the loss budget,
# so print the weighted share of each term. KEYPOINTS_3D should sit near 30%, not 0%.
curves () {
  ${PY} - "$1" <<'EOF' 2>/dev/null
import glob, sys, json, yaml, os
from wandb.sdk.internal.datastore import DataStore
import wandb.proto.wandb_internal_pb2 as pb
d = sys.argv[1]
f = sorted(glob.glob(d + "/wandb/run-*/run-*.wandb"))
if not f:
    print("   (no wandb history)"); raise SystemExit
ds = DataStore(); ds.open_for_scan(f[-1]); pts = []; last = None
while True:
    r = ds.scan_data()
    if r is None: break
    rec = pb.Record(); rec.ParseFromString(r)
    if rec.WhichOneof('record_type') == 'history':
        it = {}
        for i in rec.history.item:
            k = "/".join(i.nested_key) if i.nested_key else i.key
            try: it[k] = json.loads(i.value_json)
            except Exception: pass
        if 'val/loss_betas' in it and 'trainer/global_step' in it:
            pts.append((int(it['trainer/global_step']), it['val/loss_betas']))
        if 'train/loss' in it and 'train/loss_betas' in it: last = it
if pts:
    step = max(1, len(pts)//8)
    print("   val/loss_betas: " + "  ".join("%dk=%.0f" % (s/1000, v) for s, v in pts[::step]))
    print("   final=%.0f  (abl_full ended ~690; >2000 = shape head diverging)" % pts[-1][1])
cfg = os.path.join(d, "model_config.yaml")
if last and os.path.exists(cfg):
    lw = yaml.safe_load(open(cfg))['LOSS_WEIGHTS']
    terms = {'KEYPOINTS_3D':'loss_keypoints_3d','KEYPOINTS_2D':'loss_keypoints_2d',
             'GLOBAL_ORIENT':'loss_global_orient','BETAS':'loss_betas',
             'CAM_KEYPOINTS_2D':'loss_cam_2d','TOKEN_CE':'loss_token_ce'}
    rows = []
    for w, k in terms.items():
        raw = last.get('train/' + k)
        if raw is None: continue
        rows.append((w, float(lw.get(w, 0) or 0) * float(raw)))
    tot = sum(c for _, c in rows) or 1.0
    print("   loss budget: " + "  ".join("%s=%.0f%%" % (w, 100*c/tot)
          for w, c in sorted(rows, key=lambda r: -r[1]) if c > 0))
    kp3 = dict(rows).get('KEYPOINTS_3D', 0.0)
    if 100*kp3/tot < 10:
        print("   !! KEYPOINTS_3D is under 10%% of the budget -- this is the v1 failure mode")
EOF
}

do_eval () {                     # $1=run dir  $2=results prefix
  local run_dir="$1" prefix="$2" ckpt="$1/checkpoints/last.ckpt"
  if [ ! -f "${ckpt}" ]; then say "  !! no checkpoint in ${run_dir}, skipping eval"; return 1; fi
  for mode in hard soft; do
    say "  eval ${prefix}_${mode}"
    ${PY} tokenhmr/eval.py \
      --dataset EMDB,3DPW-TEST --batch_size 32 --log_freq 200 \
      --dataset_dir dataset_dir/evaluation_data \
      --checkpoint "${ckpt}" --model_config "${run_dir}/model_config.yaml" \
      --exp_name "${prefix}_${mode}" --decode_mode "${mode}" \
      >> "${LOGDIR}/eval_${prefix}_${mode}.log" 2>&1 \
      || say "  !! eval FAILED: ${prefix} ${mode} (see ${LOGDIR}/eval_${prefix}_${mode}.log)"
  done
  ${PY} - "${prefix}" <<'EOF' 2>/dev/null | tee -a "${QLOG}"
import csv, sys, os
p = f"results/release/{sys.argv[1]}_hard/eval_regression.csv"
if os.path.exists(p):
    rows = list(csv.DictReader(open(p)))
    for ds in ('EMDB', '3DPW-TEST'):
        g = {r['metric_name']: r['metric_value'] for r in rows if r['dataset'] == ds}
        keep = {k: v for k, v in g.items() if 're' in k or 'mpjpe' in k or 'pve' in k or 'top1' in k}
        if keep: print(f"   {ds}: " + "  ".join(f"{k}={v}" for k, v in keep.items()))
EOF
}

# $1=phase letter  $2=task_name  $3=exp_name  $4..=extra hydra overrides
phase () {
  local letter="$1" task="$2" exp="$3"; shift 3
  local dir="logs/${task}/runs/${exp}"
  if [ -f "${dir}/checkpoints/last.ckpt" ]; then
    say "phase ${letter} (${exp}) already has a checkpoint -> skipping training"
  else
    say "phase ${letter}: ${exp}  [$*]"
    say "  free disk: $(df -h /home/marco | awk 'NR==2{print $4}')   GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    ${PY} tokenhmr/train.py \
      datasets=mix_all experiment=tokenhmr_additive \
      task_name="${task}" exp_name="${exp}" \
      GENERAL.TOTAL_STEPS=200000 \
      "$@" \
      > "${LOGDIR}/train_${exp}.log" 2>&1 \
      || { say "  !! TRAINING FAILED (${exp}); see ${LOGDIR}/train_${exp}.log"; return 1; }
  fi
  say "  watchdogs:"; curves "${dir}" | tee -a "${QLOG}"
  do_eval "${dir}" "${exp}"
}

# ---------------------------------------------------------------- preflight
while pgrep -f 'train\.py datasets=' >/dev/null 2>&1; do
  say "another training holds the GPU; waiting"; sleep 120
done
FREE=$(df --output=avail -BG /home/marco | tail -1 | tr -dc '0-9')
say "chain v2 starting. logs in ${LOGDIR}. free disk ${FREE}G (need ~70G for 4 runs)"
if [ "${FREE}" -lt 90 ]; then say "!! ABORT: less than 90G free"; exit 1; fi

# ---------------------------------------------------------------- the chain
# B: CE + dataset gate, severed decode, keypoint losses supervising the non-token heads.
#    If this does not reach the mid-50s, the plan changes.
phase B tokenhmr_chain2_gate chain_gate \
  MODEL.VQHPS_CE_GATE=true

# A: the control for B -- identical, gate off. Isolates the dataset gate on its own.
phase A tokenhmr_chain2_nogate chain_nogate \
  MODEL.VQHPS_CE_GATE=false

# D: B with a straight-through backward (identical argmax forward).
phase D tokenhmr_chain2_st chain_st \
  MODEL.VQHPS_CE_GATE=true MODEL.VQHPS_DECODE=st-argmax MODEL.VQHPS_ST_TAU=1.0

# E: B with a Gumbel-sampled forward index.
phase E tokenhmr_chain2_gumbel chain_gumbel \
  MODEL.VQHPS_CE_GATE=true MODEL.VQHPS_DECODE=gumbel MODEL.VQHPS_ST_TAU=1.0

say "CHAIN v2 FINISHED. results in results/release/{chain_gate,chain_nogate,chain_st,chain_gumbel}_{hard,soft}/"
