#!/usr/bin/env bash
# Unattended experiment queue (~35h). Each arm is evaluated as soon as it finishes, so partial
# results are usable at any point -- if you come back early, whatever has completed is already
# in results/release/*/eval_regression.csv.
#
# Phase 0 (~20m) fills the two missing baseline cells in the comparison table.
# Phase 1 (~19h) TOKEN_CE weight sweep, 5 arms  -> isolates the CE WEIGHT, target shape fixed.
# Phase 2 (~16h) Part 4 decoder-aware soft CE, 2 taus -> isolates the TARGET SHAPE, weight fixed.
#
# Together those two phases separate the two candidate explanations for ce_soft's +16.6mm EMDB
# PA-MPJPE deficit against the pose-loss-only baseline.
#
# Deliberately does NOT use `set -e`: one failing arm must not cancel the rest of the queue.

set -uo pipefail
cd /home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr

PY=/home/marco/miniconda3/envs/thesis-HMR/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/unattended_${STAMP}
mkdir -p "${LOGDIR}"
QUEUE_LOG="${LOGDIR}/queue.log"

say () { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"; }

# Evaluate one run directory in both decode modes.
#   $1 = run dir, $2 = results exp_name prefix, $3 = expected final step
do_eval () {
  local run_dir="$1" prefix="$2" step="$3"
  local ckpt
  ckpt=$(ls "${run_dir}"/checkpoints/*step=${step}.ckpt 2>/dev/null | head -1)
  if [ -z "${ckpt}" ]; then
    ckpt="${run_dir}/checkpoints/last.ckpt"
  fi
  if [ ! -f "${ckpt}" ]; then
    say "  !! no checkpoint under ${run_dir}, skipping eval"
    return 1
  fi
  say "  eval ${prefix} using $(basename "${ckpt}")"
  for mode in soft hard; do
    ${PY} tokenhmr/eval.py \
      --dataset EMDB,3DPW-TEST --batch_size 32 --log_freq 200 \
      --dataset_dir dataset_dir/evaluation_data \
      --checkpoint "${ckpt}" --model_config "${run_dir}/model_config.yaml" \
      --exp_name "${prefix}_${mode}" --decode_mode "${mode}" \
      >> "${LOGDIR}/eval_${prefix}_${mode}.log" 2>&1 \
      || say "  !! eval failed: ${prefix} ${mode} (see ${LOGDIR}/eval_${prefix}_${mode}.log)"
  done
}

say "queue starting; logs in ${LOGDIR}"

# ---------------------------------------------------------------- phase 0
# The baseline's hard-decode eval stopped after EMDB (no 3DPW rows), and its original May eval
# predates token_metrics so it has no entropy/top-k. Re-run both to complete the table.
say "phase 0: baseline eval gaps"
BASE_RUN=logs/tokenhmr_fsq/runs/tokenhmr_fsq_0
BASE_CKPT=${BASE_RUN}/checkpoints/epoch=9-step=600000.ckpt
for mode in soft hard; do
  # Idempotent: eval.py APPENDS, so re-running a completed mode would duplicate rows.
  if [ -f "results/release/fsq_baseline_full_${mode}/eval_regression.csv" ]; then
    say "  skip baseline ${mode} (already have results)"
    continue
  fi
  ${PY} tokenhmr/eval.py \
    --dataset EMDB,3DPW-TEST --batch_size 32 --log_freq 200 \
    --dataset_dir dataset_dir/evaluation_data \
    --checkpoint "${BASE_CKPT}" --model_config "${BASE_RUN}/model_config.yaml" \
    --exp_name "fsq_baseline_full_${mode}" --decode_mode "${mode}" \
    >> "${LOGDIR}/eval_baseline_${mode}.log" 2>&1 \
    || say "  !! baseline ${mode} eval failed"
done
say "phase 0 done"

# ---------------------------------------------------------------- phase 1
# cls_id -> TOKEN_CE: 0->1.0 (CONTROL, must land near ce_soft's EMDB 73.68/110.08 or the LR is
# too hot and the sweep is measuring the LR restart), 1->0.3, 2->0.1, 3->0.01, 4->0.0.
# Order is deliberate, not numeric. cls_id=4 (TOKEN_CE=0.0, CE off entirely) is the DECISIVE
# arm and runs first among the remaining ones: a from-scratch pose-only model sits at token
# entropy 6.5, so if this arm -- fine-tuned from ce_soft with no CE at all -- still sits near
# 2.9, the model cannot traverse the entropy frontier by fine-tuning at all, and phase 2's
# Part 4 arms would be equally stuck. Getting that answer early is worth more than arm order.
say "phase 1: TOKEN_CE weight sweep (40k fine-tunes)"
for CID in 4; do
  if [ -f "results/release/cesweep_${CID}_hard/eval_regression.csv" ]; then
    say "  skip arm cls_id=${CID} (already evaluated)"
    continue
  fi
  say "  arm cls_id=${CID} training"
  ${PY} tokenhmr/train.py datasets=mix_all \
    experiment=tokenhmr_fsq_ce_weight_sweep cls_id=${CID} \
    > "${LOGDIR}/train_cesweep_${CID}.log" 2>&1 \
    || say "  !! training failed for cls_id=${CID}"
  do_eval "logs/tokenhmr_fsq_ce_weight_sweep/runs/fsq_cesweep_${CID}" "cesweep_${CID}" 40000
done
say "phase 1 done"

# ---------------------------------------------------------------- phase 2
# ---------------------------------------------------------------- phase 1b
# THE PRIORITY. From-scratch, matched-budget TOKEN_CE comparison: 1.0 vs 0.1, 200k steps each.
# The 40k fine-tunes showed a complete null (CE 2.887->2.867, entropy 2.974->3.025, top-1
# 0.383->0.386 for a 3.3x weight cut) with a control that reproduced ce_soft to 0.2mm -- i.e.
# the protocol worked but the model would not move. Training each arm from scratch lets it
# reach its own equilibrium, so a difference here is causal. Read the DELTA between the two
# arms; absolute values are not comparable to the 600k runs.
say "phase 1b: from-scratch TOKEN_CE comparison (2 arms x 200k steps, ~14.4h each)"
for CID in 0 1; do
  if [ -f "results/release/scratch_${CID}_hard/eval_regression.csv" ]; then
    say "  skip scratch arm cls_id=${CID} (already evaluated)"
    continue
  fi
  say "  scratch arm cls_id=${CID} training"
  ${PY} tokenhmr/train.py datasets=mix_all \
    experiment=tokenhmr_fsq_ce_scratch cls_id=${CID} \
    > "${LOGDIR}/train_scratch_${CID}.log" 2>&1 \
    || say "  !! training failed for scratch cls_id=${CID}"
  do_eval "logs/tokenhmr_fsq_ce_scratch/runs/fsq_scratch_${CID}" "scratch_${CID}" 200000
done
say "phase 1b done"

# ---------------------------------------------------------------- phase 2
# LOWEST PRIORITY, runs only if time remains. Subject to the same hysteresis that flattened the
# phase 1 fine-tunes, so a null here is uninformative -- but tau=5.0 is the value the proposal
# specifies, so it is worth having if the queue gets that far. Order: 0.1 (best-calibrated),
# then 5.0 (the proposal's), then 0.02.
# TOKEN_CE held at 1.0 (== sweep cls_id=0) and the step count matches phase 1, so this varies
# ONLY the target shape. cls_id -> tau: 0 -> 0.1mm (target entropy ~2.23 nats),
# 1 -> 0.02mm (~1.14 nats), 2 -> 5.0mm (the proposal's suggested value; measured entropy 2.772
# against the ln(16)=2.773 ceiling, i.e. an essentially UNIFORM target over the candidates).
# 5.0 is ordered last on purpose: it is the least likely to help and the most likely to be cut
# if the queue runs out of time, but it is worth measuring because the proposal specifies it.
say "phase 2: Part 4 decoder-aware soft CE (3 taus x 40k steps)"
for CID in 0 2 1; do
  if [ -f "results/release/softce_${CID}_hard/eval_regression.csv" ]; then
    say "  skip arm cls_id=${CID} (already evaluated)"
    continue
  fi
  say "  arm cls_id=${CID} training"
  ${PY} tokenhmr/train.py datasets=mix_all \
    experiment=tokenhmr_fsq_soft_ce_finetune cls_id=${CID} \
    > "${LOGDIR}/train_softce_${CID}.log" 2>&1 \
    || say "  !! training failed for softce cls_id=${CID}"
  do_eval "logs/tokenhmr_fsq_soft_ce_finetune/runs/fsq_softce_ft_${CID}" "softce_${CID}" 40000
done
say "phase 2 done"

say "QUEUE COMPLETE"
say "collect results with:  grep -h . results/release/{fsq_baseline_full,cesweep,softce}*/eval_regression.csv"
