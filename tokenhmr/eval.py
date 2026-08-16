import argparse
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' #'osmesa'
from pathlib import Path
import traceback
from typing import Optional
from tokenhmr.lib.utils import Evaluator, recursive_to
import pandas as pd
import cv2
import numpy as np
import torch
from filelock import FileLock
import smplx
from repro import paths
from tokenhmr.lib.configs import dataset_eval_config
from tokenhmr.lib.datasets import create_dataset

from tqdm import tqdm
from tokenhmr.lib.models import load_tokenhmr
from tokenhmr.lib.utils import MeshRenderer
from tokenhmr.lib.models.smpl_wrapper import SMPL
from tokenhmr.lib.utils.token_metrics import compute_token_metrics, _get_gt_encoder
from tokenhmr.lib.utils.geometry import aa_to_rotmat

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to pretrained model checkpoint')
    parser.add_argument('--model_config', type=str, default='model_config.yaml', help='Path to model config file')
    parser.add_argument('--results_file', type=str, default='eval_regression.csv', help='Path to results file.')
    parser.add_argument('--dataset', type=str, default='EMDB, 3DPW-TEST', help='Dataset to evaluate') 
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Evaluation data folder '
                             '(default: <project root>/dataset_dir/evaluation_data)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
    parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
    parser.add_argument('--exp_name', type=str, default=None, required=True,
                        help='Experiment name; names the results directory')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory for results and renders '
                             '(default: <project root>/results/release/<exp_name>)')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--decode_mode', type=str, default='soft', choices=['soft', 'hard'],
                        help="Token decoding at inference: 'soft' (softmax-weighted codebook mix) "
                             "or 'hard' (argmax one-hot codebook lookup).")

    args = parser.parse_args()

    # --exp_name used to default to 'eval', so unnamed evaluations all appended
    # to one file and overwrote each other's provenance. It is required now.
    if not args.exp_name:
        parser.error("--exp_name is required, it names the results directory")
    # Used to default to '', which made every dataset path relative to the
    # working directory and required eval.py to be launched from one place.
    if args.dataset_dir is None:
        args.dataset_dir = str(paths.DATASET_DIR / "evaluation_data")
    exp_name = args.exp_name
    results_dir = args.out_dir or str(paths.RELEASE_DIR / exp_name)
    render_dir = f'{results_dir}/render'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    args.render_dir = render_dir
    args.results_file = os.path.join(results_dir, args.results_file)

    # Download and load checkpoints
    model, model_cfg = load_tokenhmr(checkpoint_path=args.checkpoint, \
                                 model_cfg=args.model_config, dataset_dir=args.dataset_dir)

    # Setup model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Inference decoding-strategy toggle (soft vs hard argmax). Only the token head has it.
    decpose = getattr(getattr(model, 'smpl_head', None), 'decpose', None)
    if decpose is not None and getattr(decpose, 'vqhps_method', False):
        # VQHPS_METHOD always decodes argmax one-hot (train and eval alike), so there is no
        # soft/hard distinction to toggle: both runs measure the same hard decode.
        print(f'decode_mode = hard (VQHPS_METHOD; --decode_mode {args.decode_mode} has no effect)')
    elif decpose is not None and hasattr(decpose, 'decode_mode'):
        decpose.decode_mode = args.decode_mode
        print(f'decode_mode = {decpose.decode_mode}')
    elif args.decode_mode != 'soft':
        print(f'WARNING: --decode_mode {args.decode_mode} ignored (model has no token decode head)')
    print('model loaded!')
    print('using', args.checkpoint)
    # Load config and run eval, one dataset at a time
    print('Evaluating on datasets: {}'.format(args.dataset), flush=True)
    for dataset in args.dataset.split(','):
        dataset_cfg = dataset_eval_config()[dataset]
        if 'DATASET_FILE' in dataset_cfg:
            dataset_cfg['DATASET_FILE'] = os.path.join(args.dataset_dir, dataset_cfg['DATASET_FILE'])
        if 'IMG_DIR' in dataset_cfg:
            dataset_cfg['IMG_DIR'] = os.path.join(args.dataset_dir, dataset_cfg['IMG_DIR'])
        args.dataset = dataset
        print(dataset)
        run_eval(model, model_cfg, dataset_cfg, device, args)

def render_predictions(args, dataset, batch, output, mesh_renderer, idx):
    batch_size = batch['keypoints_2d'].shape[0]
    images = batch['img']
    images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
    gt_keypoints_2d = batch['keypoints_2d']

    pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
    focal_length = output['focal_length'].detach().reshape(batch_size, 2)
    
    pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
    pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)
    num_images = min(batch_size, 8)
    
    predictions = mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                      pred_cam_t[:num_images].cpu().numpy(),
                                                      images[:num_images].cpu().numpy(),
                                                      pred_keypoints_2d[:num_images].cpu().numpy(),
                                                      gt_keypoints_2d[:num_images].cpu().numpy(),
                                                      focal_length=focal_length[:num_images].cpu().numpy())
    
    predictions = predictions.cpu().numpy().transpose(1,2,0)*255
    predictions = np.clip(predictions, 0, 255).astype(np.uint8)
    
    if 'pred_vertices_gt' in output:
        pred_vertices = output['pred_vertices_gt'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d_gt'].detach().reshape(batch_size, -1, 2)
        num_images = min(batch_size, 8)
        
        predictions_gt = mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                        pred_cam_t[:num_images].cpu().numpy(),
                                                        images[:num_images].cpu().numpy(),
                                                        pred_keypoints_2d[:num_images].cpu().numpy(),
                                                        gt_keypoints_2d[:num_images].cpu().numpy(),
                                                        focal_length=focal_length[:num_images].cpu().numpy())
        predictions_gt = predictions_gt.cpu().numpy().transpose(1,2,0)*255
        predictions_gt = np.clip(predictions_gt, 0, 255).astype(np.uint8)
        predictions = np.concatenate([predictions, predictions_gt[:,256:256*4]],1)
    
    cv2.imwrite(os.path.join(args.render_dir, f'render_{dataset}_{idx}.png'),
                cv2.cvtColor(predictions, cv2.COLOR_BGR2RGB))
    return predictions


def run_eval(model, model_cfg, dataset_cfg, device, args):

    if args.render:
        smpl_cfg = {k.lower(): v for k,v in dict(model_cfg.SMPL).items()}
        smpl = SMPL(**smpl_cfg)
        mesh_renderer = MeshRenderer(model_cfg, smpl.faces)

    # Create dataset and data loader
    dataset = create_dataset(model_cfg, dataset_cfg, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    J_regressor_24_SMPL = None
    # List of metrics to log
    if args.dataset in ['EMDB', '3DPW-TEST']:
        metrics = ['mode_re', 'mode_mpjpe', 'mode_pve']
        J_regressor_24_SMPL=smplx.SMPL(model_path=model_cfg.SMPL.MODEL_PATH).J_regressor.cuda().float()
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    # Setup evaluator object
    evaluator = Evaluator(
        dataset_length=int(1e8), 
        keypoint_list=dataset_cfg.KEYPOINT_LIST, 
        pelvis_ind=model_cfg.EXTRA.PELVIS_IND, 
        metrics=metrics,
        J_regressor_24_SMPL = J_regressor_24_SMPL,
        dataset=args.dataset,
    )

    # Token-predictor metrics (decode-mode-invariant): CE / top-1 / top-5 / pred-entropy vs the
    # GT tokens (GT SMPL body pose encoded through the frozen tokenizer). Accumulated as
    # valid-sample-weighted means so they aggregate correctly across batches.
    ckpt_path = model_cfg.MODEL.get('TOKENIZER_CHECKPOINT_PATH', None)
    tok_sum = {'token_ce': 0.0, 'token_top1_acc': 0.0, 'token_top5_acc': 0.0}
    tok_n = 0
    ent_sum, ent_n = 0.0, 0

    for i, batch in enumerate(tqdm(dataloader)):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        evaluator(out, batch)

        # Token metrics (only for the token head, only over samples with a GT body pose).
        if ckpt_path is not None and 'cls_logits_softmax' in out:
            try:
                bp = batch['smpl_params']['body_pose']
                B = bp.shape[0]
                bp = bp.view(B, -1)
                if batch['smpl_params_is_axis_angle']['body_pose'].all():
                    gt_bp_rotmat = aa_to_rotmat(bp.reshape(-1, 3)).view(B, -1, 3, 3)
                else:
                    gt_bp_rotmat = bp.view(B, -1, 3, 3)
                has_gt = batch['has_smpl_params']['body_pose'].reshape(B, -1)[:, 0] > 0.5
                tm = compute_token_metrics(out['cls_logits_softmax'], gt_bp_rotmat, has_gt, ckpt_path)
                ent_sum += float(tm['token_pred_entropy']) * B
                ent_n += B
                n_valid = int(has_gt.sum())
                if n_valid > 0 and 'token_top1_acc' in tm:
                    logits = out['cls_logits']
                    logits = logits[-B:] if logits.shape[0] != B else logits
                    enc = _get_gt_encoder(ckpt_path, logits.device)
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                        gt_tok = enc.encode(gt_bp_rotmat[:, :enc.num_joints].float())
                    ce = torch.nn.functional.cross_entropy(
                        logits[has_gt].reshape(-1, logits.shape[-1]).float(),
                        gt_tok[has_gt].reshape(-1))
                    tok_sum['token_ce'] += float(ce) * n_valid
                    tok_sum['token_top1_acc'] += float(tm['token_top1_acc']) * n_valid
                    tok_sum['token_top5_acc'] += float(tm['token_top5_acc']) * n_valid
                    tok_n += n_valid
            except Exception as e:
                if i == 0:
                    print(f'Token metrics skipped: {e}')

        if i % args.log_freq == args.log_freq - 1:
            evaluator.log()
            if args.render:
                predictions = render_predictions(args, args.dataset, batch, out, mesh_renderer, i)
    evaluator.log()
    error = None

    # Append results to file (regression metrics + token metrics under this decode mode).
    metrics_dict = evaluator.get_metrics_dict()
    metrics_dict = {f'{args.decode_mode}_{k}': v for k, v in metrics_dict.items()}
    if tok_n > 0:
        metrics_dict['token_ce'] = tok_sum['token_ce'] / tok_n
        metrics_dict['token_top1_acc'] = tok_sum['token_top1_acc'] / tok_n
        metrics_dict['token_top5_acc'] = tok_sum['token_top5_acc'] / tok_n
    if ent_n > 0:
        metrics_dict['token_pred_entropy'] = ent_sum / ent_n
    save_eval_result(args.results_file, metrics_dict, args.checkpoint, args.dataset, error=error, iters_done=i, exp_name=args.exp_name)


def save_eval_result(
    csv_path: str,
    metric_dict: float,
    checkpoint_path: str,
    dataset_name: str,
    # start_time: pd.Timestamp,
    error: Optional[str] = None,
    iters_done=None,
    exp_name=None,
) -> None:
    """Save evaluation results for a single scene file to a common CSV file."""

    timestamp = pd.Timestamp.now()
    exists: bool = os.path.exists(csv_path)
    exp_name = exp_name or Path(checkpoint_path).parent.parent.name

    # save each metric as different row to the csv path
    metric_names = list(metric_dict.keys())
    metric_values = list(metric_dict.values())
    
    metric_values = [float('{:.2f}'.format(value)) for value in metric_values]
    N = len(metric_names)
    df = pd.DataFrame(
        dict(
            timestamp=[timestamp] * N,
            checkpoint_path=[checkpoint_path] * N,
            exp_name=[exp_name] * N,
            dataset=[dataset_name] * N,
            metric_name=metric_names,
            metric_value=metric_values,
            error=[error] * N,
            iters_done=[iters_done] * N,
        ),
        index=list(range(N)),
    )

    # Lock the file to prevent multiple processes from writing to it at the same time.
    # lock = FileLock(f"{csv_path}.lock", timeout=10)
    # with lock:
    df.to_csv(csv_path, mode="a", header=not exists, index=False)

if __name__ == '__main__':
    main()
