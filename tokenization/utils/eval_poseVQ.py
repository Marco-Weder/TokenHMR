import os
import random
from os.path import join
import tqdm
import numpy as np
import torch
import pickle as pkl
import wandb

from utils.pose_visualize import visualize_from_mesh
from utils.rotation_conversions import axis_angle_to_matrix
from utils.utils_model import codebook_usage_stats
from torch.utils.tensorboard import SummaryWriter

# Per-joint distance thresholds (metres) for PCK reconstruction accuracy.
PCK_THRESHOLDS = {'pck_50mm': 0.05, 'pck_25mm': 0.025}


def gt_from_batch(batch, body_model):
    """Returns (gt_pose, gt_mesh, gt_jnts) on CUDA, fp32.

    Two paths depending on whether the dataset cached SMPL outputs:
    - cached: the batch already carries `gt_pose_body`, `body_vertices`, `body_joints`.
    - on-the-fly: the batch carries only `pose_body_aa`; SMPL is run on `body_model`
      (an SMPLHLayer / SMPLXLayer on cuda) once per call. Numerically matches the
      cached path to within ~1e-7 (sub-ULP for fp32 MSE losses).
    """
    if 'body_vertices' in batch:
        return (
            batch['gt_pose_body'].cuda().float(),
            batch['body_vertices'].cuda().float(),
            batch['body_joints'].cuda().float(),
        )
    pose_aa = batch['pose_body_aa'].cuda(non_blocking=True).float()
    gt_pose = axis_angle_to_matrix(pose_aa.view(-1, 21, 3))
    out = body_model(body_pose=gt_pose)
    return gt_pose, out.vertices, out.joints

def reset_err_list(type='tr'):
    err_list = {f'{type}/curr_pose_recons': 0.,
                f'{type}/curr_mesh_recons': 0.,
                f'{type}/curr_jnt_recons': 0.,
                f'{type}/curr_perplexity': 0.,
                f'{type}/curr_commit': 0.,
                # Codebook-usage token analysis.
                f'{type}/curr_entropy': 0.,
                f'{type}/curr_norm_entropy': 0.,
                f'{type}/curr_kl_uniform': 0.,
                f'{type}/curr_codebook_util': 0.,
                # Reconstruction accuracy (PCK).
                f'{type}/curr_pck_50mm': 0.,
                f'{type}/curr_pck_25mm': 0.}
    if type == 'tr':
        err_list.update({
            f'{type}/curr_loss': 0.
        })
    return err_list

def init_best_scores():
    best_scores = {
        f'val/best_iter': 0,
        f'val/best_val_score': 1e8,
        f'val/best_mesh_recons': 1e8,
        f'val/best_jnt_recons': 1e8,
    }
    return best_scores

def set_random_seed(random_seed=0):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_loggers(hparams):
    writer = None
    if hparams.EXP.LOG_TB:
        writer = SummaryWriter(hparams.EXP.OUT_DIR)
    return writer

def calculate_pose_reconstruction_error(gt_pose, pred_pose):
    return torch.sqrt(torch.pow(gt_pose-pred_pose, 2).sum(-1)).mean()

def calculate_mesh_reconstruction_error(gt_mesh, pred_mesh):
    return torch.sqrt(torch.pow(gt_mesh-pred_mesh, 2).sum(-1)).mean()

def calculate_jnts_reconstruction_error(gt_jnts, pred_jnts):
    valid_joints = [*range(1,22)] # only body joints
    return torch.sqrt(torch.pow(gt_jnts[:,valid_joints]-pred_jnts[:,valid_joints], 2).sum(-1)).mean()

def calculate_pck(gt_jnts, pred_jnts, thresh):
    """Fraction of body joints reconstructed within `thresh` metres (PCK-style accuracy)."""
    valid_joints = [*range(1,22)] # only body joints
    dist = torch.sqrt(torch.pow(gt_jnts[:,valid_joints]-pred_jnts[:,valid_joints], 2).sum(-1))
    return (dist < thresh).float().mean()

def save_results_func(batch, output, results, gt_jnts):
    valid_joints = [*range(1,22)]
    save_aa_gt = batch['pose_body_aa'].numpy()
    save_jnts_gt = gt_jnts[:, valid_joints].detach().cpu().numpy()
    save_aa_pred = output['pred_pose_body_aa'].detach().cpu().numpy()
    save_jnts_pred = output['pred_body_joints'][:,valid_joints].detach().cpu().numpy()
    results['gt_jnts'].extend(save_jnts_gt)
    results['gt_aa'].extend(save_aa_gt)
    results['pred_jnts'].extend(save_jnts_pred)
    results['pred_aa'].extend(save_aa_pred)
    return results


def eval_pose_vqvae(hparams, val_loader, net, logger, writer, nb_iter, out_dir, val_disp_iter, best_scores, optimizer=None, scheduler=None):

    net.eval()
    save_dir = join(out_dir, 'val_render')
    os.makedirs(save_dir, exist_ok=True)

    save_results = True
    results = {'gt_jnts': [], 'gt_aa': [], 'pred_jnts': [], 'pred_aa': []}

    err_list = reset_err_list('val')
    dataset_name = ''
    # Accumulate code-usage over the whole val set so utilization / entropy / KL are
    # computed from the full histogram (per-batch counts under-cover a 2048-code book).
    total_counts = None

    if hparams.ARCH.MODEL_NAME in ('vanilla', 'vanilla-v1'):
        from models.vanilla_pose_vqvae import body_model
    else:
        from models.transformer_pose_vqvae import body_model

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(val_loader)):
            gt_pose, gt_mesh, gt_jnts = gt_from_batch(batch, body_model)
            dataset_name = batch['dataset_name'][0]

            output, loss_commit, perplexity = net(gt_pose)
            pose_error = calculate_pose_reconstruction_error(gt_pose, output['pred_pose_body_rotmat'])
            mesh_error = calculate_mesh_reconstruction_error(gt_mesh, output['pred_body_vertices'])
            jnt_error = calculate_jnts_reconstruction_error(gt_jnts, output['pred_body_joints'])

            if save_results:
                results = save_results_func(batch, output, results, gt_jnts)

            err_list['val/curr_pose_recons'] += pose_error.item()
            err_list['val/curr_mesh_recons'] += mesh_error.item()
            err_list['val/curr_jnt_recons'] += jnt_error.item()
            err_list['val/curr_perplexity'] += perplexity.item()
            err_list['val/curr_commit'] += loss_commit.item()
            for key, thresh in PCK_THRESHOLDS.items():
                err_list[f'val/curr_{key}'] += calculate_pck(gt_jnts, output['pred_body_joints'], thresh).item()

            counts = getattr(net.quantizer, 'last_code_counts', None)
            if counts is not None:
                total_counts = counts.clone() if total_counts is None else total_counts + counts

            if batch_idx % val_disp_iter == 0:
                visualize_from_mesh(hparams.ARCH.SMPL_TYPE, batch, output, f'eval_{nb_iter}_{batch_idx}', save_dir)

    if save_results:
        with open(f'results{dataset_name}.pkl', 'wb') as handle:
            pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    # Number of processed batches (avoid division by zero and off-by-one)
    num_batches = (batch_idx + 1) if 'batch_idx' in locals() else 0
    if num_batches == 0:
        logger.warning('Validation loader produced 0 batches. Skipping eval logging.')
        return best_scores
    for key, value in err_list.items():
        err_list[key] /= num_batches

    err_list['val/curr_jnt_recons'] *= 1000
    err_list['val/curr_mesh_recons'] *= 1000

    # Codebook stats from the full-val histogram (overwrite the per-batch-averaged zeros).
    if total_counts is not None:
        cb_stats = codebook_usage_stats(total_counts, net.quantizer.nb_code)
        err_list['val/curr_entropy']       = cb_stats['entropy']
        err_list['val/curr_norm_entropy']  = cb_stats['norm_entropy']
        err_list['val/curr_kl_uniform']    = cb_stats['kl_to_uniform']
        err_list['val/curr_codebook_util'] = cb_stats['codebook_util']
    
    curr_score = err_list['val/curr_jnt_recons'] + err_list['val/curr_mesh_recons']
    if optimizer is not None and scheduler is not None:
        latest_dict = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': nb_iter,
            'hparams': hparams,
        }
        torch.save(latest_dict, join(out_dir, 'latest_checkpoint.pth'))

    if curr_score < best_scores['val/best_val_score']:
        best_scores['val/best_val_score'] = curr_score
        save_dict = {
            'net': net.state_dict(),
            'hparams': hparams,
        }
        best_scores['val/best_iter'] = nb_iter
        # Note: full training state (optimizer, scheduler, iteration) saved separately during training loop
        best_scores['val/best_jnt_recons']  = err_list['val/curr_jnt_recons']
        best_scores['val/best_mesh_recons'] = err_list['val/curr_mesh_recons']
        torch.save(save_dict, join(out_dir, 'best_net.pth'))
        logger.info(f"Eval. Iter {nb_iter}: !!---> BEST MODEL FOUND <---!! Validation Score - {best_scores['val/best_val_score']:.2f}")
    
    print_str = f'Eval. Iter: {nb_iter} | curr_score: {curr_score:.2f}'
    for key, value in err_list.items():
        print_str += f'\t {key[9:]}: {value:.5f}'
    for key, value in best_scores.items():
        print_str += f'\t {key[4:]}: {value:.5f}'
    logger.info(print_str)
    
    if wandb.run is not None:
        # Combine both dictionaries into one log call
        eval_logs = {**err_list, **best_scores}
        wandb.log(eval_logs, step=nb_iter)

    net.eval()

    return best_scores
    