import os
import json
import tqdm
import argparse
import warnings
from os.path import join, isdir
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim

import wandb

from dataset.dataset_poseVQ import get_dataloader
import utils.losses as losses
import utils.utils_model as utils_model
from utils.pose_visualize import visualize_from_mesh
from utils.eval_poseVQ import eval_pose_vqvae, reset_err_list, init_best_scores, set_random_seed, get_loggers, gt_from_batch, calculate_pck, PCK_THRESHOLDS
from options.option_posevq import run_grid_search_experiments


def get_model(hparams, add_noise=False):
    if hparams.ARCH.MODEL_NAME in ['vanilla', 'vanilla-v1']:
        from models.vanilla_pose_vqvae import VanillaTokenizer
        net = VanillaTokenizer(hparams.ARCH, add_noise=add_noise)
    elif hparams.ARCH.MODEL_NAME == "transformer":
        from models.transformer_pose_vqvae import TransformerTokenizer
        net = TransformerTokenizer(hparams.ARCH, add_noise=add_noise)
    elif hparams.ARCH.MODEL_NAME == "gnn":
        from models.gnn_pose_vqvae import GNNTokenizer
        net = GNNTokenizer(hparams.ARCH, add_noise=add_noise)
    else:
        raise NotImplementedError(f'{hparams.ARCH.MODEL_NAME} not implemented yet')
    return net


def build_scheduler(optimizer, hparams):
    """LinearLR warmup → main scheduler (multistep | cosine), composed via SequentialLR.

    The main scheduler's step counter is offset by `warmup_iter` (SequentialLR semantics),
    so multistep milestones must be expressed relative to the end of warmup.
    """
    warmup_iter = max(1, int(hparams.OPT.WARM_UP_ITER))
    sched_kind = str(getattr(hparams.OPT, 'SCHEDULER', 'multistep')).lower()

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_iter,
    )

    if sched_kind == 'cosine':
        cosine_iter = max(1, int(hparams.OPT.TOTAL_ITER) - warmup_iter)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_iter,
            eta_min=float(getattr(hparams.OPT, 'MIN_LR', 1e-6)),
        )
    elif sched_kind == 'multistep':
        raw = str(hparams.OPT.LR_SCHEDULER).strip()
        for sep in (',', '_', ':', ';', ' '):
            if sep in raw:
                parts = raw.split(sep)
                break
        else:
            parts = [raw]
        milestones_abs = [int(m) for m in parts if m]
        milestones_rel = [max(1, m - warmup_iter) for m in milestones_abs]
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones_rel,
            gamma=float(hparams.OPT.GAMMA),
        )
    else:
        raise ValueError(f"Unknown OPT.SCHEDULER '{sched_kind}'. Use 'cosine' or 'multistep'.")

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_iter],
    )


def main(hparams):
    torch.manual_seed(hparams.EXP.SEED)

    hparams.EXP.OUT_DIR = os.path.join(hparams.EXP.OUT_DIR, f'{hparams.EXP.NAME}')
    save_dir = join(hparams.EXP.OUT_DIR, 'train_render')
    os.makedirs(hparams.EXP.OUT_DIR, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    wandb.init(
        project="TokenHMR-Transformer",
        name=hparams.EXP.NAME,
        config=hparams,
    )
    if wandb.run is not None:
        wandb.define_metric("tr/iter")
        wandb.define_metric("tr/*", step_metric="tr/iter")

    logger = utils_model.get_logger(hparams.EXP.OUT_DIR)
    logger.info(f'Training on {hparams.DATA.DATASET}, with {hparams.ARCH.NB_JOINTS} joints')

    if hparams.EXP.EVAL_ONLY:
        eval_loader = get_dataloader(hparams, split=hparams.EXP.EVAL_DS, shuffle=False)
    else:
        train_loader_iter = get_dataloader(hparams, split='train')
        val_loader = get_dataloader(hparams, split='val', shuffle=False)

    if hparams.EXP.EVAL_ONLY:
        set_random_seed(0)
        best_scores = init_best_scores()
        writer = get_loggers(hparams)
        logger.info('EVAL-ONLY: loading checkpoint from {}'.format(hparams.EXP.RESUME_PTH))
        ckpt_file = f'{hparams.EXP.RESUME_PTH}/best_net.pth' if isdir(hparams.EXP.RESUME_PTH) else hparams.EXP.RESUME_PTH
        ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        pretrained_hparams = ckpt['hparams']
        net = get_model(pretrained_hparams)
        net.load_state_dict(ckpt['net'], strict=True)
        net.cuda()
        eval_pose_vqvae(hparams, eval_loader, net, logger, writer, 0, hparams.EXP.OUT_DIR, hparams.EXP.VAL_DISP_ITER, best_scores)
        exit()

    start_iter = 1
    warm_restart = bool(getattr(hparams.EXP, 'WARM_RESTART', False))

    if warm_restart:
        # Load model weights only; build a fresh optimizer + scheduler from the YAML.
        # Used to "warm restart" a converged run at a new (higher) LR for a fresh cosine cycle.
        print(f'WARM RESTART: loading model weights from {hparams.EXP.RESUME_PTH}')
        ckpt = torch.load(hparams.EXP.RESUME_PTH, map_location='cpu', weights_only=False)
        pretrained_hparams = ckpt['hparams']
        hparams.ARCH = pretrained_hparams.ARCH  # architecture must match the saved weights
        writer = get_loggers(hparams)
        net = get_model(hparams, hparams.DATA.ADD_NOISE)
        net.load_state_dict(ckpt['net'], strict=True)
        print(f'  Fresh training cycle: start_iter=1, LR from YAML = {hparams.OPT.LR}')
    elif hparams.EXP.RESUME_TRAINING:
        print(f'RESUME TRAINING: loading checkpoint from {hparams.EXP.RESUME_PTH}. Overiding architecture...')
        ckpt = torch.load(hparams.EXP.RESUME_PTH, map_location='cpu', weights_only=False)
        pretrained_hparams = ckpt['hparams']
        hparams.ARCH = pretrained_hparams.ARCH
        writer = get_loggers(hparams)
        net = get_model(pretrained_hparams, hparams.DATA.ADD_NOISE)
        net.load_state_dict(ckpt['net'], strict=True)
        # eval_pose_vqvae writes the iter under key 'iteration' in latest_checkpoint.pth.
        start_iter = ckpt.get('iteration', ckpt.get('nb_iter', 0)) + 1
        print(f'  Resuming from iteration {start_iter}')
    else:
        print('train params:', hparams.ARCH)
        writer = get_loggers(hparams)
        net = get_model(hparams, hparams.DATA.ADD_NOISE)

    net.train()
    net.cuda()

    optimizer = optim.AdamW(net.parameters(), lr=hparams.OPT.LR, betas=(0.9, 0.99), weight_decay=hparams.OPT.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, hparams)

    # Warm-restart deliberately skips this block — we want a fresh optimizer + scheduler.
    if hparams.EXP.RESUME_TRAINING and not warm_restart:
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        elif start_iter > 1:
            for _ in range(start_iter - 1):
                scheduler.step()

    Loss = losses.PoseReConsLoss(hparams.LOSS, hparams.ARCH.NB_JOINTS, hparams.ARCH.ROT_TYPE, hparams.ARCH.SMPL_TYPE)
    best_scores = init_best_scores()

    warmup_iter = int(hparams.OPT.WARM_UP_ITER)
    err_list = reset_err_list('tr')

    if hparams.ARCH.MODEL_NAME in ('vanilla', 'vanilla-v1'):
        from models.vanilla_pose_vqvae import body_model
    else:
        from models.transformer_pose_vqvae import body_model

    ##### ---- Unified training loop (warmup + main, both logged to wandb) ---- #####
    # Codebook-collapse fix: during warmup we call net(gt_pose) WITHOUT global_step,
    # so the noise-injection branch in TransformerTokenizer.forward stays disabled
    # and the EMA codebook initialises + stabilises on clean encoder outputs.
    for nb_iter in tqdm.tqdm(range(start_iter, hparams.OPT.TOTAL_ITER + 1)):
        is_warmup = nb_iter <= warmup_iter

        batch = next(train_loader_iter)
        gt_pose, gt_mesh, gt_jnts = gt_from_batch(batch, body_model)

        if is_warmup:
            output, loss_commit, perplexity = net(gt_pose)            # no global_step → no noise
        else:
            output, loss_commit, perplexity = net(gt_pose, nb_iter)   # noise enabled
        loss_pose = Loss.forward_pose(gt_pose, output)
        loss_mesh = Loss.forward_mesh(gt_mesh, output)
        loss_jnts = Loss.forward_joints(gt_jnts, output)

        loss = hparams.LOSS.POSE_LOSS_WT * loss_pose + \
               hparams.LOSS.MESH_LOSS_WT * loss_mesh + \
               hparams.LOSS.JNT_LOSS_WT * loss_jnts + \
               hparams.LOSS.COMMIT_LOSS_WT * loss_commit
        loss *= hparams.LOSS.LOSS_WT

        optimizer.zero_grad()
        loss.backward()
        # Optional gradient clipping (OPT.GRAD_CLIP > 0). Off by default so the transformer/CNN
        # runs are byte-for-byte unchanged; the GNN enables it to absorb the LR-driven loss spikes
        # that otherwise destabilize training as the warmup ramps the LR up.
        grad_clip = float(getattr(hparams.OPT, 'GRAD_CLIP', 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        err_list['tr/curr_pose_recons'] += hparams.LOSS.POSE_LOSS_WT * loss_pose.item()
        err_list['tr/curr_mesh_recons'] += hparams.LOSS.MESH_LOSS_WT * loss_mesh.item()
        err_list['tr/curr_jnt_recons'] += hparams.LOSS.JNT_LOSS_WT * loss_jnts.item()
        err_list['tr/curr_perplexity'] += perplexity.item()
        err_list['tr/curr_commit'] += hparams.LOSS.COMMIT_LOSS_WT * loss_commit.item()
        err_list['tr/curr_loss'] += hparams.LOSS.LOSS_WT * loss.item()

        # Token-analysis metrics: codebook usage (entropy / KL-to-uniform / utilization)
        # stashed by the quantizer, plus reconstruction accuracy (PCK).
        cb_stats = getattr(net.quantizer, 'codebook_stats', {})
        err_list['tr/curr_entropy'] += cb_stats.get('entropy', 0.)
        err_list['tr/curr_norm_entropy'] += cb_stats.get('norm_entropy', 0.)
        err_list['tr/curr_kl_uniform'] += cb_stats.get('kl_to_uniform', 0.)
        err_list['tr/curr_codebook_util'] += cb_stats.get('codebook_util', 0.)
        with torch.no_grad():
            for key, thresh in PCK_THRESHOLDS.items():
                err_list[f'tr/curr_{key}'] += calculate_pck(gt_jnts, output['pred_body_joints'], thresh).item()

        if nb_iter % hparams.EXP.PRINT_ITER == 0:
            for key in err_list:
                err_list[key] /= hparams.EXP.PRINT_ITER

            log_dict = err_list.copy()
            log_dict['tr/iter'] = nb_iter
            log_dict['tr/lr'] = scheduler.get_last_lr()[0]
            log_dict['tr/phase'] = 0 if is_warmup else 1
            wandb.log(log_dict, step=nb_iter)

            phase = 'Warmup' if is_warmup else 'Train.'
            print_str = f'{phase} Iter {nb_iter}: lr: {scheduler.get_last_lr()[0]:.6f}'
            for key, value in err_list.items():
                print_str += f'\t{key[7:]}: {value:.5f}'
            logger.info(print_str)

            err_list = reset_err_list('tr')

        if nb_iter % hparams.EXP.TR_DISP_ITER == 0:
            visualize_from_mesh(hparams.ARCH.SMPL_TYPE, batch, output, nb_iter, save_dir)

        if nb_iter % hparams.EXP.EVAL_ITER == 0:
            best_scores = eval_pose_vqvae(
                hparams,
                val_loader,
                net,
                logger,
                writer,
                nb_iter,
                hparams.EXP.OUT_DIR,
                hparams.EXP.VAL_DISP_ITER,
                best_scores,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            # eval_pose_vqvae sets net.eval() but does not restore train mode.
            # Without this, QuantizeEMAReset stops EMA codebook updates after the first eval
            # (codebook freezes at iter EVAL_ITER → perplexity stays low).
            net.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--cfg_id', type=int, default=0)
    parser.add_argument('--cluster', default=False, action='store_true')
    parser.add_argument('--bid', type=int, default=30)
    parser.add_argument('--memory', type=int, default=64000)
    parser.add_argument('--gpu_min_mem', type=int, default=20000)
    parser.add_argument('--exclude_nodes', type=str, default='')
    parser.add_argument('--num_cpus', type=int, default=8)
    parser.add_argument('--resume_training', default=False, action='store_true')
    parser.add_argument('--resume_pth', type=str, default='')
    parser.add_argument('--warm_restart', default=False, action='store_true',
                        help='Load only model weights from --resume_pth; build a fresh optimizer + scheduler '
                             'from the YAML and start a new training cycle from iter 1. Use to push a converged '
                             'run further at a new (higher) LR without inheriting the old optimizer state.')

    args = parser.parse_args()
    print(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        bid=args.bid,
        use_cluster=args.cluster,
        memory=args.memory,
        exclude_nodes=args.exclude_nodes,
        script='train_poseVQ.py',
        gpu_min_mem=args.gpu_min_mem,
    )

    if args.warm_restart:
        if not args.resume_pth:
            raise ValueError('--warm_restart requires --resume_pth pointing to a checkpoint to load weights from.')
        hparams.EXP.WARM_RESTART = True
        hparams.EXP.RESUME_PTH = args.resume_pth
        # Mutually exclusive with --resume_training; warm_restart wins.
        hparams.EXP.RESUME_TRAINING = False
    elif args.resume_training:
        hparams.EXP.RESUME_TRAINING = True
        hparams.EXP.RESUME_PTH = args.resume_pth

    main(hparams)
