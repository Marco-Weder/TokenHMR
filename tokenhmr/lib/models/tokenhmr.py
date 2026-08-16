import math
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Tuple
from torch.autograd import Variable as V
from datetime import datetime
import wandb  # <-- ADDED WANDB IMPORT

from yacs.config import CfgNode

import sys, os
from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from ..utils.misc import load_pretrained
from ..utils.token_metrics import compute_token_metrics, _get_gt_encoder
from ..utils.token_targets import decoder_aware_soft_targets, position_sampling_weights
from .backbones import create_backbone
from .heads import build_smpl_head

from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, VerticesLoss, \
    TokenLoss, Keypoint2DLossPCKT, Keypoint3DLossPCKT, ParameterLossPCKT
from .losses import joint_angle_error, angle_valid_thresh, kp2D_err_valid_thresh
from .smpl_wrapper import SMPL

log = get_pylogger(__name__)
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


class TokenHMR(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True, is_train_state = False, is_demo = False):
        """
        Setup TokenHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])
        self.cfg = cfg
        self.is_demo = is_demo

        # Model in evaluation state
        if not is_train_state:
            print(f'Model is in evaluation state')
            self.backbone, self.smpl_head = create_backbone(cfg, load_weights=False), build_smpl_head(cfg)
            self.backbone, self.smpl_head = load_pretrained(cfg, self.backbone, self.smpl_head, is_train_state)

        # Model in training state
        else:
            print(f'Model is in training state')

            # Create backbone feature extractor
            self.backbone = create_backbone(cfg)

            # Create SMPL head
            self.smpl_head = build_smpl_head(cfg)

            # Warm-start from a finished run's weights (fine-tuning). `ckpt_path` was previously
            # honoured only in the eval branch above, so setting it for training silently did
            # nothing. It loads WEIGHTS ONLY, leaving the optimiser and LR schedule fresh --
            # which is the difference from `resume_path`, which restores the whole trainer state
            # and would resume at the checkpoint's global_step (immediately hitting max_steps).
            if cfg.get('ckpt_path', None):
                self.backbone, self.smpl_head = load_pretrained(
                    cfg, self.backbone, self.smpl_head, strict=False)

            # Create discriminator
            if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
                self.discriminator = Discriminator()

            # Define loss functions
            if self.cfg.MODEL.LOOSE_SUP:
                self.keypoint_3d_loss = Keypoint3DLossPCKT(loss_type='l1')
                self.keypoint_2d_loss = Keypoint2DLossPCKT(loss_type='l1')
                self.smpl_parameter_loss = ParameterLossPCKT()
            else:
                self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
                self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
                self.smpl_parameter_loss = ParameterLoss()
            
            self.vertices_loss = VerticesLoss(loss_type='l1')
            self.token_classification_loss = TokenLoss()
            # Camera-only 2D loss (CAM_KEYPOINTS_2D): plain L1, applied to a projection of
            # DETACHED 3D joints so only pred_cam is supervised. Needed by CE-only training,
            # where the normal KEYPOINTS_2D loss (the camera's only gradient source) is zeroed.
            self.cam_keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')

            if self.cfg.MODEL.SMPL_HEAD.TYPE == 'token':
                if self.cfg.MODEL.FROZEN_LEARNED:
                    self.frozen_backbone()

        # Instantiate SMPL model
        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smpl.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False
        if is_train_state:
            self.validation_step_outputs = []
            self.best_validation_loss = cfg.MODEL.get('VAL_LOSS_SAVE_THRESH', 5.0)
            self.best_MPJPE, self.MPJPE = 75.0, []
            self.best_PAMPJPE, self.PAMPJPE = 50.0, []
            self.best_PVE, self.PVE = 87.0, []
            
    def get_parameters(self):
        all_params = list(self.smpl_head.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def frozen_backbone(self):
        self.backbone.eval()
        for name, params in self.backbone.named_parameters():
            params.requires_grad = False

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and discriminator optimizers with linear warmup + cosine decay scheduler.
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        if bool(self.cfg.MODEL.get('VQHPS_SPLIT_LR', self.cfg.MODEL.get('VQHPS_METHOD', False))):
            # VQ-HPS trains its (randomly initialised) mesh-token regressor at 1e-4 while our
            # base recipe runs everything at TRAIN.LR=1e-6 — far too cold for a fresh classifier.
            # Split: pretrained ViT backbone keeps TRAIN.LR, the whole SMPL head (random init)
            # gets VQHPS_HEAD_LR. LambdaLR applies the same warmup+cosine factor to both groups.
            head_lr = float(self.cfg.MODEL.get('VQHPS_HEAD_LR', 1e-4))
            param_groups = [
                {'params': [p for p in self.backbone.parameters() if p.requires_grad], 'lr': self.cfg.TRAIN.LR},
                {'params': [p for p in self.smpl_head.parameters() if p.requires_grad], 'lr': head_lr},
            ]
        else:
            param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        warmup_steps = self.cfg.TRAIN.get('WARMUP_STEPS', 1000)
        total_steps = self.cfg.GENERAL.TOTAL_STEPS
        min_lr = self.cfg.TRAIN.get('MIN_LR', 1e-8)
        peak_lr = self.cfg.TRAIN.LR

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr / peak_lr, cosine_decay)

        # Suppress false "step before optimizer.step" warning from LambdaLR.__init__
        optimizer._step_count = 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        optimizer._step_count = 0
        self.lr_scheduler = scheduler

        # Apply state stashed by on_load_checkpoint (see there). This restores last_epoch, so the
        # run continues at its true step. NOTE lr_lambda closes over `total_steps` read from the
        # CURRENT config, not the checkpoint's -- so resuming with a different GENERAL.TOTAL_STEPS
        # deliberately re-shapes the remaining cosine decay rather than replaying the old one.
        resumed = getattr(self, '_resumed_lr_scheduler_state', None)
        if resumed is not None:
            self.lr_scheduler.load_state_dict(resumed)
            self._resumed_lr_scheduler_state = None

        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                                lr=self.cfg.TRAIN.LR,
                                                weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
            return optimizer, optimizer_disc
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        # Lightning calls this BEFORE configure_optimizers, so self.lr_scheduler does not exist yet
        # (restoring it here raised AttributeError and made `resume_path` unusable). Stash the state
        # and let configure_optimizers apply it once the scheduler has actually been built.
        if 'lr_scheduler' in checkpoint:
            self._resumed_lr_scheduler_state = checkpoint['lr_scheduler']

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        conditioning_feats = self.backbone(x)

        pred_smpl_params, pred_cam, pred_smpl_params_list = self.smpl_head(conditioning_feats)

        # Store useful regression outputs to the output dict
        output = {}
        if self.cfg.MODEL.SMPL_HEAD.TYPE == 'token':
            output['cls_logits_softmax'] = pred_smpl_params_list['cls_logits_softmax']
            output['cls_logits'] = pred_smpl_params_list['cls_logits']
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        smpl_output = self.smpl(**{k: v for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        """

        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        batch_size = pred_smpl_params['body_pose'].shape[0]

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']

        # Camera-only 2D reprojection loss: same projection as forward_step but with DETACHED
        # 3D joints, so the gradient reaches pred_cam only — never the body-pose/token path.
        # Computed before the loose branch below mutates gt_keypoints_2d confidence in place.
        loss_cam_2d = None
        if self.cfg.LOSS_WEIGHTS.get('CAM_KEYPOINTS_2D', 0) > 0:
            pred_kp2d_cam = perspective_projection(
                pred_keypoints_3d.detach(),
                translation=output['pred_cam_t'].reshape(-1, 3),
                focal_length=output['focal_length'].reshape(-1, 2) / self.cfg.MODEL.IMAGE_SIZE)
            loss_cam_2d = self.cam_keypoint_2d_loss(
                pred_kp2d_cam.reshape(batch_size, -1, 2), gt_keypoints_2d.clone())

        if self.cfg.MODEL.LOOSE_SUP and train:
            dataset_names = batch['dataset']
            batch_size = pred_keypoints_2d.shape[0]

            kp2D_err = gt_keypoints_2d[:, :, -1] * torch.nn.functional.mse_loss(
                pred_keypoints_2d, gt_keypoints_2d[:, :, :-1], reduction='none').sum(dim=2)
            valid_mask2D = kp2D_err > kp2D_err_valid_thresh[None].repeat(batch_size,1).to(kp2D_err.device)
            weak_mask = gt_keypoints_2d[:, :, -1] * (~valid_mask2D).float()
            # Compute 3D keypoint loss
            gt_keypoints_2d[:,:,-1] = gt_keypoints_2d[:,:,-1] * valid_mask2D
            loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d, weak_mask, self.cfg.MODEL.LOOSE_WEIGHT)

            valid_3D_mask = torch.Tensor([name in ['H36M-TRAIN-WMASK', 'BEDLAM'] for name in dataset_names]).float().to(gt_keypoints_3d.device)
            gt_keypoints_3d[:, :, -1] = gt_keypoints_3d[:, :, -1] * ((valid_3D_mask.unsqueeze(-1) + gt_keypoints_2d[:,:,-1]) > 0.5)
            loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14)

            # Compute loss on SMPL parameters
            loss_smpl_params = {}
            for k, pred in pred_smpl_params.items():
                gt = gt_smpl_params[k].view(batch_size, -1)
                if is_axis_angle[k].all():
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
                has_gt = has_smpl_params[k]
                if k in ['betas']:
                    valid_mask3D = None
                    weak_mask = None
                    has_gt *= valid_3D_mask
                elif k in ['body_pose', 'global_orient']:
                    angle_error = joint_angle_error(pred, gt) 
                    valid_mask3D = angle_error > angle_valid_thresh[k][None].repeat(batch_size,1).to(angle_error.device)
                    valid_mask3D = (valid_mask3D * has_gt.unsqueeze(1) + valid_3D_mask.unsqueeze(1)).bool()
                    weak_mask = (~valid_mask3D * has_gt.unsqueeze(1)).float()
                    valid_mask3D = valid_mask3D.float()
                loss_smpl_params[k] = self.smpl_parameter_loss(pred, gt, has_gt, valid_mask3D, weak_mask, self.cfg.MODEL.LOOSE_WEIGHT)
        else:
            # Compute 3D keypoint loss
            loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
            loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14)

            # Compute loss on SMPL parameters
            loss_smpl_params = {}
            for k, pred in pred_smpl_params.items():
                gt = gt_smpl_params[k].view(batch_size, -1)
                if is_axis_angle[k].all():
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
                has_gt = has_smpl_params[k]
                loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d +\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d +\
               sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params])
        if loss_cam_2d is not None:
            loss = loss + self.cfg.LOSS_WEIGHTS['CAM_KEYPOINTS_2D'] * loss_cam_2d

        # GT tokens = the GT SMPL body pose encoded through the frozen tokenizer. They are shared
        # by the optional token cross-entropy loss and the token-accuracy metrics, so encode once.
        gt_bp_rotmat, gt_has, gt_tok = None, None, None
        if 'cls_logits_softmax' in output:
            gt_bp_rotmat = gt_smpl_params['body_pose'].view(batch_size, -1)
            if is_axis_angle['body_pose'].all():
                gt_bp_rotmat = aa_to_rotmat(gt_bp_rotmat.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            else:
                gt_bp_rotmat = gt_bp_rotmat.view(batch_size, -1, 3, 3)
            gt_has = has_smpl_params['body_pose'].reshape(batch_size, -1)[:, 0] > 0.5

        # Optional cross-entropy on encoder-assigned token IDs (Part 3 supervision ablation).
        # Enabled by a positive LOSS_WEIGHTS.TOKEN_CE; a real loss term, so errors are not swallowed.
        loss_token_ce = None
        token_target_stats = {}
        if self.cfg.LOSS_WEIGHTS.get('TOKEN_CE', 0) > 0 and 'cls_logits' in output \
                and gt_bp_rotmat is not None and bool(gt_has.any()):
            enc = _get_gt_encoder(self.cfg.MODEL.TOKENIZER_CHECKPOINT_PATH, gt_bp_rotmat.device)
            # 'hard' = CE against the argmax token ID. 'soft' = CE against the soft codebook
            # assignment of the GT latent (see encode_soft) — the hard ID is near-arbitrary
            # among duplicate codes in crowded codebooks, which caps hard-CE at the marginal.
            # 'decoder_soft' = CE against decoder-aware soft targets (Part 4): candidate codes
            # are scored by the MPJPE their decoded pose incurs, so codes that decode to nearly
            # the same pose are no longer punished as equally wrong. Exact, but only over a
            # random subset of token positions per step (see token_targets).
            ce_target_mode = self.cfg.MODEL.get('TOKEN_CE_TARGET', 'hard')
            # VQ-HPS-style CE hygiene (VQHPS_METHOD): token IDs flip under pose noise far smaller
            # than pseudo-GT annotation error, so during training restrict the CE to samples whose
            # SMPL pose is real 3D GT — the same dataset gate as valid_3D_mask above. Validation
            # (COCO-VAL) and the token metrics keep the full gt_has mask.
            ce_has = gt_has
            if train and bool(self.cfg.MODEL.get('VQHPS_CE_GATE', self.cfg.MODEL.get('VQHPS_METHOD', False))):
                reliable = torch.tensor([name in ['H36M-TRAIN-WMASK', 'BEDLAM'] for name in batch['dataset']],
                                        device=gt_has.device, dtype=torch.bool)
                ce_has = gt_has & reliable
            token_target_stats = {}
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                gt_in = gt_bp_rotmat[:, :enc.num_joints].float()
                if ce_target_mode == 'soft':
                    gt_soft = enc.encode_soft(gt_in, tau=float(self.cfg.MODEL.get('TOKEN_CE_SOFT_TAU', 0.01)))
                    gt_tok = gt_soft.argmax(-1)         # == enc.encode(); shared with the token metrics
                else:
                    gt_tok = enc.encode(gt_in)          # (B, token_num)
                if ce_target_mode == 'decoder_soft' and bool(ce_has.any()):
                    # Optional influence-weighted position sampling. The decoder-aware target
                    # can only be afforded at a few of the TOKEN_NUM positions per step, and
                    # those positions are far from equal: swapping one moves the mesh by
                    # 0.3-8 mm (Appendix token map). Sampling in proportion to the
                    # joint-normalised influence spends that budget where a position can
                    # actually change the pose, and makes the loss weight each position by how
                    # much it moves the body. Set TOKEN_CE_POS_UNBIASED to keep the plain mean
                    # over all positions instead (see token_targets for why that is noisier).
                    pos_w = None
                    wpath = self.cfg.MODEL.get('TOKEN_CE_POS_WEIGHTS', None)
                    if wpath:
                        pos_w = position_sampling_weights(
                            wpath, self.cfg.MODEL.SMPL_HEAD.TOKENIZER.TOKEN_NUM, gt_bp_rotmat.device)
                    ce_positions, ce_dense, ce_pos_ips, token_target_stats = decoder_aware_soft_targets(
                        gt_bp_rotmat, ce_has,
                        self.cfg.MODEL.TOKENIZER_CHECKPOINT_PATH,
                        self.cfg.SMPL.MODEL_PATH,
                        tau=float(self.cfg.MODEL.get('TOKEN_CE_DECODER_TAU', 5.0)),
                        num_positions=int(self.cfg.MODEL.get('TOKEN_CE_NUM_POSITIONS', 8)),
                        num_candidates=int(self.cfg.MODEL.get('TOKEN_CE_NUM_CANDIDATES', 16)),
                        pos_weights=pos_w,
                        unbiased=bool(self.cfg.MODEL.get('TOKEN_CE_POS_UNBIASED', False)),
                    )
            if bool(ce_has.any()):
                logits = output['cls_logits']
                B = gt_tok.shape[0]
                if logits.shape[0] != B:            # align IEF iterations (keep the final B rows)
                    logits = logits[-B:]
                if ce_target_mode == 'decoder_soft':
                    # Mean over sampled positions: an unbiased estimate of the mean over all
                    # token positions, so the loss keeps the same scale as hard CE and
                    # LOSS_WEIGHTS.TOKEN_CE carries over unchanged from the Part 3 runs.
                    ce_logits = logits[ce_has][:, ce_positions].float()
                    loss_token_ce = self.token_classification_loss(ce_logits, ce_dense,
                                                                   position_weights=ce_pos_ips)
                else:
                    ce_target = gt_soft[ce_has] if ce_target_mode == 'soft' else gt_tok[ce_has]
                    loss_token_ce = self.token_classification_loss(logits[ce_has].float(), ce_target)
                loss = loss + self.cfg.LOSS_WEIGHTS['TOKEN_CE'] * loss_token_ce

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())
        if loss_token_ce is not None:
            losses['loss_token_ce'] = loss_token_ce.detach()
        losses.update(token_target_stats)
        if loss_cam_2d is not None:
            losses['loss_cam_2d'] = loss_cam_2d.detach()

        for k, v in loss_smpl_params.items():
            losses['loss_' + k] = v.detach()

        # Token-analysis metrics for the token predictor (accuracy / entropy / prob-distance).
        # Non-essential telemetry: never let a metric error kill a (multi-day) training run.
        if 'cls_logits_softmax' in output:
            try:
                token_metrics = compute_token_metrics(
                    output['cls_logits_softmax'], gt_bp_rotmat, gt_has,
                    self.cfg.MODEL.TOKENIZER_CHECKPOINT_PATH, gt_tok=gt_tok,
                )
                losses.update(token_metrics)
            except Exception as e:
                if not getattr(self, '_token_metric_warned', False):
                    log.warning(f'Token metrics skipped (logged once): {e}')
                    self._token_metric_warned = True

        output['losses'] = losses

        return loss

    # CHANGED: Renamed function and updated to WandB native logging
    @pl.utilities.rank_zero.rank_zero_only
    def wandb_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_logger: bool = True) -> None:
        """
        Log results and visualizations directly to WandB
        """
        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        # 1. Collect all scalars into a dictionary for WandB
        log_dict = {}
        if write_to_logger:
            for loss_name, val in losses.items():
                log_dict[f'{mode}/{loss_name}'] = val.detach().item()
            if train and getattr(self, '_gumbel_tau', None) is not None:
                log_dict['train/gumbel_tau'] = self._gumbel_tau

        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        if pred_vertices.dtype != torch.float32:
            pred_vertices = pred_vertices.float()
            pred_cam_t = pred_cam_t.float()
            images = images.float()
            pred_keypoints_2d = pred_keypoints_2d.float()
            gt_keypoints_2d = gt_keypoints_2d.float()
            focal_length = focal_length.float()
            
        # The renderer usually returns a NumPy array or Tensor in CHW (Channels-First) format for TB
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                       pred_cam_t[:num_images].cpu().numpy(),
                                                       images[:num_images].cpu().numpy(),
                                                       pred_keypoints_2d[:num_images].cpu().numpy(),
                                                       gt_keypoints_2d[:num_images].cpu().numpy(),
                                                       focal_length=focal_length[:num_images].cpu().numpy())
        
        # 2. Add the rendered predictions to the WandB log dictionary as an Image
        if write_to_logger:
            # Check shape formatting: wandb expects HWC or a clean CHW NumPy array
            img_to_log = predictions
            if isinstance(predictions, torch.Tensor):
                img_to_log = (predictions.cpu().numpy() * 255).astype(np.uint8)
            # If the renderer returned a flat 3D CHW array, roll it to HWC to be safe with WandB
            if len(img_to_log.shape) == 3 and img_to_log.shape[0] == 3:
                img_to_log = np.transpose(img_to_log, (1, 2, 0))

            log_dict[f'{mode}/predictions'] = wandb.Image(img_to_log, caption=f"Step: {step_count}")
            
            # Send the entire dictionary to WandB in one API call
            self.logger.experiment.log(log_dict, step=step_count)

        return predictions

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        """
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def gumbel_tau_schedule(self) -> float:
        # MoRo's cosine temperature anneal (MoRo transformer_module.diff_mesh): high -> low over
        # TEMP_END_RATIO of TOTAL_STEPS, then held at TEMP_END. Early training decodes a smooth
        # codebook mixture; by the end the Gumbel sample is effectively the argmax the model will
        # face at hard-decode inference.
        temp_start = float(self.cfg.MODEL.get('TOKEN_GUMBEL_TEMP_START', 1.0))
        temp_end = float(self.cfg.MODEL.get('TOKEN_GUMBEL_TEMP_END', 0.01))
        end_ratio = float(self.cfg.MODEL.get('TOKEN_GUMBEL_TEMP_END_RATIO', 1.0))
        ratio = min(self.global_step / (self.cfg.GENERAL.TOTAL_STEPS * end_ratio), 1.0)
        return temp_end + 0.5 * (temp_start - temp_end) * (1 + math.cos(ratio * math.pi))

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']

        # The classifier can't see global_step, so the anneal is pushed in from here each step.
        if self.cfg.MODEL.get('TOKEN_GUMBEL', False):
            tau = self.gumbel_tau_schedule()
            self.smpl_head.decpose.set_gumbel_tau(tau)
            # Stashed for wandb_logging; tau reaches wandb through that manual log call, NOT
            # through self.log(logger=True). Lightning's WandbLogger advances wandb's internal
            # step counter to global_step+1, after which validation's explicit
            # `experiment.log(..., step=global_step)` is silently dropped as non-monotonic
            # ("Tried to log to step N that is less than the current step N+1") -- that is how
            # the 40k pilot lost every val/* metric. Every other self.log here is logger=False
            # for the same reason (train/grad_norm is logger=True but gated off by default).
            self._gumbel_tau = tau
            if self.global_step % 6 == 0:
                self.log('train/gumbel_tau', tau, on_step=True, prog_bar=True, logger=False)
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
            print('dis gan loss:'+'{:.2f}'.format(self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv))

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        self.lr_scheduler.step()

        if self.global_step % 6 == 0:
            self.log('train/lr', self.lr_scheduler.get_last_lr()[0], on_step=True, prog_bar=True, logger=False)

        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            # CHANGED to wandb_logging
            self.wandb_logging(batch, output, self.global_step, train=True)

        if self.global_step % 6 == 0:
            self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to WandB
        """
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.validation_step_outputs.append(loss)

        if self.global_step > 0 and batch_idx % self.cfg.GENERAL.LOG_STEPS == 0:
            self.wandb_logging(batch, output, self.global_step, train=False)

        output['loss'] = loss

        return output