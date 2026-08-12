import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (constant_init, normal_init)
from .modules import MixerLayer, FCBlock, BasicBlock
import sys, os
sys.path.append(os.path.join(__file__.replace(os.path.basename(__file__), ''), '..', '..', '..', '..'))

from tokenization.models.vanilla_pose_vqvae import DecodeTokens as VanillaDecodeTokens
from tokenization.models.transformer_pose_vqvae import TransformerDecodeTokens

class Proxy(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.set_gpu = False
    def tokenize(self, x):
        if not self.set_gpu:
            self.tokenizer = self.tokenizer.to(x.device)
            self.set_gpu = True
        return self.tokenizer(x)

class TokenClassfier(nn.Module):
    """ Head of Pose Compositional Tokens.
        paper ref: Zigang Geng et al. "Human Pose as
            Compositional Tokens"

        The pipelines of two stage during training and inference:

        Tokenizer Stage & Train: 
            Joints -> (Img Guide) -> Encoder -> Codebook -> Decoder -> Recovered Joints
            Loss: (Joints, Recovered Joints)
        Tokenizer Stage & Test: 
            Joints -> (Img Guide) -> Encoder -> Codebook -> Decoder -> Recovered Joints

        Classifer Stage & Train: 
            Img -> Classifier -> Predict Class -> Codebook -> Decoder -> Recovered Joints
            Joints -> (Img Guide) -> Encoder -> Codebook -> Groundtruth Class
            Loss: (Predict Class, Groundtruth Class), (Joints, Recovered Joints)
        Classifer Stage & Test: 
            Img -> Classifier -> Predict Class -> Codebook -> Decoder -> Recovered Joints
            
    Args:
        stage_pct (str): Training stage (Tokenizer or Classifier).
        in_channels (int): Feature Dim of the backbone feature.
        image_size (tuple): Input image size.
        num_joints (int): Number of annotated joints in the dataset.
        cls_head (dict): Config for PCT classification head. Default: None.
        tokenizer (dict): Config for PCT tokenizer. Default: None.
        loss_keypoint (dict): Config for loss for training classifier. Default: None.
    """

    def __init__(self, in_channels=2048, token_num=40, token_class_num=2046, token_code_dim=None, \
                 tokenizer_checkpoint_path=None, tokenizer_type='Vanilla', decode_mode='soft', \
                 vqhps_queries=False, vqhps_hard_decode=None, gumbel=False, gumbel_hard=True,
                 vqhps_decode='nograd', st_tau=1.0):
        super().__init__()

        # Inference decoding strategy for turning predicted token logits into a pose:
        #   'soft' -> softmax-weighted codebook mix (differentiable; used in training).
        #   'hard' -> argmax one-hot codebook lookup (non-differentiable; eval-only).
        assert decode_mode in ('soft', 'hard'), decode_mode
        self.decode_mode = decode_mode
        # The two halves of the old single `vqhps_method` switch, now independent so the recipe
        # can be ablated one change at a time (see MODEL.VQHPS_QUERIES / VQHPS_HARD_DECODE):
        #   vqhps_queries     -- (a) per-token query features in, shared LayerNorm+Linear
        #                        classifier out, instead of the pooled-vector MLP-Mixer path.
        #   vqhps_hard_decode -- (b) argmax one-hot forward through the frozen decoder in
        #                        training AND eval, instead of the softmax mixture.
        # They are orthogonal: the decode operates on cls_logits, which both classifier paths
        # produce with shape (B, token_num, token_class_num).
        self.vqhps_queries = vqhps_queries
        self.vqhps_hard_decode = vqhps_queries if vqhps_hard_decode is None else vqhps_hard_decode
        # Backward path for the VQ-HPS decode (the forward is argmax one-hot either way).
        assert vqhps_decode in ('nograd', 'st-argmax', 'gumbel'), vqhps_decode
        self.vqhps_decode = vqhps_decode
        self.st_tau = float(st_tau)

        # MoRo-style Gumbel-softmax training decode (MoRo transformer_module.diff_mesh): during
        # training the decoder input is a Gumbel-softmax sample of the logits instead of the plain
        # softmax; with gumbel_hard=True the sample is one-hot with a straight-through gradient, so
        # the pose loss trains the logits through a decode that matches hard-argmax inference.
        # gumbel_tau is owned by the LightningModule (it knows global_step) via set_gumbel_tau;
        # eval is untouched (decode_mode applies).
        self.gumbel = gumbel
        self.gumbel_hard = gumbel_hard
        self.gumbel_tau = 1.0

        self.conv_num_blocks = 1
        self.dilation = 1
        self.conv_channels = 256
        self.hidden_dim = 64
        self.num_blocks = 4
        self.hidden_inter_dim = 256
        self.token_inter_dim = 64
        self.dropout = 0.0

        self.token_num = token_num # number of token
        self.token_class_num = token_class_num # token class number

        self.token_code_dim = token_code_dim

        if self.vqhps_queries:
            # VQ-HPS-style classifier: forward() receives one feature vector PER token position
            # (B, token_num, in_channels) from the head's learned queries, and a single shared
            # LayerNorm + Linear maps each to codebook logits (mirror of VQ-HPS's ln_f + head).
            # The pooled-vector MLP-Mixer path below is not built at all in this mode.
            self.vqhps_norm = nn.LayerNorm(in_channels)
            self.class_pred_layer = nn.Linear(in_channels, self.token_class_num)
        else:
            #input_size = 12 * 14
            self.mixer_trans = FCBlock(
                in_channels, #self.conv_channels * input_size,
                self.token_num * self.hidden_dim)

            self.mixer_head = nn.ModuleList(
                [MixerLayer(self.hidden_dim, self.hidden_inter_dim,
                    self.token_num, self.token_inter_dim,
                    self.dropout) for _ in range(self.num_blocks)])
            self.mixer_norm_layer = FCBlock(
                self.hidden_dim, self.hidden_dim)

            self.class_pred_layer = nn.Linear(self.hidden_dim, self.token_class_num)

        # Use the pretrained decoder
        tokenizer_proxy = Proxy(eval(f'{tokenizer_type.capitalize()}DecodeTokens')(tokenizer_checkpoint_path))
        self.tokenize = tokenizer_proxy.tokenize
        if self.vqhps_queries:
            # The classifier geometry must match the frozen decoder's checkpoint, otherwise the
            # one-hot @ codebook decode silently mispairs classes and codes.
            dec = tokenizer_proxy.tokenizer
            ckpt_tokens = getattr(dec, 'num_tokens', token_num)
            ckpt_codes = getattr(dec, 'num_code', token_class_num)
            assert (ckpt_tokens, ckpt_codes) == (token_num, token_class_num), (
                f'TOKEN_NUM/TOKEN_CLASS_NUM ({token_num}/{token_class_num}) do not match the '
                f'tokenizer checkpoint ({ckpt_tokens}/{ckpt_codes}): {tokenizer_checkpoint_path}')


    def forward(self, x):
        """Forward function."""
        batch_size = x.shape[0]
        if self.vqhps_queries:
            # x: (B, token_num, in_channels) per-token query features.
            cls_logits = self.class_pred_layer(self.vqhps_norm(x))
        else:
            # x: (B, in_channels) pooled feature.
            # B x 1024 -> B x (token_num x hidden_dim)
            cls_feat = self.mixer_trans(x)
            cls_feat = cls_feat.reshape(batch_size, self.token_num, -1)

            for mixer_layer in self.mixer_head:
                cls_feat = mixer_layer(cls_feat)
            cls_feat = self.mixer_norm_layer(cls_feat)

            # logits: B x token_num x token_class_num
            cls_logits = self.class_pred_layer(cls_feat)

        # cls_logits_softmax ((B * token_number)x token_dim)
        cls_logits_softmax = cls_logits.softmax(-1)

        # Weights fed to the frozen decoder (dequantize_logits = weights @ codebook).
        # 'hard': one-hot(argmax) -> exact codebook lookup; 'soft': softmax mixture.
        if self.vqhps_hard_decode:
            # VQ-HPS decode: the FORWARD pass is always argmax one-hot through the frozen
            # decoder, in training AND eval, so 'soft' eval == 'hard' eval by construction and
            # decode_mode is ignored. VQHPS_DECODE selects only what happens in the BACKWARD:
            #   'nograd'    -- the published recipe: no_grad, so dL_pose/dlogits = 0 exactly and
            #                  cross-entropy is the token path's only training signal.
            #   'st-argmax' -- straight-through: same one-hot forward, but the backward uses the
            #                  Jacobian of softmax(logits/tau), restoring the pose gradient so
            #                  the head can learn which of its mistakes are geometrically cheap.
            #   'gumbel'    -- stochastic variant: the forward index is SAMPLED from the
            #                  Gumbel-perturbed logits instead of taken as the argmax. Only
            #                  during training; eval still decodes the plain argmax.
            # Eval always takes the no_grad path: no gradient is needed and a sampled index
            # would not be the one inference actually uses.
            idx = cls_logits.argmax(-1)
            hard_weights = F.one_hot(idx, self.token_class_num).to(cls_logits_softmax.dtype)
            if not self.training or self.vqhps_decode == 'nograd':
                with torch.no_grad():
                    smpl_thetas6D = self.tokenize(hard_weights)
            elif self.vqhps_decode == 'st-argmax':
                # fp32: the trainer runs fp16 autocast and logits/tau can overflow half.
                soft = (cls_logits.float() / self.st_tau).softmax(-1)
                # Forward value is exactly the one-hot; the gradient flows through `soft`.
                decode_weights = hard_weights.float() - soft.detach() + soft
                smpl_thetas6D = self.tokenize(decode_weights)
            elif self.vqhps_decode == 'gumbel':
                decode_weights = F.gumbel_softmax(
                    cls_logits.float() / self.st_tau, tau=1.0, hard=True, dim=-1)
                smpl_thetas6D = self.tokenize(decode_weights)
            else:
                raise ValueError(f'unknown VQHPS_DECODE: {self.vqhps_decode}')
        elif self.training and self.gumbel:
            # Anneal the SAMPLING distribution, not just the backward surrogate. Passing tau to
            # gumbel_softmax only rescales the soft surrogate used for gradients: the hard sample
            # is argmax(logits + g), and dividing by tau is monotonic, so tau CANNOT change which
            # code is picked (verified: identical noise, tau 1.0 vs 0.01 -> same code 100% of the
            # time). With a near-uniform predictor that means the decoder sees a random code at
            # every position for the whole run, the pose loss trains on garbage, and the logits
            # never sharpen -- which is exactly how the 40k pilot failed.
            # Scaling the logits instead makes the sample ~ Categorical(softmax(logits/tau)), so
            # the schedule genuinely converges the training decode to argmax. This is a deliberate
            # deviation from MoRo, which can pass tau straight through because its logits are
            # already sharp (token embeddings frozen from the VQ-VAE + CE on masked tokens as the
            # primary objective) -- a precondition that does not hold for a stage-2 head here.
            # fp32 is required: the trainer runs fp16 autocast and logits/tau overflows half.
            scaled_logits = cls_logits.float() / self.gumbel_tau
            decode_weights = F.gumbel_softmax(
                scaled_logits, tau=1.0, hard=self.gumbel_hard, dim=-1)
            smpl_thetas6D = self.tokenize(decode_weights)
        elif self.decode_mode == 'hard':
            idx = cls_logits.argmax(-1)
            decode_weights = F.one_hot(idx, self.token_class_num).to(cls_logits_softmax.dtype)
            smpl_thetas6D = self.tokenize(decode_weights)
        else:
            decode_weights = cls_logits_softmax
            smpl_thetas6D = self.tokenize(decode_weights) # B x 21 x 6

        smpl_thetas6D = smpl_thetas6D.reshape(batch_size, -1)
        # Raw (pre-softmax) logits are returned so a downstream CE loss can consume them.
        return smpl_thetas6D, cls_logits_softmax, cls_logits

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_cls_head(self, conv_channels, conv_num_blocks, dilation):
        feature_convs = []
        feature_conv = self._make_layer(
            BasicBlock,
            conv_channels,
            conv_channels,
            conv_num_blocks,
            dilation=dilation)
        feature_convs.append(feature_conv)
        
        return nn.ModuleList(feature_convs)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def set_gumbel_tau(self, tau):
        self.gumbel_tau = float(tau)

    def frozen_tokenizer(self):
        self.tokenize.eval()
        for name, params in self.tokenize.named_parameters():
            params.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
