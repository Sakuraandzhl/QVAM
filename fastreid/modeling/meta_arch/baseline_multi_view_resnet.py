# encoding: utf-8
import torch
import torch.nn.functional as F
from torch import nn
from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from fastreid.layers import trunc_normal_
from .build import META_ARCH_REGISTRY
from fastreid.modeling.QVAM import build_QVAM

class RobustArctanLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.two_over_pi = 2 / 3.141592653589793

    def forward(self, x, y):
        dis = torch.norm(x - y, p=2, dim=0) 
        scaled_dis = self.two_over_pi * torch.atan(dis)
        loss = -torch.log(1 - scaled_dis + self.eps)
        return loss.mean()
    
@META_ARCH_REGISTRY.register()
class Baseline_multiview_resnet(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            qvam,
            view_invariant_head,
            view_head,
            prompts_view_head,
            num_prompts,
            num_id,
            num_view,
            momentum,
            pixel_mean,
            pixel_std,
            feat_dim,
            loss_kwargs=None
    ):
        """
        Args:
            backbone: 骨干网络模块，用于提取图像特征
            heads: 身份分类头，用于常规身份预测
            num_id: 身份类别总数
            num_view: 视角类别总数
            view_identity_heads: 视角-身份联合分类头，输出视角+身份的联合预测
            pixel_mean: 图像归一化的均值
            pixel_std: 图像归一化的标准差
            loss_kwargs: 损失函数的配置参数
        
        Note: This is an **experimental** multi-view re-identification model.
        """
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.num_id = num_id
        self.num_view = num_view
        self.view_head = view_head
        self.qvam = qvam
        self.ArctanLoss = RobustArctanLoss()
        self.view_invariant_head = view_invariant_head
        self.prompts_view_head = prompts_view_head
        self.num_prompts = num_prompts
        self.loss_kwargs = loss_kwargs
        self.register_buffer('memory_bank', torch.zeros(num_id, 2, feat_dim))
        self.momentum = momentum
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        
        qvam = None
        view_invariant_head = None
        view_head = None
        prompts_view_head = None
        if cfg.MODEL.QVAM.IS_RUN:
            qvam = build_QVAM(cfg)
            view_invariant_head = build_heads(cfg)
            if cfg.MODEL.QVAM.VIEW_CLS:
                cfg0 = cfg.clone()
                cfg0.defrost()
                cfg0.MODEL.HEADS.NUM_CLASSES = 2
                view_head = build_heads(cfg0)
            if cfg.MODEL.QVAM.PROMPTS_VIEW_CLS:
                cfg0 = cfg.clone()
                cfg0.defrost()
                cfg0.MODEL.HEADS.NUM_CLASSES = 2
                prompts_view_head = build_heads(cfg0)

        return {
            'backbone': backbone,
            'heads': heads,
            'num_id': cfg.MODEL.HEADS.NUM_CLASSES,
            'num_view': cfg.MODEL.HEADS.VIEW_CLASSES,
            'view_head': view_head,
            'qvam': qvam,
            'prompts_view_head': prompts_view_head,
            'view_invariant_head': view_invariant_head,
            'num_prompts': getattr(cfg.MODEL.QVAM, 'NUM_PROMPTS', 3),
            'feat_dim': cfg.MODEL.BACKBONE.FEAT_DIM,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'momentum': getattr(cfg.MODEL.QVAM, 'MOMENTUM', 0.9),

            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs': {
                'loss_names': cfg.MODEL.LOSSES.NAME,
                'ce': {
                    'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                    'scale': cfg.MODEL.LOSSES.CE.SCALE,
                    'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                    'margin_distinct': getattr(cfg.MODEL.QVAM, 'MARGIN_DISTINCT', 0.),
                    'margin_align': getattr(cfg.MODEL.QVAM, 'MARGIN_ALIGN', 0.),
                    'cvpa_lambda': getattr(cfg.MODEL.QVAM, 'CVPA_LAMBDA', 0.),
                    'view_lambda': getattr(cfg.MODEL.QVAM, 'VIEW_LAMBDA', 0.),
                    'binarize_lambda': getattr(cfg.MODEL.QVAM, 'BINARIZE_LAMBDA', 0.),
                    'diversity_lambda': getattr(cfg.MODEL.QVAM, 'DIVERSITY_LAMBDA', 0.),
                    'prompts_view_cls_lambda': getattr(cfg.MODEL.QVAM, 'PROMPTS_VIEW_CLS_LAMBDA', 0.),
                },
                'tri': {
                    'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                    'scale': cfg.MODEL.LOSSES.TRI.SCALE,
                    'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                    'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                }
            }
        }
        
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, epoch=None):
        images = self.preprocess_image(batched_inputs)
        camids = batched_inputs['camids']
        view_ids = batched_inputs['viewids']
        
        features = self.backbone(images)
        
        B, C, H, W = features.shape
        
        # 使用全局平均池化 (GAP) 生成替代的 [CLS] token, shape: [B, C, 1, 1]
        global_feats = F.adaptive_avg_pool2d(features, 1)
        
        # 将空间维度展平并转置作为 patch tokens, shape: [B, C, H*W] -> [B, H*W, C]
        patch_tokens = features.view(B, C, -1).permute(0, 2, 1)
        
        if self.training:
            label_id = batched_inputs["targets"]
            label_view = torch.tensor([1 if v == 'Aerial' else 0 for v in view_ids], device=label_id.device)

            global_outputs = self.heads(global_feats, label_id)
            
            qvam_out = None
            view_invariant_feats_outputs = None
            view_related_feats_outputs = None
            prompts_view_cls_outputs = None
            if self.qvam:
                qvam_out = self.qvam(
                    global_feats.squeeze(-1).squeeze(-1), 
                    patch_tokens, 
                )
                
                view_invariant_feats = qvam_out['view_invariant_feats_enhance']
                view_invariant_feats_outputs = self.view_invariant_head(view_invariant_feats.unsqueeze(-1).unsqueeze(-1), label_id)

                view_related_feats = qvam_out['view_related_feats']
                if self.view_head:
                    view_related_feats_outputs = self.view_head(view_related_feats.unsqueeze(-1).unsqueeze(-1), label_view)

                view_aware_prompts = None
                if 'prompts' in qvam_out:
                    view_aware_prompts = qvam_out['prompts']
                    if view_aware_prompts is not None:
                        view_aware_prompts = view_aware_prompts.mean(dim=1)
                if self.prompts_view_head and view_aware_prompts is not None:
                    prompts_view_cls_outputs = self.prompts_view_head(view_aware_prompts.unsqueeze(-1).unsqueeze(-1), label_view)

            losses = self.losses(
                label_id, 
                label_view,
                global_outputs,
                view_invariant_feats_outputs,
                view_related_feats_outputs,
                prompts_view_cls_outputs,
                qvam_out
            )
            return losses
            
        else:
            if self.qvam:
                qvam_out = self.qvam(global_feats.squeeze(-1).squeeze(-1), patch_tokens)
                return torch.cat([qvam_out['view_invariant_feats_enhance'], self.heads(global_feats)], dim=1)
            return self.heads(global_feats)

    def preprocess_image(self, batched_inputs):
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs")
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, label_id, label_view, global_out, view_invariant_feats_output, view_related_feats_outputs, prompts_view_cls_outputs, qvam_out):
        loss_dict = {}
        ce_conf = self.loss_kwargs.get('ce')
        tri_conf = self.loss_kwargs.get('tri')
        
        if global_out:
            loss_dict['loss_cls_global'] = cross_entropy_loss(global_out['cls_outputs'], label_id, ce_conf['eps']) * ce_conf['scale']
            loss_dict['loss_tri_global'] = triplet_loss(global_out['features'], label_id, tri_conf['margin'], tri_conf['norm_feat'], tri_conf['hard_mining']) * tri_conf['scale']
            
        if view_invariant_feats_output:
            loss_dict['loss_cls_invariant'] = cross_entropy_loss(view_invariant_feats_output['cls_outputs'], label_id, ce_conf['eps']) * ce_conf['scale']
            loss_dict['loss_tri_invariant'] = triplet_loss(view_invariant_feats_output['features'], label_id, tri_conf['margin'], tri_conf['norm_feat'], tri_conf['hard_mining']) * tri_conf['scale']
            
        if view_related_feats_outputs:
            loss_dict['loss_cls_related'] = cross_entropy_loss(view_related_feats_outputs['cls_outputs'], label_view, ce_conf['eps']) * ce_conf['scale'] * ce_conf['view_lambda']
        if prompts_view_cls_outputs:
            loss_dict['loss_cls_prompts'] = cross_entropy_loss(prompts_view_cls_outputs['cls_outputs'], label_view, ce_conf['eps']) * ce_conf['scale'] * ce_conf['prompts_view_cls_lambda']
        if qvam_out:
            if 'view_related_feats' in qvam_out:
                loss_dict['loss_tri_related'] = triplet_loss(qvam_out['view_related_feats'],label_view,tri_conf['margin'],tri_conf['norm_feat'],tri_conf['hard_mining']) * tri_conf['scale'] * ce_conf['view_lambda']
            if 'view_invariant_feats' in qvam_out:
                loss_dict['loss_tri_inv_pre'] = triplet_loss(qvam_out['view_invariant_feats'],label_id,tri_conf['margin'],tri_conf['norm_feat'],tri_conf['hard_mining']) * tri_conf['scale']
            if 'prompts' in qvam_out and qvam_out['prompts'] is not None:
                loss_dict['loss_tri_prompts'] = triplet_loss(qvam_out['prompts'].mean(dim=1),label_view,tri_conf['margin'],tri_conf['norm_feat'],tri_conf['hard_mining']) * tri_conf['scale'] * ce_conf['prompts_view_cls_lambda']

            unique_ids = torch.unique(label_id)
            zero_tensor = torch.tensor(0.0, device=label_id.device)
            loss_dict['loss_global_mse_batch'] = zero_tensor.clone()
            loss_dict['loss_global_mse_to_memory1'] = zero_tensor.clone()
            loss_dict['loss_global_mse_to_memory2'] = zero_tensor.clone()
            cnt1, cnt2, cnt3 = 0, 0, 0
            for uid in unique_ids:
                mask_id = (label_id == uid)
                id_views = label_view[mask_id]          
                ground_mask = (id_views == 0)
                aerial_mask = (id_views == 1)
                curr_feats = qvam_out['view_invariant_feats'][mask_id]  # [K,
                curr_feats_detach = curr_feats.detach()
                with torch.no_grad():
                    for v in [0, 1]:
                        view_mask = (id_views == v)
                        if view_mask.sum() > 0:
                            mean_feat = curr_feats_detach[view_mask].mean(0)
                            old_feat = self.memory_bank[uid, v]
                            if torch.equal(old_feat, torch.zeros_like(old_feat)):
                                new_feat = mean_feat
                            else:
                                new_feat = self.momentum * old_feat + (1 - self.momentum) * mean_feat
                            self.memory_bank[uid, v] = new_feat
                if ground_mask.sum() > 0:
                    curr_ground = curr_feats[ground_mask].mean(0)
                if aerial_mask.sum() > 0:
                    curr_aerial = curr_feats[aerial_mask].mean(0)
                if ground_mask.sum() > 0 and aerial_mask.sum() > 0:
                    loss_dict['loss_global_mse_batch'] += F.pairwise_distance(curr_ground, curr_aerial)
                    cnt1 += 1
                proto_ground = self.memory_bank[uid, 0].detach()
                proto_aerial = self.memory_bank[uid, 1].detach()
                if (proto_aerial.abs().sum() > 1e-6) and ground_mask.sum() > 0:
                    loss_dict['loss_global_mse_to_memory1'] += F.pairwise_distance(curr_ground, proto_aerial)
                    cnt2 += 1
                if (proto_ground.abs().sum() > 1e-6) and aerial_mask.sum() > 0:
                    loss_dict['loss_global_mse_to_memory2'] += F.pairwise_distance(curr_aerial, proto_ground)
                    cnt3 += 1
            if cnt1 > 0:
                loss_dict['loss_global_mse_batch'] = loss_dict['loss_global_mse_batch'] / cnt1 * ce_conf['cvpa_lambda']
            if cnt2 > 0:
                loss_dict['loss_global_mse_to_memory1'] = loss_dict['loss_global_mse_to_memory1'] / cnt2 * ce_conf['cvpa_lambda']
            if cnt3 > 0:
                loss_dict['loss_global_mse_to_memory2'] = loss_dict['loss_global_mse_to_memory2'] / cnt3 * ce_conf['cvpa_lambda']
            

            epsilon = 1e-7
            mask = qvam_out['mask']
            binarize_loss = - (mask * torch.log(mask + epsilon) + 
                        (1 - mask) * torch.log(1 - mask + epsilon)).mean()
            loss_dict['binarize_loss'] = binarize_loss * ce_conf['binarize_lambda']

            if 'attn_weights' in qvam_out and qvam_out['attn_weights'] is not None:
                loss_dict['loss_div'] = self.compute_diversity_loss(qvam_out['attn_weights']) * ce_conf.get('diversity_lambda', 0.1)
        return loss_dict
    def compute_diversity_loss(self, attn_map):
        gram = torch.bmm(attn_map, attn_map.transpose(1, 2))
        B, K, _ = gram.shape
        mask = 1 - torch.eye(K, device=gram.device).unsqueeze(0)
        return ((gram * mask) ** 2).sum() / (B * K * (K - 1))
