import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from fastreid.layers import trunc_normal_
from .build import QVAM_REGISTRY
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, proj_drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class VAD(nn.Module):
    """
    View Aware Decoder
    """
    def __init__(self, feat_dim, num_prompts=6, num_heads=8, mlp_ratio=4., vad_layer=2):
        super().__init__()
        self.num_prompts = num_prompts
        
        self.view_aware_prompts = nn.Parameter(torch.zeros(num_prompts, feat_dim))
        self.vad_layer = vad_layer

        self.view_aware_decoder = nn.ModuleList()
        for _ in range(vad_layer):
            block = nn.ModuleDict()
            block['norm_sa'] = nn.LayerNorm(feat_dim)
            block['self_attn'] = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
            block['norm_ca'] = nn.LayerNorm(feat_dim)
            block['cross_attn'] = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
            block['norm_prompts'] = nn.LayerNorm(feat_dim)
            block['mlp_prompts'] = Mlp(feat_dim, int(feat_dim * mlp_ratio), feat_dim)
            self.view_aware_decoder.append(block)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.view_aware_prompts, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, tokens):
        """
        Args:
            patch_tokens: [B, N, C] 图像的 Patch 特征
        Returns:
            prompts: [B, P, C] view aware prompts
        """
        B = tokens.shape[0]
        prompts = self.view_aware_prompts.unsqueeze(0).expand(B, -1, -1)
        
        for block in self.view_aware_decoder:
            prompts_resi = prompts
            prompts = block['norm_sa'](prompts) 
            prompts, _ = block['self_attn'](
                query=prompts, 
                key=prompts, 
                value=prompts
            )
            prompts = prompts + prompts_resi 

            prompts_resi = prompts
            prompts = block['norm_ca'](prompts)
            prompts, attn_weights = block['cross_attn'](
                query=prompts, 
                key=tokens, 
                value=tokens,
                )
            prompts = prompts + prompts_resi
            
            prompts_resi = prompts
            prompts = block['norm_prompts'](prompts)
            prompts = block['mlp_prompts'](prompts) + prompts_resi
        
        return prompts, attn_weights

class AFM(nn.Module):
    """
    mask based view disentangling
    """
    def __init__(self, feat_dim, num_heads=8, mlp_ratio=4., vad_is_run=True, afm_layer=2):
        super().__init__()
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.vad_is_run = vad_is_run
        self.afm_layer = afm_layer

        if self.vad_is_run:
            self.adaptive_feature_modulation = nn.ModuleList()

            self.norm_sa = nn.LayerNorm(feat_dim)
            self.self_attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
            
            for _ in range(afm_layer):
                block = nn.ModuleDict()
                block['mlp_cls_token'] = Mlp(feat_dim, int(feat_dim * mlp_ratio), feat_dim)
                block['norm_mask_1'] = nn.LayerNorm(feat_dim)
                block['cross_attn'] = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
                block['norm_mask_2'] = nn.LayerNorm(feat_dim)
                block['mlp_mask'] = Mlp(feat_dim, int(feat_dim * mlp_ratio), feat_dim)
                self.adaptive_feature_modulation.append(block)
        else:
            self.mlp_cls_token = Mlp(feat_dim, int(feat_dim * mlp_ratio), feat_dim)
            self.norm_mask = nn.LayerNorm(feat_dim)
            self.mlp_mask = Mlp(feat_dim, int(feat_dim * mlp_ratio), feat_dim)

        self.norm_inv = nn.LayerNorm(feat_dim)
        self.feature_enhance_mlp = Mlp(feat_dim, int(feat_dim * mlp_ratio), feat_dim)
        self.norm_inv_enhance = nn.LayerNorm(feat_dim)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, cls_token, prompts):
        """
        Args:
            cls_token: [B, C] 全局特征
            prompts: [B, P, C]
        Returns:
            view_invariant_feats: [B, C]
            view_related_feats: [B, C]
            mask: [B, C]
        """
        B = cls_token.shape[0]
        
        mask = self.mask_token.expand(B, -1, -1)

        if self.vad_is_run:
            prompts_resi = prompts
            prompts = self.norm_sa(prompts)
            prompts, _ = self.self_attn(
                query=prompts,
                key=prompts,
                value=prompts
            )
            prompts = prompts + prompts_resi

            for block in self.adaptive_feature_modulation:
                mask = block['mlp_cls_token'](cls_token.unsqueeze(1)) + mask
                mask_resi = mask
                mask = block['norm_mask_1'](mask)
                mask, _ = block['cross_attn'](
                    query=mask, 
                    key=prompts, 
                    value=prompts
                )
                mask = mask + mask_resi

                mask_resi = mask
                mask = block['norm_mask_2'](mask)
                mask= block['mlp_mask'](mask) + mask_resi
        else:
            mask = self.mlp_cls_token(cls_token).unsqueeze(1) + mask
            mask_resi = mask
            mask = self.norm_mask(mask)
            mask = self.mlp_mask(mask) + mask_resi
        refined_mask = mask
        refined_mask = refined_mask.squeeze(1)
        refined_mask = torch.sigmoid(refined_mask)
        
        view_invariant_feats = cls_token * refined_mask
        view_related_feats = cls_token * (1 - refined_mask)
        view_invariant_feats_norm = self.norm_inv(view_invariant_feats)
        view_invariant_feats_enhance = self.feature_enhance_mlp(view_invariant_feats_norm) + view_invariant_feats
        view_invariant_feats_enhance = self.norm_inv_enhance(view_invariant_feats_enhance)
        return view_invariant_feats_enhance, view_invariant_feats, view_related_feats, refined_mask

class QVAM(nn.Module):
    """
    Prompts-driven Adaptive View Disentangling Transformer
    Added: Global View-Invariant Memory Bank
    """
    def __init__(self, feat_dim=768, num_prompts=6, num_heads=8, mlp_ratio=4.,
                 vad_is_run=True, vad_layer=2, afm_layer=2,
                 ):
        super().__init__()
        if vad_is_run:
            self.vad = VAD(feat_dim, num_prompts, num_heads, mlp_ratio, vad_layer=vad_layer)
        self.afm = AFM(feat_dim, num_heads, mlp_ratio, vad_is_run, afm_layer=afm_layer)
        self.vad_is_run = vad_is_run

    def forward(self, cls_token, patch_tokens):
        """
        Args:
        """
        prompts = None
        attn_weights = None
        if self.vad_is_run:
            prompts, attn_weights = self.vad(patch_tokens)
        
        view_invariant_feats_enhance, view_invariant, view_related, mask = self.afm(cls_token, prompts)
        output = {
            'view_invariant_feats_enhance': view_invariant_feats_enhance,
            "view_invariant_feats": view_invariant,
            "view_related_feats": view_related,
            "mask": mask,
            'attn_weights': attn_weights,
        }
        return output

@QVAM_REGISTRY.register()
def build_QVAM(cfg):
    return QVAM(
        feat_dim=cfg.MODEL.BACKBONE.FEAT_DIM,
        num_prompts=cfg.MODEL.QVAM.NUM_PROMPTS,
        num_heads=cfg.MODEL.QVAM.NUM_HEADS,
        mlp_ratio=cfg.MODEL.QVAM.MLP_RATIO,
        vad_is_run=cfg.MODEL.QVAM.VAD_IS_RUN,
        afm_layer=cfg.MODEL.QVAM.AFM_LAYER,
        vad_layer=cfg.MODEL.QVAM.VAD_LAYER,
    )
