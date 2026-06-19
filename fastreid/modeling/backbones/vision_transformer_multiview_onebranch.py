""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""
"""
load_state_dict会加载除了以下之外的模型的所有参数，包括嵌入层卷积参数、位置嵌入、Transformer编码器的block和FFN、LN参数
view_token
pos_embed 中对应 view_token 的那一行
sie_embed


"""
import logging
import math
import pdb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastreid.layers import DropPath, trunc_normal_, to_2tuple
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#norm1和norm2不能修改，因为官方预训练权重也是这个名字，否则加载时不能匹配
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        # 遍历当前模块（self）以及它包含的所有子模块
        for m in self.modules():

            # 如果当前模块是二维卷积层（nn.Conv2d）
            if isinstance(m, nn.Conv2d):
                # 计算每个输出通道对应的权重元素个数
                # kernel_size[0] * kernel_size[1] = 卷积核的空间尺寸
                # m.out_channels = 输出通道数
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                # 使用Kaiming正态分布初始化卷积核权重
                # 均值=0，方差= sqrt(2 / n)
                # 适用于ReLU等激活函数，缓解梯度消失/爆炸
                m.weight.data.normal_(0, math.sqrt(2. / n))

            # 如果当前模块是批归一化层（nn.BatchNorm2d）
            elif isinstance(m, nn.BatchNorm2d):
                # 将缩放系数γ初始化为1
                m.weight.data.fill_(1)
                # 将偏移系数β初始化为0
                m.bias.data.zero_()

            # 如果当前模块是InstanceNorm2d
            elif isinstance(m, nn.InstanceNorm2d):
                # 将缩放系数γ初始化为1
                m.weight.data.fill_(1)
                # 将偏移系数β初始化为0
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768] Transformer要求输入序列为[B, patch_nums, C]
        return x

#patch_embed是嵌入网络。pose_embed是位置嵌入参数
class VisionTransformer_multiview_onebranch(nn.Module):
    """
    多视图单分支 Vision Transformer（ViT）
    论文参考:
    - Vision Transformer: https://arxiv.org/abs/2010.11929
    - DeiT: https://arxiv.org/abs/2012.12877
    扩展支持多视图 + 相机感知嵌入（SIE）
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.0, inner_sub=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # 记录特征维度，与其他模型接口一致

        # 如果使用混合骨干（CNN + Transformer）
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            # 默认使用重叠patch embedding（可控制patch步长）
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size,
                in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches  # 图像被分割成的patch数量

        # 定义 cls_token（分类token）和 view_token（视图token）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # [1,1,embed_dim]
        self.view_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # [1,1,embed_dim]

        # 位置嵌入：patch_embed + cls_token + view_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))

        # 相机数量（用于SIE相机嵌入）
        self.cam_num = camera
        self.sie_xishu = sie_xishu  # SIE嵌入的权重系数

        # 如果相机数>1，则初始化相机感知嵌入
        if camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)

        # 位置嵌入dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer 编码器层堆叠
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

        # 最终LayerNorm
        self.norm = norm_layer(embed_dim)

        # 初始化可学习参数
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        # 是否在Transformer内部做 cls_token - view_token
        self.inner_sub = inner_sub

    def _init_weights(self, m):
        """
        模型参数初始化
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        告诉优化器这些参数不做权重衰减
        """
        return {'pos_embed', 'cls_token', 'view_token'}

    def forward(self, x, camera_id=None):
        """
        前向传播
        x: [B, C, H, W] 输入图像
        camera_id: [B] 每个样本的相机ID（用于SIE嵌入）
        """
        B = x.shape[0]  # batch size

        # patch embedding：把图像分块并映射到embed_dim
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # 扩展 cls_token 和 view_token 到 batch 维度
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        view_tokens = self.view_token.expand(B, -1, -1)  # [B, 1, embed_dim]

        # 拼接 token：[cls_token, view_token, patch_embeddings]
        x = torch.cat((cls_tokens, view_tokens, x), dim=1)  # [B, num_patches+2, embed_dim]

        # 添加位置嵌入 + 相机感知嵌入（SIE）
        if self.cam_num > 0 and camera_id is not None:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        else:
            x = x + self.pos_embed

        # 位置嵌入dropout
        x = self.pos_drop(x)

        # 经过 Transformer 编码器
        for blk in self.blocks:
            x = blk(x)
            # 如果开启 inner_sub，则在每个block后执行 cls_token -= view_token
            if self.inner_sub:
                x[:, 0] = x[:, 0] - x[:, 1]#分层减法分离！！！！！！！！！！！！！！！！！！！！！！！！

        # 最终LayerNorm
        x = self.norm(x)

        # 返回 cls_token 和 view_token（reshape成 [B, C, 1, 1] 方便与CNN特征图融合）
        return x[:, 0].reshape(x.shape[0], -1, 1, 1), \
               x[:, 1].reshape(x.shape[0], -1, 1, 1)


def resize_pos_embed(posemb, posemb_new, hight, width, cls_token_num):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :cls_token_num], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    logger.info('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                      posemb_new.shape,
                                                                                                      hight,
                                                                                                      width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


# 注册该函数为BACKBONE_REGISTRY的成员，使Detectron2框架能通过配置文件调用此 backbone 构建函数
# 装饰器@BACKBONE_REGISTRY.register()是Detectron2的标准接口，实现"配置驱动模型构建"
@BACKBONE_REGISTRY.register()
def build_multiview_vit_backbone_onebranch(cfg):  # !!!!!!!!!!!!!!!!!!!!!!创建你需要的多视图 ViT 主干网络
    """
    从配置文件创建多视图 Vision Transformer（ViT）主干网络实例
    Returns:
        VisionTransformer_multiview_onebranch: 自定义的单分支多视图ViT模型实例
    """
    # fmt: off  # 关闭格式化，保持配置参数代码的可读性（避免自动换行打乱参数顺序）
    # 从配置文件（cfg）中读取输入图像尺寸（训练时的尺寸）
    input_size = cfg.INPUT.SIZE_TRAIN
    # 读取是否使用预训练权重的开关（True/False）
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    # 读取预训练权重文件的路径
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    # 读取ViT模型的深度规格（如'small'/'base'，对应不同层数）
    depth = cfg.MODEL.BACKBONE.DEPTH
    # 读取SIE（可能是"Scale-Invariant Embedding"等自定义模块）的系数参数
    sie_xishu = cfg.MODEL.BACKBONE.SIE_COE
    # 读取ViT patch embedding层的步长参数（控制图像分块的大小，如2/4）
    stride_size = cfg.MODEL.BACKBONE.STRIDE_SIZE
    # 读取全连接层的dropout概率（防止过拟合）
    drop_ratio = cfg.MODEL.BACKBONE.DROP_RATIO
    # 读取Transformer层的drop path概率（结构化dropout，比普通dropout更适合Transformer）
    drop_path_ratio = cfg.MODEL.BACKBONE.DROP_PATH_RATIO
    # 读取自注意力层的dropout概率（防止注意力权重过拟合）
    attn_drop_rate = cfg.MODEL.BACKBONE.ATT_DROP_RATE
    # 读取自定义参数inner_sub（可能用于Transformer层内部的子模块控制，如分层特征提取）
    inner_sub = cfg.MODEL.BACKBONE.INNER_SUB
    # fmt: on  # 重新开启代码格式化

    # 根据depth规格（'small'/'base'）定义ViT的Transformer层数（num_depth）
    # 'small'对应8层，'base'对应12层，符合ViT的经典架构设计（如ViT-Base为12层）
    num_depth = {
        'small': 8,
        'base': 12,
    }[depth]

    # 根据depth规格定义每个Transformer层的注意力头数（num_heads）
    # 'small'对应8头，'base'对应12头，确保模型宽度与深度匹配（避免参数失衡）
    num_heads = {
        'small': 8,
        'base': 12,
    }[depth]

    # 根据depth规格定义MLP层的扩张比例（mlp_ratio）
    # 'small'为3倍，'base'为4倍，控制MLP层的通道数变化（如输入48维→48*4=192维）
    mlp_ratio = {
        'small': 3.,
        'base': 4.
    }[depth]

    # 根据depth规格定义QKV矩阵是否使用偏置（qkv_bias）
    # 'small'不使用偏置，'base'使用偏置，平衡模型复杂度与训练稳定性
    qkv_bias = {
        'small': False,
        'base': True
    }[depth]

    # 根据depth规格定义QK缩放因子（qk_scale）
    # 'small'使用固定缩放（768^-0.5，768为ViT-Base的特征维度），'base'自动计算（None）
    qk_scale = {
        'small': 768 ** -0.5,
        'base': None,
    }[depth]

    # 实例化自定义的单分支多视图ViT模型
    # 传入所有从配置读取的参数，确保模型结构完全由配置驱动（便于修改和复用）
    model = VisionTransformer_multiview_onebranch(
        img_size=input_size,  # 输入图像尺寸
        sie_xishu=sie_xishu,  # 自定义SIE模块系数
        stride_size=stride_size,  # patch embedding步长
        depth=num_depth,  # Transformer层数
        num_heads=num_heads,  # 注意力头数
        mlp_ratio=mlp_ratio,  # MLP扩张比例
        qkv_bias=qkv_bias,  # QKV偏置开关
        qk_scale=qk_scale,  # QK缩放因子
        drop_path_rate=drop_path_ratio,  # drop path概率
        drop_rate=drop_ratio,  # 全连接层dropout概率
        attn_drop_rate=attn_drop_rate,  # 注意力层dropout概率
        inner_sub=inner_sub,  # 自定义内部子模块参数
    )

    # 如果配置开启了预训练权重加载（pretrain=True）
    if pretrain:
        try:
            # 加载预训练权重文件，先映射到CPU（避免GPU内存不足或设备不匹配）
            # torch.load的map_location=torch.device('cpu')确保在任意设备上都能加载权重
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
            # 日志记录加载路径，便于追踪预训练权重来源
            logger.info(f"Loading pretrained model from {pretrain_path}")

            # 处理不同格式的权重文件：
            # 1. 如果权重字典中包含'model'键（如某些自定义训练框架的输出），提取其值
            if 'model' in state_dict:
                state_dict = state_dict.pop('model')
            # 2. 如果权重字典中包含'state_dict'键（如PyTorch Lightning的输出），提取其值
            if 'state_dict' in state_dict:
                state_dict = state_dict.pop('state_dict')
            """
            权重字典:key为各个层的name，value为参数，格式为张量
            {
                "patch_embed.proj.weight": tensor([[[[...]]]]),  # patch嵌入层卷积权重
                "patch_embed.proj.bias": tensor([...]),          # patch嵌入层偏置
                "pos_embed": tensor([[...]]),                    # 位置嵌入
                "blocks.0.attn.qkv.weight": tensor([...]),       # 第1层Transformer注意力QKV权重
                "head.fc.weight": tensor([...]),                 # 分类头全连接层权重
                "dist.emb": tensor([...])                        # 分布式训练相关参数
            }
            """
            # 遍历权重字典，过滤无效参数并调整不匹配的参数形状
            for k, v in state_dict.items():
                # 跳过分类头（'head'）和分布式训练相关参数（'dist'）
                # 原因：分类头是任务特定的（如ImageNet分类vs行人ReID），需重新训练；dist参数无用
                if 'head' in k or 'dist' in k:#为什么用in而不是head 原因为k通常为head.xxx.xxx
                    continue
                # 处理patch embedding层权重形状不匹配的情况（旧模型可能用非卷积分块）
                # 若预训练权重的'patch_embed.proj.weight'维度<4（如2维），调整为4维（O,I,H,W）
                if 'patch_embed.proj.weight' in k and len(v.shape) < 4:#原始模型用FC，新模型用Conv计算embeding
                    # 获取当前模型patch_embed层的权重形状（输出通道O，输入通道I，卷积核H,W）
                    O, I, H, W = model.patch_embed.proj.weight.shape
                    # 将旧权重reshape为4维，适配当前模型的卷积层
                    v = v.reshape(O, -1, H, W)
                # 处理位置嵌入（pos_embed）形状不匹配的情况（输入图像尺寸与预训练不同）
                #假设预训练模型用224×224图像、16×16 patch，那么图像会分成14×14=196个 patch，加上 1 个 cls token，pos_embed 形状为(1, 197, 768)（1=batch 维度，197=196+1，768 = 特征维度）；
                #当前模型用256×128图像（AG-ReID 常用尺寸）、16×16 patch，图像会分成16×8=128个 patch，加上 1 个 cls token，pos_embed 形状为(1, 129, 768)
                elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
                    # 若预训练权重是蒸馏模型（含'distilled'关键词），需移除蒸馏相关的cls token
                    if 'distilled' in pretrain_path:
                        logger.info("distill need to choose right cls token in the pth.")
                        # 保留第一个cls token，删除第二个蒸馏token（v[:,0:1]为原始cls，v[:,2:]为后续位置嵌入）
                        v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                    # 调用自定义函数resize_pos_embed，将预训练位置嵌入缩放到当前图像尺寸
                    # 参数：预训练pos_embed、当前模型pos_embed、图像分块的高/宽数量（num_y/num_x）、cls token数量（2）

                    #官方的预训练vit默认输入图像为224*224，为了适配reid的256*128要对pos_embed进行resize
                    v = resize_pos_embed(v, model.pos_embed.data, model.patch_embed.num_y, model.patch_embed.num_x, 2)
                # 更新权重字典中的当前参数（确保形状匹配）
                state_dict[k] = v

        # 捕获权重文件不存在的错误，记录日志并重新抛出（便于调试）
        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        # 捕获权重字典键不匹配的错误（如模型结构与权重不对应），记录日志并重新抛出
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e

        # 将处理后的权重加载到模型中，strict=False表示允许部分参数不匹配（如过滤掉的head参数）
        incompatible = model.load_state_dict(state_dict, strict=False)
        # 记录缺失的参数（如模型新增的模块无预训练权重）
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        # 记录意外的参数（如预训练权重中有模型不存在的模块）
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    # 返回构建好的多视图ViT主干网络（供后续ReID模型使用，如添加分类头、属性预测头）
    return model