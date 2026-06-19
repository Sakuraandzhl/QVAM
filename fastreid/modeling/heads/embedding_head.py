# 导入PyTorch核心库与FastReID自定义模块
import torch  # PyTorch基础库（张量操作、神经网络等）
import torch.nn.functional as F  # PyTorch函数库（激活函数、损失计算等）
from torch import nn  # PyTorch神经网络模块（定义层、模型等）

from fastreid.config import configurable  # FastReID配置注解（用于从配置文件读取参数）
from fastreid.layers import *  # FastReID自定义层（如归一化、池化等）
from fastreid.layers import pooling, any_softmax  # 特定层：池化操作、各类softmax（如ArcSoftmax）
from fastreid.layers.weight_init import weights_init_kaiming  # 权重初始化方法（ kaiming初始化）
from .build import REID_HEADS_REGISTRY  # FastReID注册器（将当前Head类注册到模型组件库）

"""
配置参数	                         defaults.py 默认值	你的 YAML 是否覆盖	最终生效值
MODEL.HEADS.EMBEDDING_DIM	           0	              否	           0
MODEL.HEADS.NUM_CLASSES	               0	              否	           0（需注意！）
MODEL.HEADS.MARGIN	                   0.0	              否	           0.0
MODEL.HEADS.SCALE	                   1	              否 	           1
MODEL.HEADS.POOL_LAYER	          "GlobalAvgPool"	  是（改为 "Identity"）	"Identity"
MODEL.HEADS.WITH_BNNECK	               False	        是（改为 True）	   True


with_bnneck:带BN的瓶颈层，指在经过分类器前进行BN防止过拟合、增强特征判别性，解决内部协变量偏移
embedding_dim:控制最终输出的特征向量维度。如果设为 0，表示不额外增加卷积降维层，直接使用主干网络输出的特征维度
identity:恒等池化，即不做 任何池化
margin:ArcFace、tripletloss等损失的边界，过大难以训练过小效果不好
scale:缩放系数，如arcface的s


logits 就是分类模型最后一层的原始输出，没有经过 softmax 的值

模块	作用	是否属于 bottleneck？
Pooling（聚合）	从特征图中聚合出全局特征（Avg/Max/GeM/Identity）	❌ 不属于，是独立步骤
1×1 Conv	    通道降维（如果 embedding_dim > 0）	            ✅ 属于瓶颈层
BN 归一化	    特征归一化（如果 with_bnneck = True）	        ✅ 属于瓶颈层
[B, C, 1, 1]→[B, C] 挤压	去除空间维度冗余	                    ❌ 不属于，是后续 flatten 步骤


分类头:Linear
计算Tripletloss的特征:bottleneck前(未经过BN的特征)

Backbone（ViT）输出：cls_token → [B, C, 1, 1]
Pooling 层：Identity → [B, C, 1, 1]（没变）
瓶颈层前特征：pool_feat → [B, C, 1, 1]
瓶颈层处理：BN → [B, C, 1, 1]（数值归一化）
瓶颈层后特征：neck_feat → [B, C, 1, 1] → 挤压成 [B, C]
分类器输入：neck_feat（瓶颈层后特征）→ 线性层（无归一化）
三元组损失输入：pool_feat[..., 0, 0]（瓶颈层前特征）
"""
# ---------------------------
# 1. 注册Head类：让FastReID框架能识别并调用该Head
# ---------------------------
@REID_HEADS_REGISTRY.register()  # 装饰器：将EmbeddingHead注册为可用的ReID头网络
class EmbeddingHead(nn.Module):  # 继承nn.Module（PyTorch所有模型的基类）
    """
    EmbeddingHead的核心作用：
    1. 特征聚合（如池化操作，将 backbone 输出的特征图转为向量）
    2. 可选的特征处理（如BN归一化、卷积降维，对应配置WITH_BNNECK、EMBEDDING_DIM）
    3. 训练时计算分类损失（结合margin/scale的softmax，对应MARGIN、SCALE配置）
    适用场景：ReID（行人/货物检索）、图像检索、人脸识别等需要"特征嵌入"的任务
    """

    # ---------------------------
    # 2. 配置参数注入：从YAML/defaults.py读取参数并初始化
    # ---------------------------
    @configurable  # 装饰器：标记该方法从配置文件获取参数（对应from_config方法）

    def __init__(
            self,
            *,  # 强制关键字参数（避免传参顺序错误）
            feat_dim,          # 主干网络输出特征维度（如ViT的768，来自MODEL.BACKBONE.FEAT_DIM）
            embedding_dim,     # 嵌入特征维度（0表示不降维，来自MODEL.HEADS.EMBEDDING_DIM，默认0）
            num_classes,       # 数据集类别数（需在YAML中修改，默认0会报错，来自MODEL.HEADS.NUM_CLASSES）
            neck_feat,         # 损失计算用的特征来源（"before"/"after" BN，来自MODEL.HEADS.NECK_FEAT）
            pool_type,         # 池化方式（如Identity/GlobalAvgPool，来自MODEL.HEADS.POOL_LAYER）
            cls_type,          # 分类层类型（如Linear/ArcSoftmax，来自MODEL.HEADS.CLS_LAYER）
            scale,             # 缩放系数（如ArcFace的s，来自MODEL.HEADS.SCALE，默认1）
            margin,            # 类别间隔（如ArcFace的m，来自MODEL.HEADS.MARGIN，默认0.0）
            with_bnneck,       # 是否用BN瓶颈层（True/False，来自MODEL.HEADS.WITH_BNNECK，默认False）
            norm_type          # 归一化类型（如BN，来自MODEL.HEADS.NORM）
    ):
        """
        This is an experimental feature.  # 新增这行，包含 "experimental"
        Args:
            feat_dim (int): dimension of feature before embedding head.
            embedding_dim (int): dimension of embedding feature.
            num_classes (int): number of classes for classification.
            # ... 其他参数说明 ...
        """
        super().__init__()  # 调用父类nn.Module的初始化方法

        # ---------------------------
        # 2.1 初始化池化层（对应POOL_LAYER配置）
        # ---------------------------
        # 校验池化类型是否合法（确保配置的pool_type在fastreid.layers.pooling中存在）
        assert hasattr(pooling, pool_type), \
            f"合法池化类型为{pooling.__all__}，但配置的是{pool_type}"
        # 根据配置创建池化层（如配置为Identity则不做池化，GlobalAvgPool则全局平均池化）
        self.pool_layer = getattr(pooling, pool_type)()

        # 记录特征来源配置（后续训练时选择用BN前/后的特征计算损失）
        self.neck_feat = neck_feat

        # ---------------------------
        # 2.2 初始化瓶颈层（对应EMBEDDING_DIM、WITH_BNNECK配置）
        # ---------------------------
        neck = []  # 用列表存储瓶颈层的组件（后续拼接为Sequential）
        # 1. 若EMBEDDING_DIM>0：添加1×1卷积降维（将backbone输出维度转为embedding_dim）
        if embedding_dim > 0:
            # 1×1卷积：输入维度feat_dim，输出维度embedding_dim，无偏置（避免与BN冲突）
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim  # 降维后，后续层的输入维度更新为embedding_dim
        # 2. 若WITH_BNNECK=True：添加BN层（稳定特征分布，增强判别性，解决协变量偏移）
        if with_bnneck:
            # get_norm：FastReID工具函数，根据norm_type创建归一化层（如BN）
            # bias_freeze=True：冻结BN的偏置（BN+Conv无偏置是常见最佳实践）
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))
        # 拼接瓶颈层组件（如EMBEDDING_DIM=0且WITH_BNNECK=True时，仅含BN层）
        self.bottleneck = nn.Sequential(*neck)

        # ---------------------------
        # 2.3 初始化分类层（对应CLS_TYPE、SCALE、MARGIN配置）
        # ---------------------------
        # 校验分类层类型是否合法（确保配置的cls_type在fastreid.layers.any_softmax中存在）
        assert hasattr(any_softmax, cls_type), \
            f"合法分类层类型为{any_softmax.__all__}，但配置的是{cls_type}"
        # 初始化分类权重（维度：[类别数, 特征维度]，每个类别对应一个权重向量）
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        # 根据配置创建分类层（如Linear/ArcSoftmax，传入scale和margin控制类间分离度）
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        # 初始化权重（瓶颈层用kaiming，分类权重用正态分布）
        self.reset_parameters()

    # ---------------------------
    # 3. 权重初始化：确保模型训练前参数处于合理范围
    # ---------------------------
    def reset_parameters(self) -> None:
        # 瓶颈层组件（卷积/BN）用kaiming初始化（适合ReLU激活后的层，避免梯度消失）
        self.bottleneck.apply(weights_init_kaiming)
        # 分类权重用正态分布初始化（均值0，标准差0.01，避免初始权重过大导致梯度爆炸）
        nn.init.normal_(self.weight, std=0.01)

    # ---------------------------
    # 4. 配置读取：从YAML/defaults.py提取参数，传给__init__
    # ---------------------------
    @classmethod  # 类方法：无需实例化即可调用，用于解析配置
    def from_config(cls, cfg):  # cfg：FastReID的配置对象（整合了YAML和默认值）
        # fmt: off  # 关闭代码格式化（保持参数对齐可读性）
        # 从配置中读取Head所需的所有参数（对应__init__的输入）
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM       # 主干网络输出维度（如ViT的768）
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM     # 嵌入维度（默认0，不降维）
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES       # 类别数（需在YAML中修改为数据集实际类别数）
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT         # 损失用特征来源（"before"/"after" BN）
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER        # 池化方式（如Identity）
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER         # 分类层类型（如Linear）
        scale         = cfg.MODEL.HEADS.SCALE             # 缩放系数（默认1）
        margin        = cfg.MODEL.HEADS.MARGIN            # 类别间隔（默认0.0）
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK       # 是否用BN瓶颈层（配置为True）
        norm_type     = cfg.MODEL.HEADS.NORM              # 归一化类型（如BN）
        # fmt: on  # 开启代码格式化
        # 返回参数字典（__init__会自动接收这些参数）
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    # ---------------------------
    # 5. 前向传播：模型的核心计算流程（训练/推理分支不同）
    # ---------------------------
    def forward(self, features, targets=None):  # features：主干网络输出的特征图；targets：训练时的标签（推理时为None）
        """
        前向传播逻辑：
        - 推理时：输出原始特征向量（用于ReID检索匹配）
        - 训练时：输出分类损失、带缩放的logits、损失用特征
        """
        # ---------------------------
        # 5.1 步骤1：特征聚合（池化操作）
        # ---------------------------
        # 对主干网络输出的特征图做池化（如Identity则直接保留特征图，GlobalAvgPool则转为向量）
        pool_feat = self.pool_layer(features)

        # ---------------------------
        # 5.2 步骤2：瓶颈层处理（降维+BN）
        # ---------------------------
        # 经过瓶颈层（如配置为EMBEDDING_DIM=0+WITH_BNNECK=True，则仅做BN归一化）
        neck_feat = self.bottleneck(pool_feat)
        # 将特征从4维（B, C, H, W）压缩为2维向量（B, C）：取H/W维度的第0个位置（适用于无池化/1×1特征图）
        neck_feat = neck_feat[..., 0, 0]  # ...表示匹配前面所有维度（B, C），0,0表示取空间维度的第一个元素

        # ---------------------------
        # 5.3 推理分支：仅输出特征向量（无需计算损失）
        # ---------------------------
        # fmt: off  # 关闭格式化（保持代码紧凑）
        if not self.training: return neck_feat  # self.training：PyTorch内置属性，推理时为False
        # fmt: on

        # ---------------------------
        # 5.4 训练分支：计算logits和分类损失
        # ---------------------------
        # 计算logits（分类层原始输出，未经过softmax，对应之前定义的logits概念）
        if self.cls_layer.__class__.__name__ == 'Linear':
            # 若分类层是普通Linear：直接用特征向量与分类权重做线性变换（logits = 特征 × 权重^T）
            logits = F.linear(neck_feat, self.weight)
        else:
            # 若分类层是ArcSoftmax/CosFace：先对特征和权重做L2归一化，再计算点积（确保余弦相似度范围）
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # 计算分类损失：传入logits副本（避免cls_layer内部的原地操作修改原始logits）
        # cls_layer会根据配置的margin/scale处理logits，输出适合交叉熵损失的结果
        cls_outputs = self.cls_layer(logits.clone(), targets)

        # ---------------------------
        # 5.5 选择损失计算用的特征（对应neck_feat配置）
        # ---------------------------
        # fmt: off
        if self.neck_feat == 'before':  # 若配置为"before"：用BN前的池化特征计算损失（如TripletLoss）
            feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after':  # 若配置为"after"：用BN后的瓶颈特征计算损失
            feat = neck_feat
        else:  # 配置非法时抛出错误
            raise KeyError(f"{self.neck_feat} 不是合法的MODEL.HEADS.NECK_FEAT值（仅支持before/after）")
        # fmt: on

        # ---------------------------
        # 5.6 返回训练所需的所有结果
        # ---------------------------
        return {
            "cls_outputs": cls_outputs,          # 分类损失的输入（已处理margin/scale）
            "pred_class_logits": logits.mul(self.cls_layer.s),  # 带缩放的logits（用于后续评估分类性能）
            "features": feat                     # 损失计算用的特征（如TripletLoss用）
        }