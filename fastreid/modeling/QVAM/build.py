# encoding: utf-8

from ...utils.registry import Registry

QVAM_REGISTRY = Registry("QVAM")
QVAM_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`fastreid.config.CfgNode`
It must returns an instance of :class:`Backbone`.
"""


def build_QVAM(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """

    QVAM_name = cfg.MODEL.QVAM.NAME
    qvam = QVAM_REGISTRY.get(QVAM_name)(cfg)
    return qvam
