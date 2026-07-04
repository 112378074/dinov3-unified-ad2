"""UnifiedAD — single-training anomaly detector (design v0, see docs/PROPOSAL.md).

One frozen DINOv3 encoder -> one trainable neck (INP-Former++ bottleneck+decoder, enlarged
prototype memory) -> three decoupled heads:
    head_D  distance head — distilled from the memory-bank kNN distance (teacher, train-time only)
    head_S  residual segmentation head (LAS synthetic-mask supervised)
    head_R  reconstruction residual (self-supervised)
Inference is a single forward pass; the decoupled per-metric routing of the dual-branch system is
preserved as OUTPUT HEADS instead of separate branches.

Skeleton status: E1 (single-cat distillation) not yet run — constructors defined, heads reuse the
proven INP-Former++ modules from the sibling dinov3-dual-branch-ad2 package.
"""
import os
import sys

# reuse the proven engine from the sibling packaged project
_SIBLING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dinov3-dual-branch-ad2')
if os.path.isdir(_SIBLING) and _SIBLING not in sys.path:
    sys.path.insert(0, _SIBLING)

import torch
import torch.nn as nn


def build_unified(backbone='dinov3_vit_huge_16', n_proto=64, decoder_attn='relu', seg_res=512,
                  device='cuda:0'):
    """Construct the unified model: INP-Former++ trunk with enlarged prototype memory + an extra
    distance head. E1/E2 experiments ablate n_proto in {6,16,64,256}."""
    from model import build_branch_b                     # sibling package's proven constructor
    trunk, embed_dim = build_branch_b(backbone=backbone, n_proto=n_proto,
                                      decoder_attn=decoder_attn, seg_res=seg_res, device=device)
    trunk.head_d = nn.Sequential(                        # distance head (bank-distillation student)
        nn.Conv2d(embed_dim, embed_dim // 4, 3, padding=1), nn.GELU(),
        nn.Conv2d(embed_dim // 4, 1, 1),
    ).to(device)
    return trunk, embed_dim
