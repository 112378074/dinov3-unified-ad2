"""Unified-AD losses (design v0): the four proven objectives + the unification distillation loss.
    L = L_recon + L_coh + w_d*L_disc + w_s*L_seg + w_dist*L_distill
L_distill = SmoothL1(head_D(x), stopgrad(bank_knn(x))) on normal AND synthetic tiles — the memory
bank is a TRAIN-TIME teacher only (dropped at inference). See docs/PROPOSAL.md §3."""
import os, sys
_S = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dinov3-dual-branch-ad2')
sys.path.insert(0, _S) if os.path.isdir(_S) and _S not in sys.path else None
import torch.nn.functional as F
from loss import reconstruction_loss, segmentation_loss, focal_bce, gas_attack   # proven set


def distillation_loss(student_dist, teacher_dist):
    """Bank-distillation: student distance map vs stop-grad kNN teacher (SmoothL1)."""
    return F.smooth_l1_loss(student_dist, teacher_dist.detach())
