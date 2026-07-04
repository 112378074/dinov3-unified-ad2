# DINOv3 Unified-AD2 — single-training anomaly detection (experimental)

**Goal.** The proven system ([dinov3-dual-branch-ad2](https://github.com/112378074/dinov3-dual-branch-ad2))
meets all four MVTec-AD-2 targets (AU-PRO 72.46 / AU-ROC 85.05 / Seg-F1 63.89 / Class-F1 83.65) but
needs **two separate constructions**: a non-parametric memory bank (Branch A) *and* a trained
INP-Former++ (Branch B), fused per category per metric. This project explores collapsing that into
**ONE training run → one network → all four metrics**, plus the improvement proposals from the
dual-branch error analysis.

**Core idea — bank distillation + multi-head decoupling (design v0, see docs/PROPOSAL.md):**
during training, the memory bank acts as a *teacher only*: a student "distance head" learns to
reproduce the bank's kNN distance map, alongside the existing reconstruction / segmentation /
discriminative objectives. At inference the bank is gone — a single forward pass emits three
decoupled outputs (float map, binary-seg map, image score), mirroring the per-metric routing that
made the dual-branch work.

**Status: experimental.** Baseline to beat (from the dual-branch retrain verification, single-recipe):
AU-PRO 72.20 / AU-ROC 84.56 / Seg-F1 63.69 / Class-F1 83.65.

## Layout
```
main.py / model.py / train.py / loss.py / eval.py / utils.py   standard entry points (skeleton)
docs/PROPOSAL.md     unified-architecture design + experiment plan + risks
configs/unified.yaml experiment configuration
```
Engine dependencies (DINOv3 loader, dataset/synthesis, metrics) are reused from the sibling
`dinov3-dual-branch-ad2` package during experimentation.
