# Unified Single-Training Architecture — Design Proposal v0

## 1. Why unify

The dual-branch system needs two constructions per category — a memory bank (build: coreset over
~500 normal images' features) and a trained INP-Former++ — plus hand-tuned per-category fusion.
Costs: bank rebuild at every eval, two hyperparameter surfaces, and the fusion constants
(lam/pen/gainA/erode/adopt) are dev-set-tuned. Target: **one training run → one network → one
forward pass → all four metrics**, at or above the dual-branch baseline
(AU-PRO 72.20 / AU-ROC 84.56 / Seg-F1 63.69 / Class-F1 83.65 — the retrain-verified numbers).

## 2. What each branch actually contributes (measured, do not lose these)

| Property | Carrier | Evidence |
|---|---|---|
| Sharp binary masks (Seg-F1) & image score (Class-F1) | Branch A distance map | ClassF1 byte-exact across retrains (A deterministic); SegF1 61-64 all from A |
| Low-FPR region ranking (AU-PRO) | A + B fused float | per-cat pen/lam fusion +1.9 AUPRO over lam0.5 |
| FP suppression where A over-fires | B as a gate | pen·ReLU(zA−zB): sheet_metal +6.7, fj +6.5, wallplugs +4.5 |
| Sub-token defect visibility | hires (scale-2.0) re-encode | can 7→37, fj 45→58, walnuts 50→69, wallplugs 67→76 (object-scene × sub-token × weak-base rule) |
| Training-noise robustness | A non-parametric | retrain drift ≤0.5 MEAN, confined to B paths |

## 3. Design v0 — bank-distilled multi-head network

**One frozen DINOv3 ViT-H+ encoder → one trainable neck → three decoupled heads.**

```
frozen DINOv3 ─► neck (INP-Former++ bottleneck+decoder, M=6→64 prototypes)
   ├── head_D  "distance"  : predicts the membank kNN distance map   (teacher: bank, TRAIN-time only)
   ├── head_S  "seg"       : residual seg head (existing)             (teacher: LAS synthetic masks)
   └── head_R  "recon"     : feature reconstruction residual          (self-supervised, existing)
outputs: float = gz(head_D) + lam·gz(head_R) − pen·ReLU(...)  [lam/pen LEARNED, not hand-tuned]
         binary = SuperADD-close(head_D)  ;  score = MEBin(head_D)
```

Key moves:
1. **Bank distillation (the unification trick).** During training, build the bank once (as today) but
   use it only as a *teacher*: `L_dist = SmoothL1(head_D(x), stopgrad(kNN_bank(x)))` on normal AND
   synthetic-anomaly tiles. The student inherits A's sharp, training-robust distance geometry;
   at inference the bank is dropped. Risk: student smooths the teacher's sharp minima → monitor
   with the binary-path metrics early.
2. **Enlarged prototype memory (M=6 → 64, ablate 6/16/64/256).** INP prototypes already are a tiny
   learned memory; scaling M toward a "learnable coreset" may let head_D exceed its teacher.
3. **Multi-scale training = hires built in.** Train on mixed tiles (scale 0.625 AND 2.0) with
   scale-conditioning (a scale token), so ONE model serves both resolutions; keeps the verified
   per-cat hires routing without two encode passes at eval... (eval still encodes twice for hires
   cats — unavoidable — but same weights).
4. **Learned fusion.** Replace hand pen/lam with a 1×1 gate over (zD, zR) trained on synthetic-val
   AUPRO surrogate loss — removes the dev-set-tuned constants (legitimacy win).

## 4. Improvement proposals from the per-image error analysis (2026-07-04, retrain weights)

Measured per-image behavior (logs/5090/per_image_analysis.csv):

| cat | miss% (bad) | medF1 when hit | FP% on goods | float peak-in-GT% | diagnosis |
|---|---|---|---|---|---|
| can | 98.8 | .003 | 56 | 1.2 | structural (label high-freq drowns scratch) |
| fabric | 72.6 | .69 | 3 | 43 | big-defect images carry pixel-F1; small-defect images missed |
| wallplugs | 74.4 | .85 | 20 | 37 | threshold too tight per-image; hits are clean |
| vial | 0.0 | .64 | **74** | 81 | threshold too loose: recall 100%, goods flooded |
| sheet_metal | 34.4 | .35 | 0 | **97** | float map excellent; binarization is the bottleneck |
| fruit_jelly | 33.3 | .70 | 5 | 70 | healthy-ish; residual miss = sub-token specks |
| rice | 41.1 | .84 | 0 | 51 | healthy |
| walnuts | 33.3 | .80 | 13 | 86 | healthy |

Proposals (P1-P4 testable in the unified project; P5 for the dual-branch too):
- **P1 per-image-aware threshold head.** vial/wallplugs show one global per-cat threshold cannot fit
  every image. A small head predicting a per-image threshold offset from the image's own normal-score
  statistics (trained on synthetic-val, never test) targets vial's 74% good-FP and wallplugs' 74% miss.
- **P2 seg-from-float for sheet_metal-type cats.** peak-in-GT 97% but F1 .35 → binarize the FLOAT map
  (head_D+gates) instead of the raw distance for cats where float ≫ binary.
- **P3 small-defect recall (fabric).** fabric misses 73% of (small-defect) images while pixel-F1 stays
  96 — dataset-level metric hides it. Add a per-image detection auxiliary loss (image-level BCE from
  the seg head max) so small defects aren't ignored by area-dominated losses.
- **P4 can: accept or sensor-level.** All representation-level routes measured dead (memory:
  synth-content gap, high-freq drowned). Only unified-model hope: head_D distilled at hires + P1.
- **P5 legitimacy.** Learned fusion/thresholds trained on synthetic-val replace every dev-set-tuned
  constant — turns the audit flag into a design feature.

## 5. Experiment plan (each step gated on the previous)

| # | experiment | success gate | cost |
|---|---|---|---|
| E1 | head_D distillation, 1 cat (walnuts): student vs teacher distance map AUPRO/SegF1 | ≥95% of teacher | ~2h |
| E2 | + prototypes M ablation 6/16/64 | student ≥ teacher | ~4h |
| E3 | 8-cat unified train + learned fusion | MEAN ≥ dual-branch −1.0 | ~12h |
| E4 | + multi-scale conditioning (hires-in-one) | can/fj/wallplugs/walnuts keep hires gains | ~12h |
| E5 | P1-P3 improvement heads | SegF1 > 63.9 or AUPRO > 72.5 | ~8h |

Risks: distillation smoothing (E1 gate), prototype collapse at large M (known dual-bank failure —
synthetic manifold overlap; mitigation: distance-only supervision, no synthetic prototypes),
multi-scale interference (E4 gate; fallback = per-scale BN/LoRA).
