"""E5a — GLOBAL-embedding kNN (logical-anomaly path) for the KEEP-the-bank merge (vial + walnuts).

The local patch-kNN membank is STRUCTURALLY blind to compositional/logical anomalies (each patch
locally normal). This adds a SECOND, coarser bank over region- and image-level embeddings — the
global context the local branch lacks. Measured here, not distilled away; both banks are kept.

  region path: whole image @512 -> DINOv3 layers [23,31] -> [C,32,32] -> avg-pool 8x8 = 64 region
    embeddings; bank = all normal region embeddings (position-agnostic, shift-robust); test region
    min-L2 -> 8x8 map -> upsample 256 (coarse region-appearance anomaly)
  image path: global-avg-pool -> 1 embedding; bank = normal image embeddings; kNN -> image logical score

Compares vs the deployed LOCAL fused map (from fuse_cache_retrain) and a CONTEXT-GATE fusion
(local modulated by global). Reports full-test AUPRO/AUROC of local / global / fused, image-level
logical separation, and peak-in-GT on the user-named logical cases (vial 008, walnuts open-shell).
GATE: global path adds AUPRO on the logical-heavy cat OR image-score separates logical from normal.
"""
import os, sys, glob, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), 'dinov3-dual-branch-ad2'))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from scipy import ndimage
from models import vit_encoder
from ad2_pipeline_eval import pro_curve, aucpro_at

DATA = r'C:\Users\user\Desktop\datasets\ad2_dinomaly'
ENC_RES, GRID, LAYERS, RES = 512, 8, [23, 31], 256
DEV = 'cuda:0'
TT = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
HIRES = {'walnuts'}
FUS = {'vial': (0.25, 0.0), 'walnuts': (0.25, 0.1)}
torch.manual_seed(0); np.random.seed(0)
gz = lambda x, r: (x - r.mean()) / (r.std() + 1e-6)


@torch.no_grad()
def embeds(enc, f):
    img = TT(Image.open(f).convert('RGB').resize((ENC_RES, ENC_RES))).unsqueeze(0).to(DEV).half()
    out = enc.get_intermediate_layers(img, n=LAYERS, reshape=True)
    feat = torch.cat(out, 1)[0].float()                              # [C,32,32]
    reg = F.avg_pool2d(feat[None], feat.shape[-1] // GRID)[0]        # [C,8,8]
    reg = reg.reshape(reg.shape[0], -1).T                            # [64,C]
    reg = F.normalize(reg, dim=1)
    im = F.normalize(feat.mean((1, 2)), dim=0)                       # [C]
    return reg.cpu(), im.cpu()


def local_map(cat):
    src = 'fuse_cache_retrain_hires' if cat in HIRES else 'fuse_cache_retrain'
    d = np.load(f'C:\\Users\\user\\Desktop\\Dinomaly2\\{src}\\{cat}.npz', allow_pickle=True)
    base = np.load(f'C:\\Users\\user\\Desktop\\Dinomaly2\\fuse_cache_retrain\\{cat}.npz', allow_pickle=True)
    zA, zB = gz(d['tA'], d['hA']), gz(d['tB'], d['hB'])
    lam, pen = FUS[cat]
    tF = zA + lam * zB - pen * np.maximum(zA - zB, 0)
    if cat in HIRES:                                                  # 768 -> 256
        tF = np.stack([np.asarray(Image.fromarray(m).resize((RES, RES), Image.BILINEAR)) for m in tF])
    return tF, base['gts'], [str(x) for x in base['files']]


def aupro(g, m): return aucpro_at(*pro_curve(g, m), 0.05) * 100
def auroc(g, m): return roc_auc_score(g.ravel(), m.ravel().astype(np.float64), max_fpr=0.05) * 100 if g.max() else float('nan')


def main():
    enc = vit_encoder.load('dinov3_vit_huge_16').to(DEV).eval().half()
    for p in enc.parameters():
        p.requires_grad_(False)
    out_all = {}
    for cat in ['vial', 'walnuts']:
        tr = sorted(glob.glob(os.path.join(DATA, cat, 'train', 'good', '*.png')))
        reg_bank, im_bank = [], []
        for i, f in enumerate(tr):
            r, im = embeds(enc, f)
            reg_bank.append(r); im_bank.append(im)
            if i % 80 == 0: print(f'  [{cat}] bank {i}/{len(tr)}', flush=True)
        reg_bank = torch.cat(reg_bank).to(DEV)                        # [N*64,C]
        im_bank = torch.stack(im_bank).to(DEV)                        # [N,C]
        print(f'[{cat}] region bank {tuple(reg_bank.shape)} image bank {tuple(im_bank.shape)}', flush=True)

        local, gts, files = local_map(cat)
        glob_maps, img_scores = [], []
        for i, f in enumerate(files):
            r, im = embeds(enc, f)
            dr = torch.cdist(r.to(DEV), reg_bank).min(1).values.reshape(GRID, GRID)   # [8,8]
            gm = F.interpolate(dr[None, None], size=RES, mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            glob_maps.append(gm)
            img_scores.append(float(torch.cdist(im[None].to(DEV), im_bank).min()))
            if i % 40 == 0: print(f'  [{cat}] test {i}/{len(files)}', flush=True)
        glob_maps = np.stack(glob_maps); img_scores = np.array(img_scores)
        labels = np.array([1 if gts[i].sum() > 0 else 0 for i in range(len(files))])

        # context-gate fusion: local modulated by global (z-scored), tuned-light
        zl = (local - local.mean()) / (local.std() + 1e-9)
        zgm = (glob_maps - glob_maps.mean()) / (glob_maps.std() + 1e-9)
        gate = 1 / (1 + np.exp(-zgm))
        fused = zl * (0.5 + 0.5 * gate) + 0.3 * zgm

        res = {'AUPRO_local': aupro(gts, local), 'AUPRO_global': aupro(gts, glob_maps),
               'AUPRO_fused': aupro(gts, fused), 'AUROC_local': auroc(gts, local),
               'AUROC_global': auroc(gts, glob_maps), 'AUROC_fused': auroc(gts, fused),
               'img_logical_AUROC': roc_auc_score(labels, img_scores) * 100 if labels.max() else float('nan')}
        # peak-in-GT on defective images: local vs global vs fused
        def pk(maps):
            hit = tot = 0
            for i in range(len(files)):
                if gts[i].sum() == 0: continue
                py, px = np.unravel_index(np.argmax(maps[i]), maps[i].shape)
                gd = ndimage.binary_dilation(gts[i] > 0, iterations=4)
                hit += int(gd[py, px]); tot += 1
            return 100 * hit / max(tot, 1)
        res['peakGT_local'] = pk(local); res['peakGT_global'] = pk(glob_maps); res['peakGT_fused'] = pk(fused)
        out_all[cat] = res
        print(f"\n[E5a {cat}] AUPRO local {res['AUPRO_local']:.1f} global {res['AUPRO_global']:.1f} "
              f"fused {res['AUPRO_fused']:.1f} | peakGT local {res['peakGT_local']:.0f} global "
              f"{res['peakGT_global']:.0f} fused {res['peakGT_fused']:.0f} | img-logical AUROC "
              f"{res['img_logical_AUROC']:.1f}", flush=True)
        np.savez_compressed(os.path.join(_ROOT, f'e5a_{cat}_maps.npz'), glob=glob_maps, local=local,
                            fused=fused, gts=gts, files=np.array(files), img_scores=img_scores)
    json.dump(out_all, open(os.path.join(_ROOT, 'e5a_result.json'), 'w'), indent=1)
    print('\n=== E5a SUMMARY ===')
    for c, r in out_all.items():
        d = r['AUPRO_fused'] - r['AUPRO_local']
        print(f"{c:10s} AUPRO local {r['AUPRO_local']:.1f} -> fused {r['AUPRO_fused']:.1f} ({d:+.1f}) | "
              f"global-alone {r['AUPRO_global']:.1f} | img-logical AUROC {r['img_logical_AUROC']:.1f}")


if __name__ == '__main__':
    main()
