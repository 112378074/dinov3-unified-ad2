"""E2c — prototype-distance head + synthetic-anomaly distillation (walnuts).

Fixes E2b's two diagnosed root causes:
  (1) kNN sharpness doesn't distill into conv regression -> student head IS a min-distance:
      out(token) = min_m || sc*f(token) - p_m ||,  M=8192 learnable prototypes initialized from
      normal token vectors (teacher's per-layer scales sc applied as fixed constants).
  (2) student never saw the high-distance regime -> ~1/3 of distillation images are LAS
      synthetic-anomaly versions (teacher kNN scores them like any image).
Inference stays bank-free: prototypes are model weights. GATE >= 95% of teacher AUPRO."""
import os, sys, glob, random, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), 'dinov3-dual-branch-ad2'))
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import vit_encoder
from membank_derisk import build_bank, knn_map, load_img
from push_final import layer_scales_probe
from tiled_eval import hann2d
from dataset import PhotometricAug
from synth import las_augment
from ad2_pipeline_eval import pro_curve, aucpro_at
from utils import get_gaussian_kernel

CAT, DATA = 'walnuts', r'C:\Users\user\Desktop\datasets\ad2_dinomaly'
DTD = r'C:\Users\user\Desktop\EfficientAD\EfficientAD-main\dtd'
LAYERS, SCALE, RES, CROP, STRIDE = [7, 15, 23, 31], 0.625, 256, 576, 448
BANK_IMGS, STU_IMGS, N_SYN, M_PROTO = 300, 132, 60, 8192
ITERS, BS, LR = 3000, 8, 3e-4
DEV = 'cuda:0'
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TO_T = transforms.ToTensor()
torch.manual_seed(0); np.random.seed(0); random.seed(0)


def tiles_of(W, H):
    xs = list(range(0, max(W - CROP, 0) + 1, STRIDE)) or [0]
    ys = list(range(0, max(H - CROP, 0) + 1, STRIDE)) or [0]
    if xs[-1] != W - CROP: xs.append(max(W - CROP, 0))
    if ys[-1] != H - CROP: ys.append(max(H - CROP, 0))
    return [(x, y) for y in ys for x in xs]


class ProtoDist(nn.Module):
    """out(token) = min_m ||x - p_m|| — kNN's functional form with learnable prototypes."""
    def __init__(self, protos):
        super().__init__()
        self.p = nn.Parameter(protos.clone())

    def forward(self, x):                                   # [B,C,H,W] (already sc-scaled)
        B, C, H, W = x.shape
        q = x.permute(0, 2, 3, 1).reshape(-1, C)
        d = torch.cdist(q, self.p).min(1).values
        return d.reshape(B, 1, H, W)


def main():
    enc = vit_encoder.load('dinov3_vit_huge_16').to(DEV).eval().half()
    for p in enc.parameters(): p.requires_grad_(False)
    nreg = getattr(enc, 'num_register_tokens', getattr(enc, 'n_storage_tokens', 4))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEV)
    win = hann2d(CROP); photo = PhotometricAug()
    dtd = glob.glob(os.path.join(DTD, '**', '*.jpg'), recursive=True)
    tr = sorted(glob.glob(os.path.join(DATA, CAT, 'train', 'good', '*.png'))); random.shuffle(tr)
    bank_files, stu_files = tr[:BANK_IMGS], tr[BANK_IMGS:BANK_IMGS + STU_IMGS]
    tg = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'good', '*.png')))
    tb = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'bad', '*.png')))
    mb = SimpleNamespace(scale=SCALE, crop_size=CROP, overlap=128, metric='l2', keep_frac=0.3,
                         max_bank=100000, subsample='coreset', aug_every=2, chunk=30000,
                         resize=RES, knn=1, faithful=True, merge='hann')
    banks = build_bank(enc, bank_files, mb, LAYERS, nreg, DEV, photo)
    sc = layer_scales_probe(enc, banks, stu_files[:16], mb, LAYERS, nreg, DEV, SCALE)
    sc_t = torch.tensor([float(s) for s in sc], device=DEV).repeat_interleave(1280).view(1, -1, 1, 1)
    print(f'[E2c] bank ready, sc={[round(float(s),3) for s in sc]}', flush=True)

    def scaled_np(f):
        im = Image.open(f).convert('RGB')
        return np.array(im.resize((int(im.size[0] * SCALE), int(im.size[1] * SCALE))))

    @torch.no_grad()
    def tile_feats(til):
        t = NORM(torch.from_numpy(np.ascontiguousarray(til)).permute(2, 0, 1).float() / 255)
        out = enc.get_intermediate_layers(t.unsqueeze(0).to(DEV).half(), n=LAYERS, reshape=True)
        return torch.cat(out, 1).float() * sc_t             # sc-scaled -> teacher's metric space

    def teacher_map_np(img_np):                              # full scaled image (np) -> [256,256]
        t = TO_T(Image.fromarray(img_np))
        return np.asarray(knn_map(enc, banks, t, mb, LAYERS, nreg, gk, DEV, win, RES, sc), dtype=np.float32)

    # ---- distillation set: normal + synthetic-anomaly images ----
    TIL, TGT = [], []
    def harvest(img, tm):
        tmt = torch.from_numpy(tm)[None, None]
        H, W = img.shape[:2]
        for (x, y) in tiles_of(W, H):
            til = img[y:y + CROP, x:x + CROP]
            if til.shape[:2] != (CROP, CROP): continue
            gy = torch.linspace(y * RES / H, (y + CROP - 1) * RES / H, 36) / (RES - 1) * 2 - 1
            gx = torch.linspace(x * RES / W, (x + CROP - 1) * RES / W, 36) / (RES - 1) * 2 - 1
            gr = torch.stack(torch.meshgrid(gy, gx, indexing='ij'), -1)[None][..., [1, 0]]
            TIL.append(til.copy()); TGT.append(F.grid_sample(tmt, gr, align_corners=True).squeeze())

    for i, f in enumerate(stu_files):
        img = scaled_np(f)
        harvest(img, teacher_map_np(img))
        if i < N_SYN:                                        # synthetic-anomaly version
            aug, _m = las_augment(img, dtd)
            harvest(aug, teacher_map_np(aug))
        if i % 25 == 0: print(f'  distill imgs {i}/{len(stu_files)} tiles={len(TIL)}', flush=True)
    TGT = torch.stack(TGT); mu, sd = TGT.mean().item(), TGT.std().item()
    print(f'[E2c] {len(TIL)} tiles mu={mu:.3f} sd={sd:.3f}', flush=True)

    # ---- prototype init: token vectors sampled from normal tiles (sc-scaled space) ----
    vecs = []
    for k in np.random.choice(len(TIL), 40, replace=False):
        vecs.append(tile_feats(TIL[k]).flatten(2).squeeze(0).T.cpu())
    vecs = torch.cat(vecs)
    protos = vecs[torch.randperm(vecs.shape[0])[:M_PROTO]].to(DEV)
    head = ProtoDist(protos).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)
    print(f'[E2c] protos {tuple(protos.shape)}', flush=True)

    for it in range(ITERS):
        j = np.random.randint(0, len(TIL), BS)
        x = torch.cat([tile_feats(TIL[k]) for k in j])
        y = ((TGT[j].to(DEV) - mu) / sd).unsqueeze(1)
        pred = (head(x) - mu) / sd
        loss = F.smooth_l1_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 300 == 0: print(f'  iter {it}/{ITERS} loss {loss.item():.4f}', flush=True)
    torch.save(head.state_dict(), os.path.join(_ROOT, 'e2c_head.pth'))

    # ---- eval: tiled proto-student (Hann stitch) vs teacher ----
    head.eval()
    hw = torch.as_tensor(np.asarray(hann2d(CROP), dtype=np.float32)).reshape(CROP, CROP)
    gts, tms, sms = [], [], []
    for i, f in enumerate(tg + tb):
        img = scaled_np(f); H, W = img.shape[:2]
        tms.append(teacher_map_np(img))
        acc = torch.zeros(H, W); wsum = torch.zeros(H, W)
        for (x, y) in tiles_of(W, H):
            til = img[y:y + CROP, x:x + CROP]
            if til.shape[:2] != (CROP, CROP): continue
            with torch.no_grad():
                pm = head(tile_feats(til)).squeeze()
            pm = F.interpolate(pm[None, None], size=CROP, mode='bilinear', align_corners=False).squeeze().cpu()
            acc[y:y + CROP, x:x + CROP] += pm * hw; wsum[y:y + CROP, x:x + CROP] += hw
        sm = F.interpolate((acc / wsum.clamp_min(1e-6))[None, None], size=RES,
                           mode='bilinear', align_corners=False).to(DEV)
        sms.append(gk(sm).squeeze().cpu().numpy())
        stem = os.path.splitext(os.path.basename(f))[0]
        is_bad = ('\\bad\\' in f) or ('/bad/' in f)
        gp = os.path.join(DATA, CAT, 'ground_truth', 'bad', stem + '_mask.png')
        gts.append((np.array(Image.open(gp).convert('L').resize((RES, RES), Image.NEAREST)) > 0).astype(np.uint8)
                   if (is_bad and os.path.exists(gp)) else np.zeros((RES, RES), np.uint8))
        if i % 40 == 0: print(f'  test {i}/{len(tg)+len(tb)}', flush=True)
    gts, tms, sms = np.stack(gts), np.stack(tms), np.stack(sms)
    pro_t = aucpro_at(*pro_curve(gts, tms), 0.05) * 100
    pro_s = aucpro_at(*pro_curve(gts, sms), 0.05) * 100
    r = pro_s / max(pro_t, 1e-9); c = float(np.corrcoef(tms.ravel(), sms.ravel())[0, 1])
    print(f'\n[E2c RESULT] teacher {pro_t:.2f} | proto-student {pro_s:.2f} | ratio {100*r:.1f}% | corr {c:.4f}', flush=True)
    print(f'[E2c GATE {"PASS" if r >= 0.95 else "FAIL"}]', flush=True)
    json.dump({'teacher_aupro': pro_t, 'student_aupro': pro_s, 'ratio': r, 'corr': c, 'M': M_PROTO},
              open(os.path.join(_ROOT, 'e2c_result.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(_ROOT, 'e2c_maps.npz'), tms=tms, sms=sms, gts=gts)


if __name__ == '__main__':
    main()
