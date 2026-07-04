"""E2d — E2b conv head + synthetic-anomaly distillation tiles (targets the tail: per-image gap mean -0.7 but worst-5 images -36 to -38) (walnuts). Fix for E2a-v2's resolution gap (86.4%):
the student now works in the SAME tile geometry as the teacher (576-crops @ scale 0.625,
stride 448), one conv-head forward PER TILE, Hann-stitched — inference stays bank/kNN-free.
Training pairs: (tile, teacher-map patch cropped from the stitched kNN map). GATE >= 95%."""
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

CAT = sys.argv[1] if len(sys.argv) > 1 else 'walnuts'
DATA = r'C:\Users\user\Desktop\datasets\ad2_dinomaly'
_CACHE = os.path.join(_ROOT, 'e3_cache'); os.makedirs(_CACHE, exist_ok=True)
LAYERS, SCALE, RES, CROP, STRIDE = [7, 15, 23, 31], 0.625, 256, 576, 448
BANK_IMGS, STU_IMGS, N_SYN, ITERS, BS, LR = 300, 132, 60, 4000, 12, 2e-4
DEV = 'cuda:0'
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
torch.manual_seed(0); np.random.seed(0); random.seed(0)


def tiles_of(W, H):
    xs = list(range(0, max(W - CROP, 0) + 1, STRIDE)) or [0]
    ys = list(range(0, max(H - CROP, 0) + 1, STRIDE)) or [0]
    if xs[-1] != W - CROP: xs.append(max(W - CROP, 0))
    if ys[-1] != H - CROP: ys.append(max(H - CROP, 0))
    return [(x, y) for y in ys for x in xs]


@torch.no_grad()
def tile_feats(enc, til):                                    # til uint8 HWC
    t = NORM(torch.from_numpy(til).permute(2, 0, 1).float() / 255).unsqueeze(0).to(DEV).half()
    return torch.cat(enc.get_intermediate_layers(t, n=LAYERS, reshape=True), 1).float()  # [1,5120,36,36]


def main():
    enc = vit_encoder.load('dinov3_vit_huge_16').to(DEV).eval().half()
    for p in enc.parameters(): p.requires_grad_(False)
    nreg = getattr(enc, 'num_register_tokens', getattr(enc, 'n_storage_tokens', 4))
    from utils import get_gaussian_kernel
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEV)
    win = hann2d(CROP); photo = PhotometricAug()
    tr = sorted(glob.glob(os.path.join(DATA, CAT, 'train', 'good', '*.png'))); random.shuffle(tr)
    # adaptive split: small cats (sheet_metal 137, fj 263, vial 291, wallplugs 293 train imgs) can't
    # afford bank=300 — take 70% for the bank (cap 300), rest for distillation (fixed E3a crash:
    # tr[300:] was EMPTY -> "stack expects a non-empty TensorList").
    bank_n = min(BANK_IMGS, max(60, int(len(tr) * 0.7)))
    bank_files, stu_files = tr[:bank_n], tr[bank_n:bank_n + STU_IMGS]
    tg = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'good', '*.png')))
    tb = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'bad', '*.png')))
    mb = SimpleNamespace(scale=SCALE, crop_size=CROP, overlap=128, metric='l2', keep_frac=0.3,
                         max_bank=100000, subsample='coreset', aug_every=2, chunk=30000,
                         resize=RES, knn=1, faithful=True, merge='hann')
    banks = build_bank(enc, bank_files, mb, LAYERS, nreg, DEV, photo)
    sc = layer_scales_probe(enc, banks, stu_files[:16], mb, LAYERS, nreg, DEV, SCALE)
    print('[E3 {CAT}] bank ready', flush=True)

    def scaled_np(f):
        im = Image.open(f).convert('RGB')
        return np.array(im.resize((int(im.size[0] * SCALE), int(im.size[1] * SCALE))))

    # ---- training tiles + teacher patch targets ----
    dtd = glob.glob(r'C:\Users\user\Desktop\EfficientAD\EfficientAD-main\dtd\**\*.jpg', recursive=True)
    from torchvision.transforms import ToTensor as _TT2
    def teacher_map_np(img_np):
        t = _TT2()(Image.fromarray(img_np))
        return np.asarray(knn_map(enc, banks, t, mb, LAYERS, nreg, gk, DEV, win, RES, sc), dtype=np.float32)
    TIL, TGT = [], []
    for i, f in enumerate(stu_files):
        tm = np.asarray(knn_map(enc, banks, load_img(f, SCALE), mb, LAYERS, nreg, gk, DEV, win, RES, sc),
                        dtype=np.float32)                    # [256,256]
        img = scaled_np(f); H, W = img.shape[:2]
        tmt = torch.from_numpy(tm)[None, None]
        for (x, y) in tiles_of(W, H):
            til = img[y:y + CROP, x:x + CROP]
            if til.shape[:2] != (CROP, CROP): continue
            # teacher patch: map region for this tile -> 36x36
            x0, y0 = x * RES / W, y * RES / H
            w0, h0 = CROP * RES / W, CROP * RES / H
            gy = torch.linspace(y0, y0 + h0 - 1, 36) / (RES - 1) * 2 - 1
            gx = torch.linspace(x0, x0 + w0 - 1, 36) / (RES - 1) * 2 - 1
            gr = torch.stack(torch.meshgrid(gy, gx, indexing='ij'), -1)[None][..., [1, 0]]
            patch = F.grid_sample(tmt, gr, align_corners=True).squeeze()
            TIL.append(til); TGT.append(patch)
        if i < N_SYN:                                    # synthetic-anomaly image (teacher scores it)
            aug, _m = las_augment(img, dtd)
            tms2 = teacher_map_np(aug)
            tmt2 = torch.from_numpy(tms2)[None, None]
            for (x, y) in tiles_of(W, H):
                til = aug[y:y + CROP, x:x + CROP]
                if til.shape[:2] != (CROP, CROP): continue
                x0, y0 = x * RES / W, y * RES / H
                w0, h0 = CROP * RES / W, CROP * RES / H
                gy = torch.linspace(y0, y0 + h0 - 1, 36) / (RES - 1) * 2 - 1
                gx = torch.linspace(x0, x0 + w0 - 1, 36) / (RES - 1) * 2 - 1
                gr = torch.stack(torch.meshgrid(gy, gx, indexing='ij'), -1)[None][..., [1, 0]]
                TIL.append(til.copy()); TGT.append(F.grid_sample(tmt2, gr, align_corners=True).squeeze())
        if i % 30 == 0: print(f'  train maps {i}/{len(stu_files)} tiles={len(TIL)}', flush=True)
    TGT = torch.stack(TGT); mu, sd = TGT.mean().item(), TGT.std().item()
    print(f'[E3 {CAT}] {len(TIL)} tiles  mu={mu:.3f} sd={sd:.3f}', flush=True)

    head = nn.Sequential(nn.Conv2d(5120, 512, 1), nn.GELU(),
                         nn.Conv2d(512, 256, 3, padding=1), nn.GELU(),
                         nn.Conv2d(256, 1, 1)).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    for it in range(ITERS):
        j = np.random.randint(0, len(TIL), BS)
        x = torch.cat([tile_feats(enc, TIL[k]) for k in j])
        y = ((TGT[j].to(DEV) - mu) / sd).unsqueeze(1)
        loss = F.smooth_l1_loss(head(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 300 == 0: print(f'  iter {it}/{ITERS} loss {loss.item():.4f}', flush=True)

    torch.save(head.state_dict(), os.path.join(_CACHE, f'{CAT}_head.pth'))   # never lose training again
    # ---- eval: student tiled+Hann-stitched vs teacher ----
    head.eval()
    hw = torch.as_tensor(np.asarray(hann2d(CROP), dtype=np.float32)).reshape(CROP, CROP)  # hann2d -> [1,1,C,C]
    gts, tms, sms = [], [], []
    for i, f in enumerate(tg + tb):
        tms.append(np.asarray(knn_map(enc, banks, load_img(f, SCALE), mb, LAYERS, nreg, gk, DEV, win, RES, sc), dtype=np.float32))
        img = scaled_np(f); H, W = img.shape[:2]
        acc = torch.zeros(H, W); wsum = torch.zeros(H, W)
        for (x, y) in tiles_of(W, H):
            til = img[y:y + CROP, x:x + CROP]
            if til.shape[:2] != (CROP, CROP): continue
            with torch.no_grad():
                pm = head(tile_feats(enc, til)).squeeze() * sd + mu       # [36,36]
            pm = F.interpolate(pm[None, None], size=CROP, mode='bilinear', align_corners=False).squeeze().cpu()
            acc[y:y + CROP, x:x + CROP] += pm * hw; wsum[y:y + CROP, x:x + CROP] += hw
        sm = (acc / wsum.clamp_min(1e-6))[None, None]
        sm = F.interpolate(sm, size=RES, mode='bilinear', align_corners=False).to(DEV)
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
    print(f'\n[E3 {CAT} RESULT] teacher {pro_t:.2f} | student(tiled) {pro_s:.2f} | ratio {100*r:.1f}% | corr {c:.4f}', flush=True)
    print(f'[E3 {CAT} GATE {"PASS" if r >= 0.95 else "FAIL"}]', flush=True)
    json.dump({'teacher_aupro': pro_t, 'student_aupro': pro_s, 'ratio': r, 'corr': c},
              open(os.path.join(_ROOT, 'e2d_result.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(_CACHE, f'{CAT}.npz'), tms=tms, sms=sms, gts=gts, files=np.array(tg+tb))


if __name__ == '__main__':
    main()
