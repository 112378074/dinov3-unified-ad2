"""E2a — distillation against a DEPLOYMENT-GRADE teacher (walnuts).

E1 passed with a weak token-space teacher (25.9 AUPRO). E2a repeats the gate against the real
thing: the deployed Branch-A recipe (576-crop tiles @ scale 0.625, coreset bank, Hann stitch,
res-256 maps; walnuts territory ~70 AUPRO). The student stays a single-forward conv head on
1024-resize DINOv3 features (64x64 tokens -> upsampled) — i.e. the exact question of the unified
design: can ONE forward pass replace the multi-tile kNN pipeline?

GATE: student test AUPRO >= 95% of teacher test AUPRO. (E2b prototype ablation follows a pass.)
"""
import os, sys, glob, random, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SIB = os.path.join(os.path.dirname(_ROOT), 'dinov3-dual-branch-ad2')
sys.path.insert(0, _SIB)
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
from ad2_pipeline_eval import pro_curve, aucpro_at
from utils import get_gaussian_kernel

CAT = 'walnuts'
DATA = r'C:\Users\user\Desktop\datasets\ad2_dinomaly'
LAYERS = [7, 15, 23, 31]
SCALE, RES = 0.625, 256
BANK_IMGS, STU_IMGS = 300, 150
ITERS, BS, LR = 2000, 8, 2e-4
DEV = 'cuda:0'
TT = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
torch.manual_seed(0); np.random.seed(0); random.seed(0)


@torch.no_grad()
def feats(enc, path):
    img = TT(Image.open(path).convert('RGB').resize((1024, 1024))).unsqueeze(0).to(DEV).half()
    return torch.cat(enc.get_intermediate_layers(img, n=LAYERS, reshape=True), 1).float()  # [1,5120,64,64]


def main():
    enc = vit_encoder.load('dinov3_vit_huge_16').to(DEV).eval().half()
    for p in enc.parameters():
        p.requires_grad_(False)
    nreg = getattr(enc, 'num_register_tokens', getattr(enc, 'n_storage_tokens', 4))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEV)
    win = hann2d(576)
    photo = PhotometricAug()

    tr = sorted(glob.glob(os.path.join(DATA, CAT, 'train', 'good', '*.png')))
    random.shuffle(tr)
    bank_files, stu_files = tr[:BANK_IMGS], tr[BANK_IMGS:BANK_IMGS + STU_IMGS]
    tg = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'good', '*.png')))
    tb = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'bad', '*.png')))
    print(f'[E2a] bank {len(bank_files)} | student {len(stu_files)} | test {len(tg)}+{len(tb)}', flush=True)

    # ---- deployment-grade teacher (deployed Branch-A recipe) ----
    mb = SimpleNamespace(scale=SCALE, crop_size=576, overlap=128, metric='l2', keep_frac=0.3,
                         max_bank=100000, subsample='coreset', aug_every=2, chunk=30000,
                         resize=RES, knn=1, faithful=True, merge='hann')
    banks = build_bank(enc, bank_files, mb, LAYERS, nreg, DEV, photo)
    sc = layer_scales_probe(enc, banks, stu_files[:16], mb, LAYERS, nreg, DEV, SCALE)
    print('[E2a] bank built + layer scales probed', flush=True)

    def tmap(f):
        return knn_map(enc, banks, load_img(f, SCALE), mb, LAYERS, nreg, gk, DEV, win, RES, sc)

    # ---- student training pairs ----
    Fs, Ts = [], []
    for i, f in enumerate(stu_files):
        Fs.append(feats(enc, f).half().cpu())
        Ts.append(torch.from_numpy(np.asarray(tmap(f), dtype=np.float32)))
        if i % 30 == 0:
            print(f'  teacher train-maps {i}/{len(stu_files)}', flush=True)
    Fs = torch.cat(Fs); Ts = torch.stack(Ts)                     # [N,5120,64,64], [N,256,256]
    mu, sd = Ts.mean().item(), Ts.std().item()
    print(f'[E2a] teacher stats mu={mu:.4f} sd={sd:.4f}', flush=True)

    # ---- student head: 64x64 feats -> 256x256 map (pixelshuffle x4) ----
    head = nn.Sequential(
        nn.Conv2d(5120, 512, 1), nn.GELU(),
        nn.Conv2d(512, 256, 3, padding=1), nn.GELU(),
        nn.Conv2d(256, 16, 3, padding=1), nn.PixelShuffle(4),        # 16ch @64 -> 1ch @256 (r^2=16)
    ).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    for it in range(ITERS):
        j = torch.randint(0, Fs.shape[0], (BS,))
        x = Fs[j].to(DEV).float()
        y = ((Ts[j].to(DEV) - mu) / sd).unsqueeze(1)                  # [B,1,256,256]
        loss = F.smooth_l1_loss(head(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 250 == 0:
            print(f'  iter {it}/{ITERS} loss {loss.item():.4f}', flush=True)

    # ---- eval gate on test ----
    head.eval()
    gts, tms, sms = [], [], []
    for i, f in enumerate(tg + tb):
        tms.append(np.asarray(tmap(f), dtype=np.float32))
        with torch.no_grad():
            sms.append((head(feats(enc, f).float()).squeeze() * sd + mu).cpu().numpy())
        stem = os.path.splitext(os.path.basename(f))[0]
        gp = os.path.join(DATA, CAT, 'ground_truth', 'bad', stem + '_mask.png')
        g = (np.array(Image.open(gp).convert('L').resize((RES, RES), Image.NEAREST)) > 0).astype(np.uint8) \
            if os.path.exists(gp) else np.zeros((RES, RES), np.uint8)
        gts.append(g)
        if i % 40 == 0:
            print(f'  test maps {i}/{len(tg)+len(tb)}', flush=True)
    gts = np.stack(gts); tms = np.stack(tms); sms = np.stack(sms)
    pro_t = aucpro_at(*pro_curve(gts, tms), 0.05) * 100
    pro_s = aucpro_at(*pro_curve(gts, sms), 0.05) * 100
    ratio = pro_s / max(pro_t, 1e-9)
    corr = float(np.corrcoef(tms.ravel(), sms.ravel())[0, 1])
    print(f'\n[E2a RESULT] teacher AUPRO {pro_t:.2f} | student AUPRO {pro_s:.2f} | '
          f'ratio {100*ratio:.1f}% | corr {corr:.4f}', flush=True)
    print(f'[E2a GATE {"PASS" if ratio >= 0.95 else "FAIL"}] (student >= 95% of deployment-grade teacher)', flush=True)
    json.dump({'teacher_aupro': pro_t, 'student_aupro': pro_s, 'ratio': ratio, 'corr': corr},
              open(os.path.join(_ROOT, 'e2a_result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
