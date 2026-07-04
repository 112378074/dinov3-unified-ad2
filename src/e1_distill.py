"""E1 — bank-distillation feasibility (walnuts).

Question: can a lightweight conv head (student) on frozen DINOv3 features reproduce the memory
bank's kNN distance geometry (teacher) well enough to carry the metrics? GATE: student test AUPRO
>= 95% of teacher's. If it fails, the unified single-training design (PROPOSAL.md §3) is dead on
arrival; if it passes -> E2 (prototype ablation).

Setup (token-space PatchCore, geometry-aligned for clean distillation):
  * encode at resize 1024 -> /16 -> 64x64 tokens; features = concat DINOv3 layers [7,15,23,31] (5120d)
  * teacher: bank = random-subsampled token vectors from 200 train/good; distance = min-L2 per token
  * student: 3-layer conv head on the same features -> 64x64 distance map; SmoothL1 to teacher
  * eval: walnuts test (all), maps -> 256 + gaussian smooth -> AUPRO@5%FPR teacher vs student
Run:  python src/e1_distill.py            (GPU, ~1-2h; logs to stdout)
"""
import os, sys, glob, random, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SIB = os.path.join(os.path.dirname(_ROOT), 'dinov3-dual-branch-ad2')
sys.path.insert(0, _SIB)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import vit_encoder                       # sibling proven loader
from ad2_pipeline_eval import pro_curve, aucpro_at
from utils import get_gaussian_kernel

CAT = 'walnuts'
DATA = r'C:\Users\user\Desktop\datasets\ad2_dinomaly'
LAYERS = [7, 15, 23, 31]
RES, GRID = 1024, 64
BANK_IMGS, TRAIN_IMGS, BANK_MAX = 200, 150, 60000
ITERS, BS, LR = 1500, 8, 2e-4
DEV = 'cuda:0'
TT = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

torch.manual_seed(0); np.random.seed(0); random.seed(0)


@torch.no_grad()
def feats(enc, path):
    img = TT(Image.open(path).convert('RGB').resize((RES, RES))).unsqueeze(0).to(DEV).half()
    out = enc.get_intermediate_layers(img, n=LAYERS, reshape=True)   # list of [1,C,64,64]
    return torch.cat(out, 1).float()                                  # [1,5120,64,64]


@torch.no_grad()
def teacher_map(f, bank, chunk=2048):
    q = f.flatten(2).squeeze(0).T.contiguous()                        # [4096,5120]
    mins = []
    for i in range(0, q.shape[0], chunk):
        d = torch.cdist(q[i:i + chunk], bank)                         # [c,Nb]
        mins.append(d.min(1).values)
    return torch.cat(mins).reshape(GRID, GRID)                        # [64,64]


def main():
    enc = vit_encoder.load('dinov3_vit_huge_16').to(DEV).eval().half()
    for p in enc.parameters():
        p.requires_grad_(False)

    tr = sorted(glob.glob(os.path.join(DATA, CAT, 'train', 'good', '*.png')))
    random.shuffle(tr)
    bank_files, stu_files = tr[:BANK_IMGS], tr[BANK_IMGS:BANK_IMGS + TRAIN_IMGS]
    tg = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'good', '*.png')))
    tb = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'bad', '*.png')))
    print(f'[E1] bank {len(bank_files)} | student-train {len(stu_files)} | test {len(tg)}+{len(tb)}', flush=True)

    # ---- teacher bank (token PatchCore, random subsample) ----
    vecs = []
    for i, f in enumerate(bank_files):
        vecs.append(feats(enc, f).flatten(2).squeeze(0).T.cpu())
        if i % 50 == 0:
            print(f'  bank encode {i}/{len(bank_files)}', flush=True)
    bank = torch.cat(vecs)                                             # [200*4096,5120]
    idx = torch.randperm(bank.shape[0])[:BANK_MAX]
    bank = bank[idx].to(DEV)
    print(f'[E1] bank {tuple(bank.shape)}', flush=True)

    # ---- precompute student training pairs (features + teacher maps) ----
    Fs, Ts = [], []
    for i, f in enumerate(stu_files):
        ff = feats(enc, f)
        Fs.append(ff.half().cpu()); Ts.append(teacher_map(ff, bank).cpu())
        if i % 50 == 0:
            print(f'  teacher maps {i}/{len(stu_files)}', flush=True)
    Fs = torch.cat(Fs); Ts = torch.stack(Ts)
    mu, sd = Ts.mean().item(), Ts.std().item()
    print(f'[E1] teacher stats mu={mu:.3f} sd={sd:.3f}', flush=True)

    # ---- student head ----
    head = nn.Sequential(
        nn.Conv2d(5120, 512, 1), nn.GELU(),
        nn.Conv2d(512, 256, 3, padding=1), nn.GELU(),
        nn.Conv2d(256, 1, 1)).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    for it in range(ITERS):
        j = torch.randint(0, Fs.shape[0], (BS,))
        x = Fs[j].to(DEV).float()
        y = ((Ts[j].to(DEV) - mu) / sd).unsqueeze(1)
        loss = F.smooth_l1_loss(head(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0:
            print(f'  iter {it}/{ITERS} loss {loss.item():.4f}', flush=True)

    # ---- eval: teacher vs student AUPRO on test ----
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEV)
    head.eval()
    gts, tmaps, smaps = [], [], []
    for f in tg + tb:
        ff = feats(enc, f)
        tm = teacher_map(ff, bank)
        with torch.no_grad():
            sm = head(ff.float()).squeeze() * sd + mu
        def up(m):
            m = m.reshape(1, 1, GRID, GRID)
            m = F.interpolate(m, size=256, mode='bilinear', align_corners=False)
            return gk(m).squeeze().cpu().numpy()
        tmaps.append(up(tm)); smaps.append(up(sm))
        stem = os.path.splitext(os.path.basename(f))[0]
        gp = os.path.join(DATA, CAT, 'ground_truth', 'bad', stem + '_mask.png')
        g = (np.array(Image.open(gp).convert('L').resize((256, 256), Image.NEAREST)) > 0).astype(np.uint8) \
            if os.path.exists(gp) else np.zeros((256, 256), np.uint8)
        gts.append(g)
    gts = np.stack(gts); tmaps = np.stack(tmaps); smaps = np.stack(smaps)
    pro_t = aucpro_at(*pro_curve(gts, tmaps), 0.05) * 100
    pro_s = aucpro_at(*pro_curve(gts, smaps), 0.05) * 100
    ratio = pro_s / max(pro_t, 1e-9)
    corr = float(np.corrcoef(tmaps.ravel(), smaps.ravel())[0, 1])
    print(f'\n[E1 RESULT] teacher AUPRO {pro_t:.2f} | student AUPRO {pro_s:.2f} | '
          f'ratio {100*ratio:.1f}% | map corr {corr:.4f}', flush=True)
    print(f'[E1 GATE {"PASS" if ratio >= 0.95 else "FAIL"}] (gate: student >= 95% of teacher)', flush=True)
    json.dump({'teacher_aupro': pro_t, 'student_aupro': pro_s, 'ratio': ratio, 'corr': corr},
              open(os.path.join(_ROOT, 'e1_result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
