"""E4a — GEOMETRY UNIFICATION gate (walnuts pilot) for the A+B merged network.

The merge (one trunk, one training) requires head_D (A's distilled distance) to live at Branch B's
tile geometry (1024-native windows, /16 -> 64x64 tokens) instead of A's 576@0.625. This is the real
new risk: the teacher (strong bank, 576@0.625, Hann) is resampled into 1024-tile targets.
Data recipe = proven E2d/E3c: disjoint bank/student split + photometric copies + LAS synthetic tiles.
GATE: full-image stitched head_D AUPRO >= 95% of the E3c 576-geometry student (46.84); teacher ref 41.87.
Pass -> E4b: attach head_D to the INP trunk and train jointly with B's four losses (interference gate)."""
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
from membank_derisk import build_bank, knn_map
from push_final import layer_scales_probe
from tiled_eval import hann2d
from dataset import PhotometricAug
from synth import las_augment
from ad2_pipeline_eval import pro_curve, aucpro_at
from utils import get_gaussian_kernel

CAT, DATA = 'walnuts', r'C:\Users\user\Desktop\datasets\ad2_dinomaly'
DTD = r'C:\Users\user\Desktop\EfficientAD\EfficientAD-main\dtd'
A_LAYERS, A_SCALE, RES = [7, 15, 23, 31], 0.625, 256
WIN, OV = 1024, 0.2                     # B geometry: 1024-native windows, 20% overlap
ITERS, BS, LR, N_SYN = 6000, 8, 2e-4, 80
DEV = 'cuda:0'
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TO_T = transforms.ToTensor()
torch.manual_seed(0); np.random.seed(0); random.seed(0)


def wins_of(W, H):
    def pos(L):
        step = max(int(WIN * (1 - OV)), 1)
        ps = list(range(0, max(L - WIN, 0) + 1, step))
        if ps[-1] != L - WIN:
            ps.append(max(L - WIN, 0))
        return ps
    return [(x, y) for y in pos(H) for x in pos(W)]


@torch.no_grad()
def tile_feats(enc, til):                                     # 1024 uint8 tile -> [1,5120,64,64]
    t = NORM(torch.from_numpy(np.ascontiguousarray(til)).permute(2, 0, 1).float() / 255)
    out = enc.get_intermediate_layers(t.unsqueeze(0).to(DEV).half(), n=A_LAYERS, reshape=True)
    return torch.cat(out, 1).float()


def main():
    enc = vit_encoder.load('dinov3_vit_huge_16').to(DEV).eval().half()
    for p in enc.parameters():
        p.requires_grad_(False)
    nreg = getattr(enc, 'num_register_tokens', getattr(enc, 'n_storage_tokens', 4))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEV)
    hann576 = hann2d(576)
    photo = PhotometricAug()
    dtd = glob.glob(os.path.join(DTD, '**', '*.jpg'), recursive=True)
    tr = sorted(glob.glob(os.path.join(DATA, CAT, 'train', 'good', '*.png'))); random.shuffle(tr)
    bank_files, stu_files = tr[:300], tr[300:432]
    tg = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'good', '*.png')))
    tb = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'bad', '*.png')))
    mb = SimpleNamespace(scale=A_SCALE, crop_size=576, overlap=128, metric='l2', keep_frac=0.3,
                         max_bank=100000, subsample='coreset', aug_every=2, chunk=30000,
                         resize=RES, knn=1, faithful=True, merge='hann')
    banks = build_bank(enc, bank_files, mb, A_LAYERS, nreg, DEV, photo)
    sc = layer_scales_probe(enc, banks, stu_files[:16], mb, A_LAYERS, nreg, DEV, A_SCALE)
    print('[E4a] teacher bank ready', flush=True)

    def teacher_tile_target(til_np):
        """teacher distance for ONE 1024-native tile at B geometry -> [64,64]:
        scale tile by 0.625 (=640, teacher's operating scale), knn_map (2x2 576-tiles inside),
        map back at RES then resample to 64x64 token grid."""
        pil = Image.fromarray(til_np)
        t = TO_T(pil.resize((int(WIN * A_SCALE), int(WIN * A_SCALE))))
        m = np.asarray(knn_map(enc, banks, t, mb, A_LAYERS, nreg, gk, DEV, hann576, RES, sc),
                       dtype=np.float32)                       # [256,256] for this tile
        mt = torch.from_numpy(m)[None, None]
        return F.interpolate(mt, size=64, mode='bilinear', align_corners=False).squeeze()

    def img_tiles(img_np):
        H, W = img_np.shape[:2]
        out = []
        for (x, y) in wins_of(W, H):
            til = img_np[y:y + WIN, x:x + WIN]
            if til.shape[:2] == (WIN, WIN):
                out.append((til, x, y))
        return out, H, W

    # ---- distillation pairs at B geometry: normal + photometric copy + synthetic ----
    TIL, TGT = [], []
    for i, f in enumerate(stu_files):
        img = np.array(Image.open(f).convert('RGB'))
        variants = [img, photo(img)]
        if i < N_SYN:
            variants.append(las_augment(img, dtd)[0])
        for v in variants:
            tiles_v, _, _ = img_tiles(v)
            for til, x, y in tiles_v:
                TIL.append(til.copy()); TGT.append(teacher_tile_target(til))
        if i % 20 == 0:
            print(f'  distill imgs {i}/{len(stu_files)} tiles={len(TIL)}', flush=True)
    TGT = torch.stack(TGT); mu, sd = TGT.mean().item(), TGT.std().item()
    print(f'[E4a] {len(TIL)} tiles at 1024-geometry  mu={mu:.3f} sd={sd:.3f}', flush=True)

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
        if it % 300 == 0:
            print(f'  iter {it}/{ITERS} loss {loss.item():.4f}', flush=True)
    torch.save(head.state_dict(), os.path.join(_ROOT, 'e4a_head.pth'))

    # ---- eval: full-image stitched head_D @1024-geometry vs cached E3c teacher/student ----
    head.eval()
    hann1024 = torch.as_tensor(np.asarray(hann2d(WIN), dtype=np.float32)).reshape(WIN, WIN)
    ref = np.load(os.path.join(_ROOT, 'e3_cache', 'walnuts.npz'), allow_pickle=True)   # E3c teacher maps + gts
    gts, tms = ref['gts'], ref['tms']
    sms = []
    for i, f in enumerate(tg + tb):
        img = np.array(Image.open(f).convert('RGB'))
        tiles, H, W = img_tiles(img)
        acc = torch.zeros(H, W); wsum = torch.zeros(H, W)
        for til, x, y in tiles:
            with torch.no_grad():
                pm = head(tile_feats(enc, til)).squeeze() * sd + mu           # [64,64]
            pm = F.interpolate(pm[None, None], size=WIN, mode='bilinear', align_corners=False).squeeze().cpu()
            acc[y:y + WIN, x:x + WIN] += pm * hann1024; wsum[y:y + WIN, x:x + WIN] += hann1024
        sm = F.interpolate((acc / wsum.clamp_min(1e-6))[None, None], size=RES,
                           mode='bilinear', align_corners=False).to(DEV)
        sms.append(gk(sm).squeeze().cpu().numpy())
        if i % 40 == 0:
            print(f'  eval {i}/{len(tg)+len(tb)}', flush=True)
    sms = np.stack(sms)
    pro_t = aucpro_at(*pro_curve(gts, tms), 0.05) * 100
    pro_s = aucpro_at(*pro_curve(gts, sms), 0.05) * 100
    e3c_student = 46.84
    print(f'\n[E4a RESULT] teacher(576geom) {pro_t:.2f} | head_D@1024geom {pro_s:.2f} | '
          f'vs-teacher {100*pro_s/max(pro_t,1e-9):.1f}% | vs-E3c-student(46.84) {100*pro_s/e3c_student:.1f}%', flush=True)
    print(f'[E4a GATE {"PASS" if pro_s >= 0.95 * e3c_student else "FAIL"}] (>=95% of the 576-geometry student)', flush=True)
    json.dump({'teacher': pro_t, 'head_d_1024geom': pro_s, 'vs_e3c_student': pro_s / e3c_student},
              open(os.path.join(_ROOT, 'e4a_result.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(_ROOT, 'e4a_maps.npz'), sms=sms, gts=gts)


if __name__ == '__main__':
    main()
