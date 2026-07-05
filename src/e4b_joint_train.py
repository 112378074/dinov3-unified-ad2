"""E4b — TRUE MERGE pilot (walnuts): head_D attached to the INP trunk, trained JOINTLY with
Branch-B's core trunk losses (recon + INP coherence) in ONE run. Interference is real: L_distill
backprops through the shared decoder/bottleneck/aggregation.

head_D input = trunk features concat(en 4x1280, de 4x1280) = 10240ch @ 64x64 (en detached-encoder
side, de trainable) — A's job now reads the SAME features B shapes.

GATES (walnuts):
  G1 head_D(joint, stitched) >= 95% of E4a standalone (45.53) — joint training must not break distill
  G2 recon-map AUPRO(joint) >= recon-map AUPRO(control trained WITHOUT distill) - 1.0 — distill must
     not break B (control = identical recipe minus L_distill, trained in the same script)
Teacher targets persisted (e4b_targets.npz) -> reruns skip the bank entirely."""
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
from models.inpformer_pp import INPFormerPP
from utils import global_cosine_hm_adaptive, get_gaussian_kernel
from tiled_eval import hann2d
from dataset import PhotometricAug
from synth import las_augment
from ad2_pipeline_eval import pro_curve, aucpro_at

CAT, DATA = 'walnuts', r'C:\Users\user\Desktop\datasets\ad2_dinomaly'
DTD = r'C:\Users\user\Desktop\EfficientAD\EfficientAD-main\dtd'
WIN, OV, RES = 1024, 0.2, 256
ITERS, BS, LR, N_SYN = 4000, 2, 2e-4, 80
E4A_REF = 45.53
DEV = 'cuda:0'
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TO_T = transforms.ToTensor()
TGT_PATH = os.path.join(_ROOT, 'e4b_targets.npz')


def wins_of(W, H):
    def pos(L):
        step = max(int(WIN * (1 - OV)), 1)
        ps = list(range(0, max(L - WIN, 0) + 1, step))
        if ps[-1] != L - WIN:
            ps.append(max(L - WIN, 0))
        return ps
    return [(x, y) for y in pos(H) for x in pos(W)]


def stu_variant(img, i, v):
    """deterministic variant v of image i: 0=orig, 1=photometric, 2=LAS synthetic (i<N_SYN)."""
    if v == 0:
        return img
    random.seed(1000 * i + v); np.random.seed(1000 * i + v)
    if v == 1:
        return PhotometricAug()(img)
    return las_augment(img, DTD_FILES)[0]


def build_targets(stu_files):
    from membank_derisk import build_bank, knn_map
    from push_final import layer_scales_probe
    enc = vit_encoder.load('dinov3_vit_huge_16').to(DEV).eval().half()
    for p in enc.parameters():
        p.requires_grad_(False)
    nreg = getattr(enc, 'num_register_tokens', getattr(enc, 'n_storage_tokens', 4))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEV)
    hann576 = hann2d(576)
    tr = sorted(glob.glob(os.path.join(DATA, CAT, 'train', 'good', '*.png')))
    random.seed(0); random.shuffle(tr)
    mb = SimpleNamespace(scale=0.625, crop_size=576, overlap=128, metric='l2', keep_frac=0.3,
                         max_bank=100000, subsample='coreset', aug_every=2, chunk=30000,
                         resize=RES, knn=1, faithful=True, merge='hann')
    banks = build_bank(enc, tr[:300], mb, [7, 15, 23, 31], nreg, DEV, PhotometricAug())
    sc = layer_scales_probe(enc, banks, stu_files[:16], mb, [7, 15, 23, 31], nreg, DEV, 0.625)
    print('[E4b] teacher bank ready', flush=True)
    metas, tgts = [], []
    for i, f in enumerate(stu_files):
        img = np.array(Image.open(f).convert('RGB'))
        H0, W0 = img.shape[:2]
        for v in range(3 if i < N_SYN else 2):
            vimg = stu_variant(img, i, v)
            for (x, y) in wins_of(W0, H0):
                til = vimg[y:y + WIN, x:x + WIN]
                if til.shape[:2] != (WIN, WIN):
                    continue
                t = TO_T(Image.fromarray(til).resize((int(WIN * .625),) * 2))
                m = np.asarray(knn_map(enc, banks, t, mb, [7, 15, 23, 31], nreg, gk, DEV,
                                       hann576, RES, sc), dtype=np.float32)
                mt = F.interpolate(torch.from_numpy(m)[None, None], size=64,
                                   mode='bilinear', align_corners=False).squeeze().numpy()
                metas.append([i, v, x, y]); tgts.append(mt)
        if i % 20 == 0:
            print(f'  targets {i}/{len(stu_files)} n={len(tgts)}', flush=True)
    np.savez_compressed(TGT_PATH, meta=np.array(metas, np.int32), tgt=np.stack(tgts))
    del banks, enc
    torch.cuda.empty_cache()


def build_model():
    enc = vit_encoder.load('dinov3_vit_huge_16').to(DEV).eval().half()
    for p in enc.parameters():
        p.requires_grad_(False)
    tl = [8, 11, 14, 17, 20, 23, 26, 29]
    m = INPFormerPP(enc, tl, [[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]],
                    1280, 20, n_proto=6, context_aware_recenter=True, use_get_intermediate=True,
                    decoder_attn='relu', seg_res=512).to(DEV)
    head = nn.Sequential(nn.Conv2d(10240, 512, 1), nn.GELU(),
                         nn.Conv2d(512, 256, 3, padding=1), nn.GELU(),
                         nn.Conv2d(256, 1, 1)).to(DEV)
    return m, head


def trunk_feats(m, x):
    en, de, g_loss, _ = m(x)
    f = torch.cat(en + de, 1)                                   # [B,10240,64,64]
    return en, de, g_loss, f


def recon_tile_map(en, de):
    ms = [(1 - F.cosine_similarity(e, d, dim=1)) for e, d in zip(en, de)]   # [B,64,64] each
    return torch.stack(ms, 0).mean(0)


def train_one(tag, distill, TILREF, TGT, mu, sd):
    torch.manual_seed(2); np.random.seed(2); random.seed(2)
    m, head = build_model()
    params = ([m.prototype_token] + list(m.aggregation.parameters()) + list(m.bottleneck.parameters())
              + list(m.decoder.parameters()) + (list(head.parameters()) if distill else []))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    n = len(TILREF)
    for it in range(ITERS):
        j = np.random.randint(0, n, BS)
        x = torch.stack([NORM(TO_T(TILREF[k]())) for k in j]).to(DEV)
        en, de, g_loss, f = trunk_feats(m, x)
        loss = global_cosine_hm_adaptive(en, de, y=3) + 0.2 * g_loss
        if distill:
            y = ((TGT[j].to(DEV) - mu) / sd).unsqueeze(1)
            loss = loss + F.smooth_l1_loss(head(f), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 250 == 0:
            print(f'  [{tag}] iter {it}/{ITERS} loss {loss.item():.4f}', flush=True)
    torch.save({'model': m.state_dict(), 'head': head.state_dict()},
               os.path.join(_ROOT, f'e4b_{tag}.pth'))
    return m, head


def main():
    global DTD_FILES
    DTD_FILES = glob.glob(os.path.join(DTD, '**', '*.jpg'), recursive=True)
    tr = sorted(glob.glob(os.path.join(DATA, CAT, 'train', 'good', '*.png')))
    random.seed(0); random.shuffle(tr)
    stu_files = tr[300:432]
    tg = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'good', '*.png')))
    tb = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'bad', '*.png')))

    if not os.path.exists(TGT_PATH):
        build_targets(stu_files)
    z = np.load(TGT_PATH)
    META, TGT = z['meta'], torch.from_numpy(z['tgt'])
    mu, sd = TGT.mean().item(), TGT.std().item()
    print(f'[E4b] {len(META)} distill pairs  mu={mu:.3f} sd={sd:.3f}', flush=True)

    IMGS = {}
    def tile_fn(k):
        i, v, x, y = META[k]
        if i not in IMGS:
            IMGS[i] = np.array(Image.open(stu_files[i]).convert('RGB'))
        vimg = stu_variant(IMGS[i], int(i), int(v))
        return Image.fromarray(vimg[y:y + WIN, x:x + WIN])
    TILREF = [lambda k=k: tile_fn(k) for k in range(len(META))]

    mJ, headJ = train_one('joint', True, TILREF, TGT, mu, sd)
    mC, _ = train_one('control', False, TILREF, TGT, mu, sd)

    # ---- eval: stitched head_D(joint) + recon maps (joint vs control) ----
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEV)
    hann1024 = torch.as_tensor(np.asarray(hann2d(WIN), dtype=np.float32)).reshape(WIN, WIN)
    ref = np.load(os.path.join(_ROOT, 'e3_cache', 'walnuts.npz'), allow_pickle=True)
    gts = ref['gts']
    outs = {'headD': [], 'reconJ': [], 'reconC': []}
    for i, f in enumerate(tg + tb):
        img = np.array(Image.open(f).convert('RGB'))
        H0, W0 = img.shape[:2]
        acc = {k: torch.zeros(H0, W0) for k in outs}
        wsum = torch.zeros(H0, W0)
        for (x, y) in wins_of(W0, H0):
            til = img[y:y + WIN, x:x + WIN]
            if til.shape[:2] != (WIN, WIN):
                continue
            xt = NORM(TO_T(Image.fromarray(til))).unsqueeze(0).to(DEV)
            with torch.no_grad():
                enJ, deJ, _, fJ = trunk_feats(mJ, xt)
                pmD = (headJ(fJ).squeeze() * sd + mu)
                pmRJ = recon_tile_map(enJ, deJ).squeeze()
                enC, deC, _, _ = trunk_feats(mC, xt)
                pmRC = recon_tile_map(enC, deC).squeeze()
            for k, pm in [('headD', pmD), ('reconJ', pmRJ), ('reconC', pmRC)]:
                up = F.interpolate(pm[None, None].float(), size=WIN, mode='bilinear',
                                   align_corners=False).squeeze().cpu()
                acc[k][y:y + WIN, x:x + WIN] += up * hann1024
            wsum[y:y + WIN, x:x + WIN] += hann1024
        for k in outs:
            sm = F.interpolate((acc[k] / wsum.clamp_min(1e-6))[None, None], size=RES,
                               mode='bilinear', align_corners=False).to(DEV)
            outs[k].append(gk(sm).squeeze().cpu().numpy())
        if i % 40 == 0:
            print(f'  eval {i}/{len(tg)+len(tb)}', flush=True)
    pro = {k: aucpro_at(*pro_curve(gts, np.stack(v)), 0.05) * 100 for k, v in outs.items()}
    g1 = pro['headD'] >= 0.95 * E4A_REF
    g2 = pro['reconJ'] >= pro['reconC'] - 1.0
    print(f"\n[E4b RESULT] head_D(joint) {pro['headD']:.2f} (G1 ref {E4A_REF}, {100*pro['headD']/E4A_REF:.1f}%) | "
          f"recon joint {pro['reconJ']:.2f} vs control {pro['reconC']:.2f}", flush=True)
    print(f"[E4b G1 {'PASS' if g1 else 'FAIL'}] [G2 {'PASS' if g2 else 'FAIL'}]", flush=True)
    json.dump({**pro, 'g1': bool(g1), 'g2': bool(g2)}, open(os.path.join(_ROOT, 'e4b_result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
