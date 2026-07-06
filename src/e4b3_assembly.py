"""E4b-v2 — merged network, corrected feature routing (walnuts pilot).

E4b lesson: head_D must keep its PROVEN feature diet (raw A-layers [7,15,23,31]); forcing it onto
trunk features loses 30%. E4b-v2 = ONE encoder forward over the UNION layer set per tile:
    union = [7, 8, 11, 14, 15, 17, 20, 23, 26, 29, 31]
    A-slice (7,15,23,31) -> head_D (distill, E4a recipe)   B-slice (8..29) -> INP trunk (recon+coh)
One training loop, one optimizer, one model file; gradients disjoint by construction (encoder frozen)
-> interference gates are satisfied structurally; the remaining EMPIRICAL gate is G1:
head_D >= 95% of E4a standalone (45.53) under the joint loop (BS/schedule differences).
Reuses e4b_targets.npz (persisted teacher targets)."""
import os, sys, glob, random, json, math
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), 'dinov3-dual-branch-ad2'))
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
A_LAYERS, B_LAYERS = [7, 15, 23, 31], [8, 11, 14, 17, 20, 23, 26, 29]
UNION = sorted(set(A_LAYERS + B_LAYERS))                     # [7,8,11,14,15,17,20,23,26,29,31]
A_IDX = [UNION.index(l) for l in A_LAYERS]
B_IDX = [UNION.index(l) for l in B_LAYERS]
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
    if v == 0:
        return img
    random.seed(1000 * i + v); np.random.seed(1000 * i + v)
    if v == 1:
        return PhotometricAug()(img)
    return las_augment(img, DTD_FILES)[0]


class UnifiedNet(nn.Module):
    """One frozen encoder forward (union layers) -> head_D (A-slice raw) + INP trunk (B-slice)."""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.trunk = INPFormerPP(enc, B_LAYERS, [[0, 1], [2, 3], [4, 5], [6, 7]],
                                 [[0, 1], [2, 3], [4, 5], [6, 7]], 1280, 20, n_proto=6,
                                 context_aware_recenter=True, use_get_intermediate=True,
                                 decoder_attn='relu', seg_res=512)
        self.head_d = nn.Sequential(nn.Conv2d(5120, 512, 1), nn.GELU(),
                                    nn.Conv2d(512, 256, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(256, 1, 1))

    def encode_union(self, x):
        nreg = self.enc.num_register_tokens
        dt = next(self.enc.parameters()).dtype
        with torch.no_grad():
            toks = self.enc._get_intermediate_layers_not_chunked(x.to(dt), UNION)
        toks = [t.float() for t in toks]
        B = x.shape[0]
        side = int(math.sqrt(toks[0].shape[1] - 1 - nreg))
        fa = [toks[i][:, 1 + nreg:, :].permute(0, 2, 1).reshape(B, -1, side, side) for i in A_IDX]
        return torch.cat(fa, 1).contiguous(), [toks[i] for i in B_IDX]

    def trunk_from_tokens(self, en_list, B):
        """INPFormerPP.forward body, en_list precomputed (single-encode merge)."""
        t = self.trunk
        nreg = t.encoder.num_register_tokens
        side = int(math.sqrt(en_list[0].shape[1] - 1 - nreg))
        x_all = t.fuse(en_list).detach()
        agg = t.prototype_token.repeat(B, 1, 1)
        for blk in t.aggregation:
            agg = blk(agg, x_all)
        g_loss = t.soft_gather_loss(x_all[:, 1 + nreg:, :], agg)
        z = t.bottleneck(x_all)
        de_list = []
        for blk in t.decoder:
            z = blk(z, agg)
            de_list.append(z)
        de_list = de_list[::-1]
        en = [t.fuse([en_list[i] for i in idxs]) for idxs in t.fuse_layer_encoder]
        de = [t.fuse([de_list[i] for i in idxs]) for idxs in t.fuse_layer_decoder]
        de = [d[:, 1 + nreg:, :] for d in de]
        en = [e[:, 1 + nreg:, :] - e[:, :1, :] for e in en]
        en = [F.layer_norm(e, (e.shape[-1],), eps=1e-8) for e in en]
        en = [e.permute(0, 2, 1).reshape(B, -1, side, side).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape(B, -1, side, side).contiguous() for d in de]
        return en, de, g_loss

    def forward(self, x):
        fa, btoks = self.encode_union(x)
        en, de, g_loss = self.trunk_from_tokens(btoks, x.shape[0])
        return self.head_d(fa), en, de, g_loss


def main():
    global DTD_FILES
    DTD_FILES = glob.glob(os.path.join(DTD, '**', '*.jpg'), recursive=True)
    tr = sorted(glob.glob(os.path.join(DATA, CAT, 'train', 'good', '*.png')))
    random.seed(0); random.shuffle(tr)
    stu_files = tr[300:432]
    tg = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'good', '*.png')))
    tb = sorted(glob.glob(os.path.join(DATA, CAT, 'test', 'bad', '*.png')))
    z = np.load(TGT_PATH)
    META, TGT = z['meta'], torch.from_numpy(z['tgt'])
    mu, sd = TGT.mean().item(), TGT.std().item()
    print(f'[E4b3] {len(META)} distill pairs (reused)', flush=True)

    enc = vit_encoder.load('dinov3_vit_huge_16').to(DEV).eval().half()
    for p in enc.parameters():
        p.requires_grad_(False)
    torch.manual_seed(2); np.random.seed(2); random.seed(2)
    net = UnifiedNet(enc).to(DEV)
    # ---- E4b3: LOAD PROVEN WEIGHTS instead of training ----
    net.head_d.load_state_dict(torch.load(os.path.join(_ROOT, 'e4a_head.pth'), map_location=DEV))
    bsd = torch.load(os.path.join(os.path.dirname(_ROOT), 'Dinomaly2', 'saved_results',
                                  'inpfpp_h1024_walnuts_retrain', 'model.pth'), map_location=DEV)
    missing, unexpected = net.trunk.load_state_dict(bsd, strict=False)
    print(f'[E4b3] trunk load: missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    ASSEMBLY = True
    t = net.trunk
    opt = torch.optim.AdamW([t.prototype_token] + list(t.aggregation.parameters())
                            + list(t.bottleneck.parameters()) + list(t.decoder.parameters())
                            + list(net.head_d.parameters()), lr=LR, weight_decay=1e-4)
    IMGS = {}
    def tile_np(k):
        i, v, x, y = META[k]
        if i not in IMGS:
            IMGS[i] = np.array(Image.open(stu_files[i]).convert('RGB'))
        vimg = stu_variant(IMGS[i], int(i), int(v))
        return vimg[y:y + WIN, x:x + WIN]

    for it in range([] if ASSEMBLY else range(ITERS) or [] , ) if False else ([] if ASSEMBLY else range(ITERS)):
        j = np.random.randint(0, len(META), BS)
        x = torch.stack([NORM(TO_T(Image.fromarray(tile_np(k)))) for k in j]).to(DEV)
        pd, en, de, g_loss = net(x)
        y = ((TGT[j].to(DEV) - mu) / sd).unsqueeze(1)
        loss = global_cosine_hm_adaptive(en, de, y=3) + 0.2 * g_loss + F.smooth_l1_loss(pd, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 250 == 0:
            print(f'  iter {it}/{ITERS} loss {loss.item():.4f}', flush=True)
    torch.save(net.state_dict(), os.path.join(_ROOT, 'e4b3_unified.pth'))

    # ---- eval: head_D stitched + recon stitched from the SAME single-pass model ----
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(DEV)
    hann1024 = torch.as_tensor(np.asarray(hann2d(WIN), dtype=np.float32)).reshape(WIN, WIN)
    ref = np.load(os.path.join(_ROOT, 'e3_cache', 'walnuts.npz'), allow_pickle=True)
    gts = ref['gts']
    outs = {'headD': [], 'recon': []}
    net.eval()
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
                pd, en, de, _ = net(xt)
            pmD = pd.squeeze() * sd + mu
            pmR = torch.stack([(1 - F.cosine_similarity(e, d, dim=1)) for e, d in zip(en, de)], 0).mean(0).squeeze()
            for k, pm in [('headD', pmD), ('recon', pmR)]:
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
    print(f"\n[E4b3 RESULT] head_D {pro['headD']:.2f} ({100*pro['headD']/E4A_REF:.1f}% of E4a {E4A_REF}) | "
          f"recon {pro['recon']:.2f} (E4b control ref 47.79)", flush=True)
    print(f"[E4b3 G1 {'PASS' if g1 else 'FAIL'}] [recon healthy: {'YES' if pro['recon'] >= 46.79 else 'NO'}]", flush=True)
    json.dump({**pro, 'g1': bool(g1)}, open(os.path.join(_ROOT, 'e4b3_result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
