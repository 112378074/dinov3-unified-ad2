"""E3a — 8-category bank-distillation training + gate. Trains one distillation head per category
(E2d recipe, base scale 0.625) into e3_cache/, then assembles the teacher-vs-student comparison:
per-cat AUPRO of the bank-free student map vs the kNN teacher, plus a v3-style binary SegF1. Gate:
MEAN student AUPRO >= 95% of MEAN teacher (bank fully replaceable by weights across all cats).
Detached-safe; skip-if-done via e3_cache/<cat>.npz."""
import subprocess, os, sys, datetime, glob
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), 'dinov3-dual-branch-ad2'))
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), 'dinov3-dual-branch-ad2', 'superadd_ref'))
PY = r'C:\Users\user\anaconda3\envs\dinov3ad2\python.exe'
LOG = r'C:\Users\user\Desktop\dinov3-ad2-relational\logs\5090'
CACHE = os.path.join(_ROOT, 'e3b_cache')
CATS = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
env = dict(os.environ, PYTHONIOENCODING='utf-8')

# ---- Phase 1: train per-cat distillation heads ----
for cat in CATS:
    if os.path.exists(os.path.join(CACHE, f'{cat}.npz')):
        print(f'=== SKIP {cat} (done) ===', flush=True); continue
    print(f'=== E3 TRAIN {cat} {datetime.datetime.now():%m-%d %H:%M} ===', flush=True)
    with open(os.path.join(LOG, f'e3_{cat}.log'), 'w') as f:
        rc = subprocess.run([PY, '-u', os.path.join(_ROOT, 'src', 'e3b_distill_cat.py'), cat],
                            cwd=_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f'    {cat} rc={rc} {datetime.datetime.now():%m-%d %H:%M}', flush=True)

# ---- Phase 2: assemble teacher-vs-student gate ----
from ad2_pipeline_eval import pro_curve, aucpro_at, cc_clean, f1_from_pred
from post_process import multi_oriented_closing, erosion_on_binary_maps, fill_closed_regions


def binseg(maps, gts, gain=1.4):
    hp = np.percentile(maps.reshape(len(maps), -1), 95, axis=1)  # crude per-set 95pct proxy
    thr = float(np.percentile(maps, 95)) * gain
    def sc(m):
        c = fill_closed_regions(multi_oriented_closing(m, threshold=thr, radius=19, n_angles=16,
                                                       lower_factor=0.8, padding=True))
        return (erosion_on_binary_maps(c, 1) > 0).astype(np.uint8)
    b = cc_clean(np.stack([sc(m) for m in maps]).astype(np.uint8)).astype(np.uint8)
    return f1_from_pred(b, gts) * 100


print(f'\n{"cat":12s} | teach_PRO stud_PRO ratio | teach_Seg stud_Seg', flush=True)
Rt, Rs, St, Ss = [], [], [], []
for cat in CATS:
    p = os.path.join(CACHE, f'{cat}.npz')
    if not os.path.exists(p):
        print(f'{cat:12s} | MISSING'); continue
    d = np.load(p, allow_pickle=True)
    tms, sms, gts = d['tms'], d['sms'], d['gts']
    pt = aucpro_at(*pro_curve(gts, tms), 0.05) * 100
    ps = aucpro_at(*pro_curve(gts, sms), 0.05) * 100
    st = binseg(tms, gts); ss = binseg(sms, gts)
    if cat != 'can':                          # can's base teacher (3.08) is noise — report only
        Rt.append(pt); Rs.append(ps); St.append(st); Ss.append(ss)
    print(f'{cat:12s} | {pt:8.2f} {ps:8.2f} {100*ps/max(pt,1e-9):5.0f}% | {st:8.2f} {ss:8.2f}', flush=True)
mt, ms = np.mean(Rt), np.mean(Rs)
print('-' * 60, flush=True)
print(f'MEAN teacher AUPRO {mt:.2f} | student AUPRO {ms:.2f} | ratio {100*ms/max(mt,1e-9):.1f}%  '
      f'SegF1 teacher {np.mean(St):.2f} student {np.mean(Ss):.2f}', flush=True)
print(f'[E3b GATE (7cats excl can) {"PASS" if ms >= 0.95*mt else "FAIL"}] (bank-free student MEAN >= 95% teacher)', flush=True)
import json
json.dump({'teacher_aupro': mt, 'student_aupro': ms, 'ratio': ms/max(mt, 1e-9),
           'per_cat_teacher': dict(zip(CATS, Rt)), 'per_cat_student': dict(zip(CATS, Rs))},
          open(os.path.join(_ROOT, 'e3b_result.json'), 'w'), indent=1)
print(f'=== E3B_DONE {datetime.datetime.now():%m-%d %H:%M} ===', flush=True)
