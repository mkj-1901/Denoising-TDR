#!/usr/bin/env python3
"""
Ambient‑Mean denoising demo (CPU‑only).
Runs the same workflow as ``demo.py`` but replaces the Gaussian stand‑in
with the ``ambient_mean_denoise`` implementation.
"""
# Ensure the project root (one level up) is on the Python path so we can import ambient_mean_denoise
import sys, os, time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from restoration import restore
from ambient_mean_denoise import ambient_mean_denoise

repo = sys.argv[1] if len(sys.argv) > 1 else "Denoising-TDR"
t0 = time.time()
line = "-" * 66

def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))

# ---------------------------------------------------------------- 1. HMI
print(line); print("1) SDO/HMI magnetogram (real data, 1024 x 1024)"); print(line)
y = np.load(os.path.join(repo, "data/denoisingHMI/hmilos0009.npy")).astype(float)
# ambient‑mean denoiser replaces the Gaussian filter
xs = ambient_mean_denoise(y, window=5, sigma_thresh=2.0)
for R in (10, 5):
    x, A, rho, viol = restore(xs, y, R, rule="init")
    lc0 = np.polyfit(y.ravel(), xs.ravel(), 1)[0]
    lc = np.polyfit(y.ravel(), x.ravel(), 1)[0]
    print(f"   TDR-Init R={R:2d} G: rho_R = {100*rho:5.1f} %   LC {lc0:4.2f} -> {lc:4.2f}   bound violations = {100*viol:.1f} %")

# ---------------------------------------------------------------- 2. synthetic
print(line); print("2) Synthetic benchmark with ground truth (519 x 519)"); print(line)
c = np.load(os.path.join(repo, "data/clean/datainte24.npy")).astype(float)[0]
c = (c - c.min()) / (c.max() - c.min())
cases = {"salt & pepper": "data/denoising_sp1all/datasp24.npy",
         "Gaussian 224": "data/denoising_gaussian005/datagaussian24.npy"}
rows = []
for name, f in cases.items():
    y = np.load(os.path.join(repo, f)).astype(float)
    sigma = np.std(y - c)
    R = sigma
    xs = ambient_mean_denoise(y, window=5, sigma_thresh=2.0)
    out = {"noisy y": y, "x* (R=inf)": xs}
    viols = {"noisy y": 0, "x* (R=inf)": None}
    for rule, lab in [("init", "TDR-Init"), ("obs", "Ambient-Obs"), ("clean", "Ambient-Clean")]:
        x, A, rho, viol = restore(xs, y, R, rule=rule)
        out[lab] = x; viols[lab] = viol
    print(f"   {name}:  R = sigma = {R:.3f},  rho_R = {100*rho:.1f} %")
    for lab, img in out.items():
        v = viols[lab]
        print(f"      {lab:14s} true PSNR = {psnr(img, c):5.2f} dB" + ("" if v is None else f"   bound violations = {100*v:4.1f} %"))
    rows.append((name, out))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 6, figsize=(13, 5.0))
for r, (name, out) in enumerate(rows):
    panels = [("clean x", c)] + list(out.items())
    for k, (lab, img) in enumerate(panels):
        a = ax[r, k]
        a.imshow(img[180:340, 180:340], cmap="gray", vmin=0, vmax=1)
        title = lab if lab == "clean x" else f"{lab}\n{psnr(img, c):.1f} dB"
        a.set_title(title, fontsize=10)
        a.axis("off")
    ax[r, 0].text(-0.15, 0.5, name, transform=ax[r, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=11)
fig.suptitle("Ambient‑Mean denoising demo (stand‑in denoiser; true PSNR vs clean image)", fontsize=12)
fig.tight_layout()
fig.savefig("ambient_demo_result.png", dpi=170)
print(line); print(f"Saved ambient_demo_result.png   (total time {time.time()-t0:.1f} s)")
