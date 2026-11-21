#!/usr/bin/env python3
"""
TDR restoration demo (runs in a few seconds on a laptop CPU).

    python3 demo.py /path/to/Denoising-TDR

What it shows, on the REAL data released with Liu et al. (2025):
  1. HMI magnetogram: noise level of the observation and how many pixels each
     threshold R flags as abnormal.
  2. Synthetic benchmark with ground truth (salt-and-pepper and Gaussian sigma=224):
     the four restoration rules side by side, with true PSNR and bound violations.

NOTE: the primary denoiser x* here is a fast STAND-IN (Gaussian filter), not the
trained Self2Self U-Net, so that the demo runs live without TensorFlow/GPU.
The restoration step (restoration.py) is exactly the one used in the report.
"""
import sys, os, time
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from restoration import restore

repo = sys.argv[1] if len(sys.argv) > 1 else "Denoising-TDR"
t0 = time.time()
line = "-" * 66

def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))

def gauss_sigma(v, lo=-20, hi=20, step=0.5):
    h, e = np.histogram(v, bins=np.arange(lo, hi + step / 2, step))
    x = 0.5 * (e[1:] + e[:-1]); h = h / v.size
    g = lambda t, a, m, s: a * np.exp(-(t - m) ** 2 / (2 * s * s))
    p, _ = curve_fit(g, x, h, p0=[h.max(), 0, 8])
    return p[1], abs(p[2])

# ---------------------------------------------------------------- 1. HMI
print(line); print("1) SDO/HMI magnetogram (real data, 1024 x 1024)"); print(line)
y = np.load(os.path.join(repo, "data/denoisingHMI/hmilos0009.npy")).astype(float)
mu, sig = gauss_sigma(y.ravel())
print(f"   observation: Gaussian-fit noise  mu = {mu:5.2f} G   sigma = {sig:5.2f} G  (paper: 8.09 G)")
xs = gaussian_filter(y, 1.2)                       # stand-in primary denoiser
_, sig_s = gauss_sigma(xs.ravel())
print(f"   stand-in x*: sigma = {sig_s:5.2f} G")
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
    xs = gaussian_filter(y, 1.5)
    out = {"noisy y": y, "x* (R=inf)": xs}
    viols = {"noisy y": 0, "x* (R=inf)": None}
    for rule, lab in [("init", "TDR-Init"), ("obs", "Ambient-Obs"), ("clean", "Ambient-Clean")]:
        x, A, rho, viol = restore(xs, y, R, rule=rule)
        out[lab] = x; viols[lab] = viol
    print(f"   {name}:  R = sigma = {R:.3f},  rho_R = {100*rho:.1f} %")
    for lab, x in out.items():
        v = viols[lab]
        print(f"      {lab:14s} true PSNR = {psnr(x, c):5.2f} dB" + ("" if v is None else f"   bound violations = {100*v:4.1f} %"))
    rows.append((name, out))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 6, figsize=(13, 5.0))
for r, (name, out) in enumerate(rows):
    panels = [("clean x", c)] + list(out.items())
    for k, (lab, img) in enumerate(panels):
        a = ax[r, k]; a.imshow(img[180:340, 180:340], cmap="gray", vmin=0, vmax=1)
        t = lab if lab == "clean x" else f"{lab}\n{psnr(img, c):.1f} dB"
        a.set_title(t, fontsize=10); a.axis("off")
    ax[r, 0].text(-0.15, 0.5, name, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=11)
fig.suptitle("TDR restoration rules on real benchmark data (stand-in denoiser; true PSNR vs clean image)", fontsize=12)
fig.tight_layout()
fig.savefig("demo_result.png", dpi=170)
print(line); print(f"Saved demo_result.png   (total time {time.time()-t0:.1f} s)")
