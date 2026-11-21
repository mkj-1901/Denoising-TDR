"""Configurable TDR restoration (post-processing of a saved Self2Self output x*)."""
import numpy as np
from scipy.ndimage import uniform_filter


def neighbour_mean(values, mask, k):
    """Mean of `values` over the k x k window (centre excluded), using only pixels where mask==1."""
    v = values * mask
    s = uniform_filter(v, size=k, mode="reflect") * k * k - v          # window sum minus centre
    c = uniform_filter(mask.astype(float), size=k, mode="reflect") * k * k - mask
    return s, np.rint(c)


def restore(x_star, y, R, rule="init", k=3):
    """Return (x_tdr, abnormal_mask, rho_R, bound_violation_fraction).

    rule = "init"  : x(i) <- y(i)                                  (Standard TDR)
           "obs"   : x(i) <- mean of y over N_k(i)                  (TDR-Ambient-Obs)
           "clean" : x(i) <- mean of x* over N_3(i) minus abnormal  (TDR-Ambient-Clean)
    """
    x_star = np.asarray(x_star, float); y = np.asarray(y, float)
    abnormal = np.abs(x_star - y) > R
    x = x_star.copy()
    if rule == "init":
        x[abnormal] = y[abnormal]
    elif rule == "obs":
        s, c = neighbour_mean(y, np.ones_like(y), k)
        x[abnormal] = (s / c)[abnormal]
    elif rule == "clean":
        ok = (~abnormal).astype(float)
        s, c = neighbour_mean(x_star, ok, 3)
        fill = np.where(c > 0, s / np.maximum(c, 1), y)                 # fallback to y(i)
        x[abnormal] = fill[abnormal]
    else:
        raise ValueError(rule)
    rho = abnormal.mean()
    violation = (np.abs(x - y) > R + 1e-9).mean()
    return x, abnormal, rho, violation


if __name__ == "__main__":                                              # self-test
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    truth = gaussian_filter(rng.random((128, 128)), 3) * 8         # smooth test scene
    y = truth + rng.normal(0, 0.2, truth.shape)
    x_star = truth + rng.normal(0, 0.05, truth.shape)                   # stand-in denoiser output
    for rule in ("init", "obs", "clean"):
        x, A, rho, viol = restore(x_star, y, R=0.2, rule=rule)
        print(f"{rule:5s} rho_R={rho:.3f} bound_violation={viol:.3f} true_MSE={np.mean((x-truth)**2):.4f}")
