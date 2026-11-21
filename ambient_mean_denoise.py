import numpy as np
from scipy.ndimage import uniform_filter


def ambient_mean_denoise(img: np.ndarray, window: int = 5, sigma_thresh: float = 2.0) -> np.ndarray:
    """Simple ambient‑mean denoiser.

    For each pixel we compute the mean of a square ``window``×``window`` neighbourhood.
    Pixels that deviate from the local mean by more than ``sigma_thresh`` standard
    deviations are considered outliers and are replaced by the neighbourhood mean.

    Parameters
    ----------
    img : np.ndarray
        Input image (2‑D or 3‑D with channel axis). Expected to be in the range ``[0, 1]``.
    window : int, optional
        Size of the sliding window (must be odd for a symmetric neighbourhood).
    sigma_thresh : float, optional
        Threshold in units of the local standard deviation; outliers are replaced.

    Returns
    -------
    np.ndarray
        Denoised image with the same shape as ``img``.
    """
    # Ensure a 2‑D image for simplicity – if a channel dimension exists, process each channel.
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    # Compute local mean and variance using uniform filter.
    local_mean = uniform_filter(img, size=window, mode='reflect')
    local_sq_mean = uniform_filter(img ** 2, size=window, mode='reflect')
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var)
    # Identify outliers.
    diff = np.abs(img - local_mean)
    mask = diff > sigma_thresh * local_std
    # Replace outliers with the local mean.
    denoised = np.where(mask, local_mean, img)
    # Preserve original dtype/range.
    return denoised.astype(img.dtype)
