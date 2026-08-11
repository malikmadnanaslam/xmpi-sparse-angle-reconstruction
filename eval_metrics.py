"""
eval_metrics.py
---------------
Metrics for comparing reconstructed volumes against the ground truth.

NCC follows the definition used for XMPI reconstruction assessment
(normalized cross-correlation over all voxels); PSNR and SSIM are the
standard image-quality measures.

Candidates should NOT need to modify this file.
"""

import numpy as np
from skimage.metrics import structural_similarity


def ncc(a, b):
    """Normalized cross-correlation between two same-shape arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())
    return 0.0 if denom == 0 else float((a * b).sum() / denom)


def psnr(recon, truth, data_range=None):
    recon = np.asarray(recon, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mse = np.mean((recon - truth) ** 2)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = truth.max() - truth.min()
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def ssim3d(recon, truth, data_range=None):
    """Volumetric SSIM (windowed, via scikit-image)."""
    recon = np.asarray(recon, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if data_range is None:
        data_range = truth.max() - truth.min()
    return float(structural_similarity(truth, recon, data_range=data_range))


def score_sequence(recon_seq, truth_seq, label="", verbose=True):
    """
    recon_seq, truth_seq: (n_steps, grid, grid, grid)
    Returns a dict of per-time-step and mean metrics.
    """
    recon_seq = np.asarray(recon_seq, dtype=np.float64)
    truth_seq = np.asarray(truth_seq, dtype=np.float64)
    assert recon_seq.shape == truth_seq.shape, (
        f"Shape mismatch: {recon_seq.shape} vs {truth_seq.shape}"
    )

    data_range = truth_seq.max() - truth_seq.min()
    rows = []
    for t in range(recon_seq.shape[0]):
        rows.append((
            ncc(recon_seq[t], truth_seq[t]),
            psnr(recon_seq[t], truth_seq[t], data_range),
            ssim3d(recon_seq[t], truth_seq[t], data_range),
        ))

    arr = np.array(rows)
    if verbose:
        if label:
            print(f"\n{label}")
        print(f"{'step':>5} {'NCC':>8} {'PSNR(dB)':>10} {'SSIM':>8}")
        for t, (n, p, s) in enumerate(rows):
            print(f"{t:>5} {n:>8.3f} {p:>10.2f} {s:>8.3f}")
        print(f"{'mean':>5} {arr[:, 0].mean():>8.3f} {arr[:, 1].mean():>10.2f} {arr[:, 2].mean():>8.3f}")

    return {
        "ncc": arr[:, 0], "psnr": arr[:, 1], "ssim": arr[:, 2],
        "mean_ncc": float(arr[:, 0].mean()),
        "mean_psnr": float(arr[:, 1].mean()),
        "mean_ssim": float(arr[:, 2].mean()),
    }
