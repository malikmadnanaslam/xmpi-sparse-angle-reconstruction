"""
evaluate.py
-----------
Scores whichever reconstructions you have produced against the held-out
ground truth, and prints a summary comparison.

Run this only at the end. data/ground_truth.npz must not be used inside
your reconstruction code.

Usage:  python evaluate.py
"""

import os
import numpy as np

from eval_metrics import score_sequence

CANDIDATES = [
    ("recon_fbp_xmpi.npz", "Part A -- FBP, 3 XMPI angles (0, 30, 48 deg)"),
    ("recon_fbp_full.npz", "Part A -- FBP, 32 angles over 180 deg"),
    ("recon_advanced.npz", "Part B -- improved reconstruction, 3 XMPI angles"),
]


def main():
    gt = np.load("data/ground_truth.npz")["volumes"]
    print(f"Ground truth: {gt.shape}, attenuation range "
          f"{gt.min():.2f} to {gt.max():.2f}")

    results = {}
    for fname, label in CANDIDATES:
        if not os.path.exists(fname):
            print(f"\n[skipped] {fname} not found -- {label}")
            continue
        recon = np.load(fname)["volumes"]
        results[label] = score_sequence(recon, gt, label=label)

    if not results:
        print("\nNo reconstructions found. Run part_A_fbp_TASK.py first.")
        return

    print("\n" + "=" * 74)
    print("SUMMARY (mean over all time steps)")
    print("=" * 74)
    print(f"{'reconstruction':<52}{'NCC':>7}{'PSNR':>8}{'SSIM':>7}")
    for label, r in results.items():
        print(f"{label:<52}{r['mean_ncc']:>7.3f}"
              f"{r['mean_psnr']:>8.2f}{r['mean_ssim']:>7.3f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
