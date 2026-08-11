"""
show_slices.py
--------------
Convenience plotting helper: puts the ground truth and whichever
reconstructions exist side by side, for one slice at one time step.

Not assessed -- it is here so you spend your time on reconstruction rather
than on matplotlib. Feel free to modify or ignore it.

Usage:
    python show_slices.py                # time step 5, slice z=20
    python show_slices.py --t 8 --z 18
    python show_slices.py --axis y       # slice along a different axis
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("data/ground_truth.npz", "volumes", "ground truth"),
    ("recon_fbp_xmpi.npz", "volumes", "FBP, 3 XMPI angles"),
    ("recon_fbp_full.npz", "volumes", "FBP, 32 angles"),
    ("recon_advanced.npz", "volumes", "Part B, 3 angles"),
]


def take_slice(vol, axis, idx):
    if axis == "z":
        return vol[idx]
    if axis == "y":
        return vol[:, idx]
    return vol[:, :, idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=5, help="time step")
    ap.add_argument("--z", type=int, default=20, help="slice index")
    ap.add_argument("--axis", choices=["z", "y", "x"], default="z")
    ap.add_argument("--out", default="slices.png")
    args = ap.parse_args()

    available = [(f, k, lab) for f, k, lab in PANELS if os.path.exists(f)]
    if not available:
        print("Nothing to plot. Run generate_dataset.py first.")
        return

    gt = np.load("data/ground_truth.npz")["volumes"]
    vmin, vmax = float(gt.min()), float(gt.max())

    fig, axes = plt.subplots(1, len(available), figsize=(3.1 * len(available), 3.4))
    if len(available) == 1:
        axes = [axes]

    for ax, (fname, key, label) in zip(axes, available):
        vol = np.load(fname)[key][args.t]
        ax.imshow(take_slice(vol, args.axis, args.z),
                  cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"time step {args.t}, {args.axis} = {args.z} "
                 f"(common grey scale)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}  ({len(available)} panels)")


if __name__ == "__main__":
    main()
