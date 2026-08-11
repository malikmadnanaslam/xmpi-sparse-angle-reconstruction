"""
generate_dataset.py
-------------------
Builds the two datasets you will compare, plus the held-out ground truth.

  data/xmpi_projections.npz  -- 3 fixed angles (0, 30, 48 deg), all time steps.
                                This is the XMPI case: three beamlets recorded
                                simultaneously, so all three views belong to the
                                SAME instant. No sample rotation is needed.

  data/full_projections.npz  -- 32 angles over 180 deg, all time steps. This is
                                the "complete" conventional tomography case. Note
                                that in a real experiment these 32 views could only
                                be acquired sequentially, while the sample rotates
                                -- so this dataset is, in effect, what you could
                                measure only if the object held still.

  data/ground_truth.npz      -- the true attenuation volumes. HELD OUT: use only
                                at the end, via evaluate.py.

Run once:  python generate_dataset.py
"""

import os
import numpy as np

from phantom import generate_phantom_sequence, GRID, N_STEPS
from projections import (
    project_volume, add_photon_noise, XMPI_ANGLES, FULL_ANGLES,
)

I0 = 2.0e5   # incident photons per detector pixel
SEED = 0


def main():
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Generating phantom: {N_STEPS} time steps of {GRID}^3 voxels ...")
    volumes = generate_phantom_sequence(grid=GRID, n_steps=N_STEPS)

    xmpi_sinos, full_sinos = [], []
    clean_all = []
    for t in range(N_STEPS):
        s_xmpi = project_volume(volumes[t], XMPI_ANGLES)
        s_full = project_volume(volumes[t], FULL_ANGLES)
        clean_all.append(s_full)
        xmpi_sinos.append(add_photon_noise(s_xmpi, i0=I0, rng=rng))
        full_sinos.append(add_photon_noise(s_full, i0=I0, rng=rng))
        print(f"  time step {t + 1}/{N_STEPS} projected", end="\r")

    xmpi_sinos = np.stack(xmpi_sinos)   # (n_steps, nz, n_det, 3)
    full_sinos = np.stack(full_sinos)   # (n_steps, nz, n_det, 32)
    print()

    # --- report the achieved projected-attenuation range ---
    clean = np.stack(clean_all)
    through = clean[clean > 1e-6]  # rays that actually cross the sample
    print()
    print("Projected attenuation p = integral(mu ds), noiseless:")
    print(f"  maximum (thickest path)      : {clean.max():.3f}")
    print(f"  median over rays through sample: {np.median(through):.3f}")
    frac = float((through >= 0.3).mean())
    print(f"  fraction of through-sample rays with p >= 0.3 : {frac:.1%}")
    print(f"  minimum transmission I/I0    : {np.exp(-clean.max()):.3f}")

    np.savez_compressed(
        "data/xmpi_projections.npz",
        sinograms=xmpi_sinos,
        angles_deg=XMPI_ANGLES,
        grid=GRID,
        n_steps=N_STEPS,
    )
    np.savez_compressed(
        "data/full_projections.npz",
        sinograms=full_sinos,
        angles_deg=FULL_ANGLES,
        grid=GRID,
        n_steps=N_STEPS,
    )
    np.savez_compressed("data/ground_truth.npz", volumes=volumes, grid=GRID)

    print()
    print("Wrote data/xmpi_projections.npz  sinograms", xmpi_sinos.shape,
          f"angles={XMPI_ANGLES.tolist()}")
    print("Wrote data/full_projections.npz  sinograms", full_sinos.shape,
          f"({len(FULL_ANGLES)} angles over 180 deg)")
    print("Wrote data/ground_truth.npz      volumes  ", volumes.shape,
          " <-- held out, do not use before evaluate.py")


if __name__ == "__main__":
    main()
