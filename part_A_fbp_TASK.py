"""
part_A_fbp_TASK.py
===================
PART A  --  FILTERED BACK-PROJECTION AND THE COST OF SPARSE ANGULAR SAMPLING

What you must do
----------------
Reconstruct the 4D sequence from BOTH datasets using the inverse Radon
transform (filtered back-projection) from scikit-image, applied slice by
slice, and save both results:

  1. from data/xmpi_projections.npz  --  only 3 angles (0, 30, 48 deg)
  2. from data/full_projections.npz  --  32 angles over 180 deg

Then compare both against the ground truth (evaluate.py does the scoring
for you once you have saved your reconstructions) and write up what you
find.

The key comparison
------------------
The 32-angle dataset is what conventional tomography would give you if the
object held still long enough to rotate the sample. The 3-angle dataset is
what XMPI actually measures at a single instant. Quantifying the gap
between them -- and explaining exactly what is lost and why -- is the point
of Part A.

Implementation notes
--------------------
* Use skimage.transform.iradon.
* Reconstruct slice by slice: for each z, feed iradon the 2D sinogram
  sino[z] of shape (n_detector, n_angles), with theta = the angles array,
  and circle=False to match how the data was generated in projections.py.
* iradon's `filter_name` argument matters. Try the default 'ramp' first,
  then see whether another filter helps in the 3-angle case.
* Everything here runs in seconds on a laptop -- 32^3 voxels, 10 time steps.

What to save
------------
Run this file to produce:
    recon_fbp_xmpi.npz   (key 'volumes', shape (10, 32, 32, 32))
    recon_fbp_full.npz   (key 'volumes', shape (10, 32, 32, 32))
evaluate.py expects exactly these filenames and that key.

What to write up (in NOTES.md)
------------------------------
* The quantitative gap between the two reconstructions (NCC / PSNR / SSIM).
* What the 3-angle reconstruction gets wrong: describe the artefacts, and
  say which features of the object survive and which do not. Show at least
  one figure comparing a central slice from each reconstruction against
  the ground truth.
* Why. Relate the failure to the angular sampling: what does sampling
  theory say about how many projections a 32-pixel-wide object needs, and
  how far short do 3 angles over 48 degrees fall?
* Whether the missing information is isotropic. Are all directions
  equally badly reconstructed, or is the damage direction-dependent?
  Explain what you observe.
"""

import numpy as np
from skimage.transform import iradon


def reconstruct_fbp(sinograms, angles_deg, grid, **iradon_kwargs):
    """
    Reconstruct a full 4D sequence by filtered back-projection.

    Parameters
    ----------
    sinograms : (n_steps, nz, n_detector, n_angles) array
    angles_deg : (n_angles,) array of projection angles
    grid : int, the side length of the output volume (32 here)

    Returns
    -------
    volumes : (n_steps, grid, grid, grid) array
    """
    # TODO: implement this.
    #
    # Sketch:
    #   for each time step t:
    #       for each slice z:
    #           sl = iradon(sinograms[t, z], theta=angles_deg,
    #                       output_size=grid, circle=False, **iradon_kwargs)
    #           store sl into volumes[t, z]
    #
    # Watch the array shapes: iradon expects (n_detector, n_angles).
    raise NotImplementedError("Implement filtered back-projection here.")


def main():
    xmpi = np.load("data/xmpi_projections.npz")
    full = np.load("data/full_projections.npz")
    grid = int(xmpi["grid"])

    print(f"XMPI dataset: {xmpi['sinograms'].shape}, angles = {xmpi['angles_deg']}")
    print(f"Full dataset: {full['sinograms'].shape}, "
          f"{len(full['angles_deg'])} angles over 180 deg")

    recon_xmpi = reconstruct_fbp(xmpi["sinograms"], xmpi["angles_deg"], grid)
    np.savez_compressed("recon_fbp_xmpi.npz", volumes=recon_xmpi)
    print("Saved recon_fbp_xmpi.npz", recon_xmpi.shape)

    recon_full = reconstruct_fbp(full["sinograms"], full["angles_deg"], grid)
    np.savez_compressed("recon_fbp_full.npz", volumes=recon_full)
    print("Saved recon_fbp_full.npz", recon_full.shape)


if __name__ == "__main__":
    main()
