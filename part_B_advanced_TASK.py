"""
part_B_advanced_TASK.py
========================
PART B  --  DO BETTER THAN FILTERED BACK-PROJECTION ON THE 3-ANGLE DATA

The goal
--------
Using ONLY the 3-angle XMPI dataset (data/xmpi_projections.npz), produce a
reconstruction that is measurably better than your Part A filtered
back-projection of the same data. You must implement something that works
and demonstrate the improvement quantitatively.

You are NOT expected to reach the quality of the 32-angle reconstruction.
Closing part of the gap, and understanding why you closed exactly that
much, is a good outcome.

The reason this is possible at all
-----------------------------------
Filtered back-projection assumes you have sampled enough angles to invert
the Radon transform. With 3 angles you have not, so the problem is
underdetermined: many volumes reproduce your measurements equally well. To
choose sensibly among them you must add information that does not come
from the projections. Where you get that information from is the research
question, and it is what we are really assessing.

Sources of extra information you might exploit
-----------------------------------------------
* Physical constraints. Attenuation is non-negative. The sample occupies a
  bounded region. The material is close to piecewise-constant -- it takes a
  small number of distinct attenuation values (solid, molten, void) rather
  than varying smoothly.
* Regularization. Total variation, in particular, encodes the
  piecewise-constant prior and is well suited to this kind of object.
* Iterative algebraic reconstruction. SART/SIRT handle sparse and irregular
  angular sampling far more gracefully than FBP.
  skimage.transform.iradon_sart is available and is a legitimate starting
  point -- but calling it once, on its own, is not a sufficient answer for
  Part B. Build on it.
* The time axis. This is a 4D dataset, and consecutive time steps are
  strongly correlated: most of the sample is static, and only a small
  region changes. A reconstruction that treats the 10 time steps jointly
  has far more information available than 10 independent reconstructions
  do. This is the idea underpinning current XMPI reconstruction work, and
  we would encourage you to think about it.
* Learned priors. A self-supervised neural field f(x, y, z, t), trained by
  making its own forward projections match the measured sinograms, is
  another route -- and is close to what the state of the art does. If you
  go this way you will need a differentiable forward projector; note that
  the Radon transform is linear, so you can build one as a sparse matrix or
  reuse skimage's geometry to construct it.

Choose ONE main approach and do it properly. A well-executed TV-regularized
SART that you understand and can defend is worth more than three
half-finished ideas.

What to save
------------
Run this file to produce:
    recon_advanced.npz   (key 'volumes', shape (10, 32, 32, 32))
evaluate.py expects exactly this filename and key.

What to write up (in NOTES.md)
------------------------------
* What you implemented and why you chose it over the alternatives.
* The quantitative improvement over Part A's 3-angle FBP, and where you
  still fall short of the 32-angle reconstruction.
* Which prior did the work. If you used regularization, how did you choose
  its strength, and what happens when it is too strong or too weak?
* An honest account of the failure modes that remain. Which features of the
  melt pool and keyhole are recovered reliably, and which are not
  trustworthy? If you were reporting this reconstruction in a paper, what
  would you refuse to claim from it?
* What you would do next with more time or compute.
"""

import numpy as np


def reconstruct_advanced(sinograms, angles_deg, grid):
    """
    Your improved reconstruction of the 3-angle XMPI data.

    Parameters
    ----------
    sinograms : (n_steps, nz, n_detector, n_angles=3) array
    angles_deg : (3,) array -- [0, 30, 48]
    grid : int -- 32

    Returns
    -------
    volumes : (n_steps, grid, grid, grid) array
    """
    # TODO: implement your chosen approach here.
    raise NotImplementedError("Implement your improved reconstruction here.")


def main():
    xmpi = np.load("data/xmpi_projections.npz")
    grid = int(xmpi["grid"])

    recon = reconstruct_advanced(xmpi["sinograms"], xmpi["angles_deg"], grid)
    np.savez_compressed("recon_advanced.npz", volumes=recon)
    print("Saved recon_advanced.npz", recon.shape)


if __name__ == "__main__":
    main()
