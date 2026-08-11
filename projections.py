"""
projections.py
--------------
Forward model: attenuation-contrast projections computed with the Radon
transform from scikit-image.

Physics
-------
Under Beer-Lambert, a monochromatic X-ray beam of incident intensity I0
that has crossed the sample arrives at the detector with intensity

    I = I0 * exp( - integral( mu ds ) )

so the quantity we actually reconstruct is the projected attenuation

    p = -log(I / I0) = integral( mu ds )

which is precisely the Radon transform of mu. We therefore compute p with
skimage.transform.radon, and (optionally) add noise in the INTENSITY
domain before taking the logarithm, which is the physically correct place
for photon-counting noise to enter.

Geometry
--------
Parallel beam, rotation about the z-axis. The volume is projected
slice-by-slice: for each z, the 2D slice vol[z] is passed to radon(),
giving one row of each projection. Stacking over z rebuilds the full 2D
detector image for every angle.

Candidates should NOT need to modify this file.
"""

import numpy as np
from skimage.transform import radon

# XMPI beamlet angles: three simultaneous, fixed viewing directions
XMPI_ANGLES = np.array([0.0, 30.0, 48.0])

# "Complete" conventional tomography sampling: 32 angles over 180 degrees
N_FULL_ANGLES = 32
FULL_ANGLES = np.linspace(0.0, 180.0, N_FULL_ANGLES, endpoint=False)


def project_volume(volume, angles_deg):
    """
    Radon-transform a 3D volume slice by slice.

    Parameters
    ----------
    volume : (nz, ny, nx) array of attenuation coefficients
    angles_deg : 1D array of projection angles in degrees

    Returns
    -------
    sinogram : (nz, n_detector, n_angles) array
        For each slice z, sinogram[z] is exactly what
        skimage.transform.radon returns for that slice.
    """
    volume = np.asarray(volume, dtype=np.float64)
    nz = volume.shape[0]

    first = radon(volume[0], theta=angles_deg, circle=False)
    sinogram = np.zeros((nz, first.shape[0], first.shape[1]), dtype=np.float64)
    sinogram[0] = first
    for z in range(1, nz):
        sinogram[z] = radon(volume[z], theta=angles_deg, circle=False)

    return sinogram


def add_photon_noise(sinogram, i0=2.0e4, rng=None):
    """
    Add Poisson photon-counting noise in the intensity domain.

    p -> I = i0 * exp(-p) -> Poisson(I) -> p' = -log(I / i0)

    `i0` is the number of incident photons per detector pixel; lower i0
    means noisier data.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    intensity = i0 * np.exp(-sinogram)
    counts = rng.poisson(intensity)
    counts = np.maximum(counts, 1)  # avoid log(0)
    return -np.log(counts / i0)


if __name__ == "__main__":
    from phantom import generate_phantom_sequence

    vols = generate_phantom_sequence()
    sino_xmpi = project_volume(vols[5], XMPI_ANGLES)
    sino_full = project_volume(vols[5], FULL_ANGLES)
    print("XMPI sinogram shape (nz, n_det, n_angles):", sino_xmpi.shape)
    print("Full sinogram shape (nz, n_det, n_angles):", sino_full.shape)
