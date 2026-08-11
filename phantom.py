"""
phantom.py
----------
Synthetic 4D (3D + time) phantom, loosely inspired by a laser melt pool
evolving inside a solid sample.

Everything here is expressed in ATTENUATION CONTRAST: the voxel values are
linear attenuation coefficients mu (arbitrary units). A projection is then
simply the line integral of mu along the ray, i.e. the Radon transform,
which is exactly what skimage.transform.radon computes.

Object size is 32 x 32 x 32 voxels per time point.

Candidates should NOT need to modify this file.
"""

import numpy as np

GRID = 32
N_STEPS = 10

# Linear attenuation coefficients, per voxel of path length.
#
# These are calibrated so that the PROJECTED attenuation p = integral(mu ds)
# falls in the range [0.3, 1.0] for rays crossing the sample:
#
#   * the thickest path through the sample (a diameter, ~25 voxels) gives
#     p = 1.0, i.e. about 37% transmission;
#   * a ray entering anywhere inside 95% of the sample radius gives
#     p >= 0.3.
#
# Only rays that clip the extreme rim of the cylinder fall below 0.3, and
# rays that miss the sample entirely give p = 0 as they must. The values
# remain true linear attenuation coefficients, so the projections are exact
# Radon transforms and the inverse Radon transform inverts them correctly.
MU_SOLID = 0.039925  # solid sample material
MU_MOLTEN = 0.45 * MU_SOLID  # molten / lower-density region
MU_VOID = 0.0  # keyhole / vapour cavity and surrounding air


def _coords(grid):
    zz, yy, xx = np.meshgrid(
        np.arange(grid), np.arange(grid), np.arange(grid), indexing="ij"
    )
    return xx, yy, zz


def make_sample(grid=GRID):
    """A static cylindrical sample of solid material, aligned with the
    rotation axis (z). Everything outside is vacuum/air (mu = 0)."""
    xx, yy, zz = _coords(grid)
    c = (grid - 1) / 2.0
    radius = grid * 0.38

    vol = np.zeros((grid, grid, grid), dtype=np.float64)
    inside = ((xx - c) ** 2 + (yy - c) ** 2) < radius ** 2
    # leave a little vacuum at top and bottom so the sample is finite in z
    inside &= (zz > grid * 0.12) & (zz < grid * 0.88)
    vol[inside] = MU_SOLID
    return vol


def _ellipsoid(grid, center, radii):
    xx, yy, zz = _coords(grid)
    return (
        ((xx - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((zz - center[2]) / radii[2]) ** 2
    ) <= 1.0


def generate_phantom_sequence(grid=GRID, n_steps=N_STEPS):
    """
    Returns an array of shape (n_steps, grid, grid, grid) of attenuation
    coefficients.

    Physical picture: a laser scans across the top of the sample. It leaves
    behind a melt pool (reduced attenuation) containing a narrow vapour
    keyhole (zero attenuation). Both translate with the laser and change
    size over the sequence -- the melt pool grows then shrinks, and the
    keyhole opens, deepens, and collapses.
    """
    volumes = np.zeros((n_steps, grid, grid, grid), dtype=np.float64)
    c = (grid - 1) / 2.0

    x_start, x_end = grid * 0.32, grid * 0.68
    z_surface = grid * 0.70  # melt pool sits near the top of the sample

    for i in range(n_steps):
        f = i / (n_steps - 1)  # 0 -> 1 through the sequence

        vol = make_sample(grid)

        # --- melt pool: translates in x, grows then shrinks ---
        pool_x = x_start + f * (x_end - x_start)
        pool_scale = 0.55 + 0.45 * np.sin(np.pi * f)
        pool_radii = (
            grid * 0.15 * pool_scale,
            grid * 0.11 * pool_scale,
            grid * 0.11 * pool_scale,
        )
        pool = _ellipsoid(grid, (pool_x, c, z_surface), pool_radii)
        vol[pool] = MU_MOLTEN

        # --- keyhole: narrow void inside the pool, opens and collapses ---
        keyhole_depth = np.clip(np.sin(np.pi * f) * 1.15, 0.0, 1.0)
        if keyhole_depth > 0.05:
            kh_radii = (
                grid * 0.035,
                grid * 0.035,
                grid * 0.085 * keyhole_depth,
            )
            kh_center = (pool_x, c, z_surface + grid * 0.02)
            keyhole = _ellipsoid(grid, kh_center, kh_radii)
            vol[keyhole] = MU_VOID

        volumes[i] = vol

    return volumes


if __name__ == "__main__":
    vols = generate_phantom_sequence()
    print("Phantom sequence shape:", vols.shape)
    print("Attenuation range: %.2f to %.2f" % (vols.min(), vols.max()))
    print("Unique values:", np.unique(np.round(vols, 3)))
