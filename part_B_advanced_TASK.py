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

from dataclasses import asdict, dataclass
import json

import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import radon


@dataclass(frozen=True)
class ReconstructionConfig:
    """Hyperparameters expressed relative to the measured noise where possible."""

    max_iterations: int = 80
    min_iterations: int = 25
    relative_tolerance: float = 2.0e-5
    temporal_weight: float = 0.25
    tv_noise_multiplier: float = 4.0
    sparsity_noise_multiplier: float = 0.20
    tv_inner_iterations: int = 12
    tv_tolerance: float = 2.0e-5
    support_energy_fraction: float = 0.10


def _build_system_matrix(grid, angles_deg, n_detector):
    """Return the exact linear operator used by ``skimage.radon``.

    The matrix maps one flattened ``grid x grid`` slice to a flattened
    ``(n_detector, n_angles)`` sinogram.  Constructing it from unit impulses
    avoids a geometry mismatch between the iterative forward model and the
    supplied projector (including ``circle=False`` padding/interpolation).
    """
    n_angles = len(angles_deg)
    operator = np.empty((n_detector * n_angles, grid * grid), dtype=np.float64)
    impulse = np.zeros((grid, grid), dtype=np.float64)

    for pixel in range(grid * grid):
        impulse.flat[pixel] = 1.0
        projection = radon(impulse, theta=angles_deg, circle=False)
        impulse.flat[pixel] = 0.0
        if projection.shape != (n_detector, n_angles):
            raise ValueError(
                "Forward-projector geometry mismatch: expected "
                f"{(n_detector, n_angles)}, obtained {projection.shape}"
            )
        operator[:, pixel] = projection.ravel()

    return operator


def _spectral_norm_squared(operator, n_iterations=30):
    """Estimate ||A||_2^2 by deterministic power iteration."""
    vector = np.ones(operator.shape[1], dtype=np.float64)
    vector /= np.linalg.norm(vector)
    for _ in range(n_iterations):
        vector = operator.T @ (operator @ vector)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return 0.0
        vector /= norm
    projected = operator @ vector
    return float(projected @ projected)


def _fit_static_cylinder(sinograms, operator, grid, config):
    """Infer a circular static host from the XMPI measurements alone.

    The median over time and material-containing z slices suppresses the
    moving, reduced-attenuation region.  A small grid search then estimates
    the cylinder centre, radius and solid attenuation.  No value from
    ``phantom.py`` or the held-out ground truth is used.
    """
    _, nz, _, _ = sinograms.shape

    # Total positive projected attenuation robustly separates sample-bearing
    # z slices from air-only slices, even in the presence of log-Poisson noise.
    energy_by_z = np.median(
        np.sum(np.maximum(sinograms, 0.0), axis=2), axis=(0, 2)
    )
    peak_energy = float(energy_by_z.max())
    if peak_energy <= 0:
        raise ValueError("No attenuating object was detected in the projections")
    active_z = energy_by_z > config.support_energy_fraction * peak_energy
    if not np.any(active_z):
        raise ValueError("The data-driven z-support estimate is empty")

    baseline = np.median(sinograms[:, active_z, :, :], axis=(0, 1)).ravel()
    yy, xx = np.mgrid[:grid, :grid]
    nominal_center = (grid - 1.0) / 2.0

    # The reconstruction grid is centred on the rotation axis.  Search a
    # modest sub-voxel neighbourhood rather than assuming perfect alignment.
    centers = np.arange(nominal_center - 1.5, nominal_center + 1.501, 0.25)
    radii = np.arange(0.25 * grid, 0.46 * grid, 0.25)
    best = None

    for center_y in centers:
        for center_x in centers:
            distance2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
            for radius in radii:
                support = distance2 < radius ** 2
                model = operator @ support.ravel().astype(np.float64)
                model_energy = float(model @ model)
                if model_energy == 0:
                    continue
                attenuation = max(0.0, float(model @ baseline) / model_energy)
                residual = attenuation * model - baseline
                mse = float(np.mean(residual ** 2))
                if best is None or mse < best[0]:
                    best = (
                        mse,
                        float(center_x),
                        float(center_y),
                        float(radius),
                        attenuation,
                        support.copy(),
                    )

    if best is None:
        raise RuntimeError("Static-cylinder fit failed")

    mse, center_x, center_y, radius, attenuation, support_2d = best
    background = np.zeros((nz, grid, grid), dtype=np.float64)
    background[active_z] = attenuation * support_2d

    fit = {
        "active_z_first": int(np.flatnonzero(active_z)[0]),
        "active_z_last": int(np.flatnonzero(active_z)[-1]),
        "center_x_voxels": center_x,
        "center_y_voxels": center_y,
        "radius_voxels": radius,
        "solid_attenuation": attenuation,
        "static_fit_rmse": float(np.sqrt(mse)),
    }
    return background, fit, baseline


def _estimate_air_noise(sinograms, baseline):
    """Robustly estimate log-intensity noise from detector bins seeing air."""
    baseline_2d = baseline.reshape(sinograms.shape[2:])
    air_threshold = max(0.01, 0.02 * float(np.max(baseline_2d)))
    air_bins = np.abs(baseline_2d) < air_threshold
    samples = sinograms[..., air_bins]
    centre = np.median(samples)
    sigma = 1.4826 * np.median(np.abs(samples - centre))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(samples))
    return max(float(sigma), np.finfo(np.float64).eps)


def _temporal_gradient(sequence, weight):
    """Gradient of 0.5*weight*sum_t ||x[t+1]-x[t]||^2."""
    gradient = np.zeros_like(sequence)
    differences = sequence[1:] - sequence[:-1]
    gradient[1:] += weight * differences
    gradient[:-1] -= weight * differences
    return gradient


def reconstruct_advanced(
    sinograms, angles_deg, grid, config=None, return_diagnostics=False
):
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
    config = ReconstructionConfig() if config is None else config
    sinograms = np.asarray(sinograms, dtype=np.float64)
    angles_deg = np.asarray(angles_deg, dtype=np.float64)

    if sinograms.ndim != 4:
        raise ValueError(
            "sinograms must have shape (time, z, detector, angle); "
            f"received {sinograms.shape}"
        )
    n_steps, nz, n_detector, n_angles = sinograms.shape
    if n_angles != angles_deg.size:
        raise ValueError(
            f"sinogram has {n_angles} views but {angles_deg.size} angles "
            "were supplied"
        )
    if nz != grid:
        raise ValueError(f"expected nz={grid}, received nz={nz}")

    operator = _build_system_matrix(grid, angles_deg, n_detector)
    background_3d, static_fit, baseline = _fit_static_cylinder(
        sinograms, operator, grid, config
    )
    noise_sigma = _estimate_air_noise(sinograms, baseline)
    tv_weight = config.tv_noise_multiplier * noise_sigma
    sparsity_weight = config.sparsity_noise_multiplier * noise_sigma

    # Arrange all (time, z) slices as columns.  The dynamic component is a
    # non-positive deviation from the static solid host: molten material and
    # vapour both attenuate less than solid material.
    measured = sinograms.reshape(n_steps * nz, n_detector * n_angles).T
    background_4d = np.broadcast_to(
        background_3d, (n_steps,) + background_3d.shape
    ).copy()
    background_matrix = background_4d.reshape(n_steps * nz, grid * grid).T
    dynamic_sinograms = measured - operator @ background_matrix

    norm_squared = _spectral_norm_squared(operator)
    step_size = 0.95 / (norm_squared + 4.0 * config.temporal_weight)

    current = np.zeros_like(background_matrix)
    extrapolated = current.copy()
    momentum = 1.0
    relative_change = np.inf
    completed_iterations = 0

    for iteration in range(config.max_iterations):
        data_residual = operator @ extrapolated - dynamic_sinograms
        gradient = operator.T @ data_residual

        extrapolated_4d = extrapolated.T.reshape(
            n_steps, nz, grid, grid
        )
        gradient += _temporal_gradient(
            extrapolated_4d, config.temporal_weight
        ).reshape(n_steps * nz, grid * grid).T

        candidate = (extrapolated - step_size * gradient).T.reshape(
            n_steps, nz, grid, grid
        )

        # Proximal spatial prior: the change field is compact and nearly
        # piecewise constant, so apply 3D TV across (z, y, x) for each time.
        if tv_weight > 0:
            for t in range(n_steps):
                candidate[t] = denoise_tv_chambolle(
                    candidate[t],
                    weight=tv_weight * step_size,
                    eps=config.tv_tolerance,
                    max_num_iter=config.tv_inner_iterations,
                    channel_axis=None,
                )

        # One-sided L1 shrinkage encodes that the changing region is small.
        # The box projection then enforces 0 <= background + change <= solid.
        candidate = np.minimum(
            candidate + sparsity_weight * step_size, 0.0
        )
        candidate = np.maximum(candidate, -background_4d)
        next_iterate = candidate.reshape(n_steps * nz, grid * grid).T

        relative_change = float(
            np.linalg.norm(next_iterate - current)
            / max(np.linalg.norm(current), np.finfo(np.float64).eps)
        )
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2))
        extrapolated = next_iterate + (
            (momentum - 1.0) / next_momentum
        ) * (next_iterate - current)
        current = next_iterate
        momentum = next_momentum
        completed_iterations = iteration + 1

        if (
            completed_iterations >= config.min_iterations
            and relative_change < config.relative_tolerance
        ):
            break

    dynamic_4d = current.T.reshape(n_steps, nz, grid, grid)
    volumes = background_4d + dynamic_4d

    final_projection_residual = operator @ current - dynamic_sinograms
    diagnostics = {
        "method": "static-host decomposition + constrained 4D TV-FISTA",
        "angles_deg": angles_deg.tolist(),
        "noise_sigma_air": noise_sigma,
        "tv_weight": tv_weight,
        "sparsity_weight": sparsity_weight,
        "temporal_weight": config.temporal_weight,
        "step_size": step_size,
        "iterations": completed_iterations,
        "final_relative_change": relative_change,
        "projection_rmse": float(np.sqrt(np.mean(final_projection_residual ** 2))),
        "dynamic_voxel_fraction_abs_gt_0p002": float(
            np.mean(dynamic_4d < -0.002)
        ),
        "static_fit": static_fit,
        "config": asdict(config),
    }

    if return_diagnostics:
        return volumes, diagnostics
    return volumes


def main():
    xmpi = np.load("data/xmpi_projections.npz")
    grid = int(xmpi["grid"])

    recon, diagnostics = reconstruct_advanced(
        xmpi["sinograms"], xmpi["angles_deg"], grid,
        return_diagnostics=True,
    )
    np.savez_compressed("recon_advanced.npz", volumes=recon)
    print("Saved recon_advanced.npz", recon.shape)
    with open("reconstruction_diagnostics.json", "w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, indent=2)
        stream.write("\n")
    print("Saved reconstruction_diagnostics.json")
    print(
        "Data-only diagnostics: "
        f"static fit RMSE={diagnostics['static_fit']['static_fit_rmse']:.4g}, "
        f"projection RMSE={diagnostics['projection_rmse']:.4g}, "
        f"iterations={diagnostics['iterations']}"
    )


if __name__ == "__main__":
    main()
