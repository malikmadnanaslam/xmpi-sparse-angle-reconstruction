"""Small geometry and reconstruction tests that do not access ground truth."""

import unittest

import numpy as np
from skimage.transform import radon

from part_A_fbp_TASK import reconstruct_fbp
from part_B_advanced_TASK import (
    ReconstructionConfig,
    _build_system_matrix,
    reconstruct_advanced,
)


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.grid = 8
        self.angles = np.array([0.0, 30.0, 48.0])
        yy, xx = np.mgrid[: self.grid, : self.grid]
        self.disk = (
            (xx - 3.5) ** 2 + (yy - 3.5) ** 2 < 2.5 ** 2
        ).astype(np.float64)
        self.sinogram = radon(self.disk, theta=self.angles, circle=False)

    def test_system_matrix_matches_supplied_projector_geometry(self):
        operator = _build_system_matrix(
            self.grid, self.angles, self.sinogram.shape[0]
        )
        matrix_projection = operator @ self.disk.ravel()
        np.testing.assert_allclose(
            matrix_projection,
            self.sinogram.ravel(),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_fbp_returns_expected_4d_shape(self):
        sequence = np.broadcast_to(
            self.sinogram, (2, self.grid) + self.sinogram.shape
        ).copy()
        reconstruction = reconstruct_fbp(
            sequence, self.angles, self.grid, filter_name="hann"
        )
        self.assertEqual(
            reconstruction.shape, (2, self.grid, self.grid, self.grid)
        )
        self.assertTrue(np.isfinite(reconstruction).all())

    def test_advanced_static_case_is_physical_and_data_consistent(self):
        volume = np.zeros((self.grid, self.grid, self.grid))
        volume[2:6] = 0.04 * self.disk
        sinograms = np.stack(
            [
                np.stack(
                    [radon(sl, theta=self.angles, circle=False) for sl in volume]
                )
                for _ in range(3)
            ]
        )
        config = ReconstructionConfig(
            max_iterations=5,
            min_iterations=1,
            relative_tolerance=1.0e-8,
        )
        reconstruction, diagnostics = reconstruct_advanced(
            sinograms,
            self.angles,
            self.grid,
            config=config,
            return_diagnostics=True,
        )
        self.assertEqual(reconstruction.shape, (3, 8, 8, 8))
        self.assertGreaterEqual(float(reconstruction.min()), 0.0)
        self.assertLessEqual(
            float(reconstruction.max()),
            diagnostics["static_fit"]["solid_attenuation"] + 1.0e-12,
        )
        self.assertLess(diagnostics["static_fit"]["static_fit_rmse"], 1.0e-10)
        self.assertLess(diagnostics["projection_rmse"], 1.0e-10)


if __name__ == "__main__":
    unittest.main()
