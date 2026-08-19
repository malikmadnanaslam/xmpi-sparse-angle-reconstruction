# XMPI sparse-angle reconstruction - technical notes

**Time spent:** Approximately 6 hours 30 minutes.  
**Incomplete by design:** I did not attempt a learned prior, uncertainty calibration, motion compensation, or a general non-cylindrical support model.

## Part A - FBP and the cost of three views

I implemented slice-wise `iradon(..., output_size=32, circle=False)`. The reported three-view baseline uses Hann filtering, while the 32-view reference uses ramp filtering. Mean held-out scores were:

| Reconstruction | NCC | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| FBP, 3 views, Hann | 0.676 | 6.46 | 0.348 |
| FBP, 32 views, ramp | 0.990 | 23.34 | 0.883 |
| Part B, 3 views | **0.999** | **35.53** | **0.983** |

Thus, relative to 32-view FBP, three-view FBP loses 0.314 NCC, 16.88 dB PSNR and 0.535 SSIM. It preserves the coarse sample extent and indicates the lower-attenuation melt region, but broad bright/dark streaks distort the circular boundary, the pool is elongated, and the narrow keyhole is not resolved reliably. The damage is anisotropic: by the Fourier-slice theorem, views at only 0, 30 and 48 degrees sample three radial Fourier lines in one angular sector. Edges with normals near those measured directions survive better; missing orientations produce directional streaks and elongation. The orthogonal slice comparisons below make this dependence visible.

For a (D)-pixel object, the usual parallel-beam sampling estimate is (N_\theta \approx \pi D/2), or about 50 views for (D=32). Three views provide only 6% of that count and, more seriously, leave 132 degrees of the 180-degree angular range unobserved. The 32-view data are also below the ideal count, but are distributed across the full range and are adequate for this small, piecewise-constant object.

The filter matters, but cannot replace missing angles. A final post-hoc filter comparison gave the following `(NCC / PSNR / SSIM)` means: ramp `0.640/5.18/0.284`, Shepp-Logan `0.656/5.66/0.303`, cosine `0.672/6.20/0.335`, Hamming `0.676/6.42/0.345`, and Hann `0.676/6.46/0.348` for three views. For 32 views, ramp gave the best NCC/PSNR (`0.990/23.34`), while Shepp-Logan gave the best SSIM (`0.890`). Strong tapering helps the sparse case by suppressing amplified high-frequency streaks, but blurs the well-sampled case.

![Axial comparison at time 5, z=22](figures/comparison_axial_t05_z22.png)

![Coronal comparison at time 5, y=15](figures/comparison_coronal_t05_y15.png)

## Part B - physics-constrained spatiotemporal reconstruction

I modelled each volume as a static host (b) plus a non-positive dynamic contrast (d_t). The host is a stated physical property of the experiment - a static cylindrical specimen - but its z-support, centre, radius and attenuation were estimated only from the XMPI data. A time/z median suppresses the translating reduced-attenuation region; fitting the exact `skimage.radon` operator then gave centre `(15.5, 15.5)` voxels, radius `12.25` voxels, solid attenuation `0.03991`, and projection RMSE `3.11e-4`. The code never imports `phantom.py` or reads the held-out truth.

For the dynamic contrast I used FISTA-style accelerated projected proximal iterations on

\[
\tfrac12\|A d-(p-A b)\|_2^2 + \lambda_s\,\mathrm{TV}_{3D}(d)
+ \lambda_1\|d\|_1 + \tfrac{\lambda_t}{2}\sum_t\|d_{t+1}-d_t\|_2^2,
\qquad -b\le d\le0.
\]

`A` is constructed from unit-impulse Radon projections, so its padding and interpolation exactly match the supplied forward model. The constraints enforce bounded support and `0 <= b+d <= b`; 3D TV encodes compact piecewise-constant changes, mild one-sided L1 shrinkage suppresses isolated noise, and the temporal term shares information across the ten frames.

TV denoising and the elementwise shrinkage/box projection are composed sequentially, so this is a practical splitting approximation rather than an exact proximal map for the sum of nonsmooth terms. A primal-dual implementation would be my next choice for a more formal optimizer.

Regularization was chosen without truth. Robust air-ray noise was `sigma=2.32e-3`; I set `lambda_s=4 sigma=9.27e-3`, `lambda_1=0.2 sigma=4.64e-4`, and `lambda_t=0.25`. The final reprojection RMSE was `2.91e-3`, close to the log-Poisson noise scale. A data-only sweep of TV multipliers `0, 2, 4, 8` gave projection RMSE `0.00247, 0.00280, 0.00291, 0.00316`; the chosen factor 4 is near the discrepancy elbow and roughly halves spatial roughness relative to no TV. Weaker regularization fits noise and leaves angular ghosts; stronger TV shrinks/rounds the pool, temporally lags motion and removes the keyhole.

Part B improves over the strongest three-view FBP by 0.323 NCC, 29.07 dB PSNR and 0.635 SSIM. It even exceeds 32-view FBP on these global metrics because the known cylindrical host occupies most voxels and is fitted extremely accurately. This does **not** mean three projections contain more information than 32: it shows how strongly a correct parametric prior can dominate whole-volume metrics on this synthetic case. Visually, the pool location and coarse envelope are credible, but its boundary is rounded/shifted and the narrow void is blended into the molten region. I would report sample support, approximate pool centroid and coarse volume only after projection-domain/uncertainty validation. I would refuse to claim keyhole diameter, depth, interface curvature, small topology changes, or isotropic spatial resolution.

With more time I would add held-out-angle/synthetic-geometry validation, primal-dual joint spatial-temporal TV with explicit uncertainty, and a less restrictive support model. With more data/compute I would compare motion-aware low-rank+sparse reconstruction and a self-supervised 4D neural field using the same projection-domain loss.

**AI assistance disclosure:** ChatGPT used for understanding the problem.
