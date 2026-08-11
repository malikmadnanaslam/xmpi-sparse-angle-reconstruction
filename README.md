# XMPI sparse-angle 4D reconstruction

Technical exercise submission for the Lund University postdoctoral project on fast time-resolved 3D X-ray multi-projection imaging (XMPI) for advanced manufacturing.

## What this repository contains

This submission reconstructs ten 3D attenuation volumes, each of size `32 x 32 x 32`, from X-ray projections.  The central challenge is that XMPI provides only three fixed-angle projections (`0°`, `30°`, `48°`) at each time step, whereas the reference conventional scan provides 32 angles over 180°.

The repository contains two complementary reconstructions:

| Part | Script | Input | Method | Output |
|---|---|---|---|---|
| A | `part_A_fbp_TASK.py` | 3-angle and 32-angle data | Slice-wise filtered back-projection (FBP) | `recon_fbp_xmpi.npz`, `recon_fbp_full.npz` |
| B | `part_B_advanced_TASK.py` | 3-angle data only | Physics-constrained spatiotemporal iterative reconstruction | `recon_advanced.npz` |

`NOTES.md` is the scientific report for the exercise: it gives the quantitative findings, parameter rationale, figures, limitations, and next steps. The original task statement is retained in `XMPI_Exercise_Instructions_for_Candidates.pdf` for reference.

## Method summary

### Part A: FBP baselines

Each `(z, time)` sinogram is reconstructed independently using `skimage.transform.iradon` with `output_size=32` and `circle=False`, matching the supplied Radon geometry.

- The three-view XMPI baseline uses a **Hann** filter to reduce severe high-frequency streak amplification.
- The 32-view reference uses the sharper **ramp** filter.
- No clipping is applied to the three-view FBP output: its negative and positive streaks are part of the limited-angle failure being evaluated.

### Part B: physics-constrained 4D reconstruction

The improved method uses **only** `data/xmpi_projections.npz`. It does not import `phantom.py`, read `ground_truth.npz`, use 32-view projections, or call the evaluation code during reconstruction.

1. A static cylindrical host is estimated directly from the repeated XMPI measurements by robust time/z aggregation and an exact forward-projection fit.
2. Each time frame is represented as `static host + dynamic contrast`, where the contrast is non-positive because the melt/keyhole region attenuates less than solid material.
3. The dynamic contrast is reconstructed jointly across time with an exact Radon system matrix, data fidelity, 3D total-variation denoising, one-sided sparsity, temporal smoothness, and physical bounds.
4. The iteration uses a FISTA-style accelerated projected/proximal update. Diagnostic quantities and regularization strengths are determined in the projection domain.

The optimized quantity is approximately

\[
\frac{1}{2}\|A d-(p-A b)\|_2^2 + \lambda_s\,TV_{3D}(d)
+ \lambda_1\|d\|_1
+ \frac{\lambda_t}{2}\sum_t\|d_{t+1}-d_t\|_2^2,
\qquad -b \le d \le 0.
\]

Here, `A` is the Radon forward operator, `p` is the measured three-view sinogram, `b` is the fitted static host, and `d` is the dynamic contrast.

## Reproduce the submission

### 1. Create an environment and install dependencies

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the pinned packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Generate the supplied synthetic data

```bash
python generate_dataset.py
```

This creates these files under `data/`:

| File | Purpose |
|---|---|
| `xmpi_projections.npz` | Three fixed-angle XMPI sinograms used by Parts A and B |
| `full_projections.npz` | 32-angle conventional-tomography reference used only in Part A |
| `ground_truth.npz` | Held-out truth, accessed only by `evaluate.py` |

### 3. Run Part A

```bash
python part_A_fbp_TASK.py
```

Expected outputs in the repository root:

```text
recon_fbp_xmpi.npz
recon_fbp_full.npz
```

### 4. Run Part B

```bash
python part_B_advanced_TASK.py
```

Expected outputs:

```text
recon_advanced.npz
reconstruction_diagnostics.json
```

The Part B implementation runs on a CPU laptop in roughly ten seconds after the environment is installed.

### 5. Evaluate and inspect results

```bash
python evaluate.py
python show_slices.py --t 5 --z 22
python -m unittest discover -s tests -v
```

`evaluate.py` is the only component that compares reconstructions with held-out ground truth. `show_slices.py` saves a visual slice comparison. The tests check output shape, finiteness, physical bounds, and reproducibility requirements.

## Results

Mean scores reported by the supplied evaluator are:

| Reconstruction | NCC | PSNR | SSIM |
|---|---:|---:|---:|
| Three-view FBP (Hann) | 0.676 | 6.46 dB | 0.348 |
| 32-view FBP (ramp) | 0.990 | 23.34 dB | 0.883 |
| Part B constrained method | **0.999** | **35.53 dB** | **0.983** |

The high global Part B score must be interpreted carefully. It is strongly influenced by accurate recovery of the static cylindrical host, which occupies most of the volume. The melt-pool location and coarse extent are credible, but the narrow keyhole and fine interface geometry are not reliably resolved from three views. Full scientific interpretation is in `NOTES.md`.

## Repository layout

```text
data/                            Generated projection data
figures/                         Saved axial and coronal comparisons
tests/                           Reproducibility and output tests
part_A_fbp_TASK.py               FBP baseline implementation
part_B_advanced_TASK.py          Physics-constrained 4D implementation
recon_fbp_xmpi.npz               Three-view FBP reconstruction
recon_fbp_full.npz               32-view FBP reconstruction
recon_advanced.npz               Improved three-view reconstruction
reconstruction_diagnostics.json  Projection-domain diagnostics
NOTES.md                         Scientific discussion and limitations
requirements.txt                 Reproducible dependency versions
```

## Important limitations

- Three fixed projection angles do not provide isotropic 3D information; the missing angular range causes direction-dependent artefacts.
- The static-cylinder model is appropriate for this synthetic exercise but is not a general solution for arbitrary sample geometries.
- The TV and temporal priors smooth boundaries and may suppress small, rapidly changing structures.
- The method should not be used to claim keyhole diameter, depth, interface curvature, small topological changes, or isotropic resolution without additional validation.

## Reproducibility notes

Run the commands in the order shown above from the repository root. The generated reconstruction `.npz` files store a single `volumes` array of shape `(10, 32, 32, 32)`. All hyperparameters are defined in the `ReconstructionConfig` dataclass in `part_B_advanced_TASK.py` and all diagnostic values are written to `reconstruction_diagnostics.json`.
