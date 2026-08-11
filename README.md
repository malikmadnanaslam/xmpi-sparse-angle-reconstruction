# XMPI sparse-angle reconstruction exercise

**Time budget: 8 hours maximum.** A thoughtful, partly finished attempt with clear
reasoning is worth more to us than a polished result that took three days.

Full instructions are in the accompanying document
`XMPI_Exercise_Instructions_for_Candidates.pdf`. This file is the quick reference.

## Setup

```bash
pip install numpy scikit-image matplotlib
python generate_dataset.py
```

That writes three files into `data/`:

| File | Contents |
|---|---|
| `xmpi_projections.npz` | 3 fixed angles (0°, 30°, 48°), all recorded at the *same instant* — the XMPI case |
| `full_projections.npz` | 32 angles over 180° — the "complete" conventional tomography case |
| `ground_truth.npz` | True attenuation volumes. **Held out** — use only via `evaluate.py` |

The object is 32×32×32 voxels at each of 10 time points. Values are linear
attenuation coefficients, calibrated so the projected attenuation ∫μ ds lies in
[0.3, 1] for rays crossing the sample. Projections are Radon transforms computed
with `skimage.transform.radon`, with Poisson photon noise added in the intensity
domain.

## Your two tasks

**Part A — `part_A_fbp_TASK.py`.** Reconstruct both datasets slice by slice with
the inverse Radon transform (`skimage.transform.iradon`). Save
`recon_fbp_xmpi.npz` and `recon_fbp_full.npz`. Compare them quantitatively and
visually against the ground truth, and explain what sparse angular sampling
costs you and why.

**Part B — `part_B_advanced_TASK.py`.** Using **only** the 3-angle XMPI data,
implement a reconstruction that measurably beats your Part A 3-angle FBP. Save
`recon_advanced.npz`. You are not expected to match the 32-angle result; closing
part of the gap and understanding why is the goal. See the file's docstring for
the priors and methods worth considering.

## Scoring and figures

```bash
python evaluate.py                 # NCC / PSNR / SSIM for whatever you have saved
python show_slices.py --t 5 --z 20 # side-by-side slice comparison -> slices.png
```

## Files

| File | Role | You edit it? |
|---|---|---|
| `phantom.py` | Builds the ground-truth 4D object | No |
| `projections.py` | Radon forward model, angle definitions, photon noise | No |
| `generate_dataset.py` | Writes the datasets | No |
| `eval_metrics.py` | NCC / PSNR / SSIM | No |
| `evaluate.py` | Scores your reconstructions | No |
| `show_slices.py` | Plotting convenience | Optional |
| **`part_A_fbp_TASK.py`** | **Part A** | **Yes** |
| **`part_B_advanced_TASK.py`** | **Part B** | **Yes** |

## What to submit

Your completed task files and any helper code, the three `.npz` reconstructions,
`NOTES.md` (roughly one page covering both parts), and a git repository with a
sensible commit history. Please also say how long you spent and flag anything
you left unfinished.

Questions are welcome: Pablo Villanueva Perez, pablo.villanueva_perez@fysik.lu.se
