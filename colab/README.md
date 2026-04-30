# Colab workflow

Heavy training (VPT, LoRA, full FT) runs on Colab. Eval and quick smoke
tests stay on the local machine.

## One-time setup

### 1. Push the repo to GitHub

```powershell
# Locally, after `git init` and the initial commit:
git remote add origin git@github.com:ImSounic/medsam-vpt.git
git push -u origin main
```

(Use HTTPS or SSH whichever you have configured.)

### 2. Upload the dataset to Google Drive

The training data needs to be in your Drive so Colab can mount it. Target
layout:

```
MyDrive/
└── medsam-vpt-data/
    ├── train_images/
    ├── train_masks/
    ├── val_images/
    ├── val_masks/
    ├── test_images/
    └── test_masks/
```

Easiest path: drag the entire local `data/` folder into Drive via the web
UI. ~13 GB total — count on a few hours depending on your upload speed.
This is a one-off; subsequent Colab sessions just mount Drive.

If you've already trained anything on the laptop and want those
checkpoints accessible from Colab, also upload `checkpoints/runs/` to
`MyDrive/medsam-vpt-data/checkpoints_runs/`.

### 3. (Optional) Sanity-check a Colab session

Open `colab/train.ipynb` in Colab → Runtime → Change runtime type → T4
GPU. Run cells 1–6. If the data symlink works (`ls data` shows the
expected folders), you're set.

## Per-session workflow

1. Open `colab/train.ipynb` in Colab.
2. Confirm runtime is GPU (T4 free, or L4/A100 on Pro).
3. Run cells in order. The notebook clones the latest code from GitHub,
   installs deps, mounts Drive, symlinks the data, downloads MedSAM
   weights, then runs the configured training.
4. Edit the `CONFIG` variable in the notebook to pick which method to
   train (`vpt_shallow`, `vpt_deep`, `full_ft`, etc.).
5. After training finishes, the final cell copies the new checkpoints
   back to Drive so they survive the runtime disconnecting.

## Why Colab, why not laptop

The laptop GPU (RTX 1000 Blackwell mobile, 8 GB) thermally throttles or
crashes on sustained ~30 minute training runs at MedSAM's native
1024×1024 resolution. Colab T4 has 16 GB VRAM and proper data-center
cooling — runs are reliable.

Eval is light enough (~8 minutes per dataset, 5 GB peak) to stay on the
laptop.

## What stays on the laptop

- Code editing (your IDE)
- `python -m src.eval --config ... --checkpoint ...` runs
- Quick smoke tests
- Final analysis / plotting
