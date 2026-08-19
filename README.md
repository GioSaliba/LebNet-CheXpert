# LebNet — CheXpert Reproduction & Robustness Extension

Reproduction of the core methodology from *CheXpert: A Large Chest Radiograph Dataset with
Uncertainty Labels and Expert Comparison* (Irvin et al., AAAI 2019), extended with a
robustness-to-image-degradation study (natural corruptions, Grad-CAM attention drift, and
FGSM adversarial testing).

Trained on a 100k-image subsample of CheXpert across all **5 of the paper's uncertainty-handling
strategies** (U-Zeros, U-Ones, U-Ignore, U-SelfTrained, U-MultiClass), with **3 independent
random-seed runs per strategy**, ensembled into a final composite model that picks the
best-performing strategy per pathology, matching the paper's own use of multiple training
runs per strategy.

## Project structure

Notebooks are numbered in the order they're meant to be run. Each one is self-contained (it
reloads whatever checkpoints it needs from the previous step), so a session can be resumed on
a fresh Kaggle runtime without re-running earlier notebooks.

| Notebook | Description |
|---|---|
| `notebooks/LebNet_Session1_Training.ipynb` | Seed 1: trains all 5 uncertainty strategies, compares them against the paper's Table 3, and builds the first composite model. |
| `notebooks/LebNet_Session1b_SecondSeed.ipynb` | Seed 2: trains a second independent run of all 5 strategies and builds a 2-seed ensemble. |
| `notebooks/LebNet_Session1c_ThirdSeed.ipynb` | Seed 3: trains a third independent run and builds the final 3-seed ensemble + composite model. |
| `notebooks/LebNet_Robustness_COMPLETE.ipynb` | Reloads all 3 seeds × 5 strategies (45 checkpoints), rebuilds the composite model, then runs the corruption sweep, Grad-CAM attention-drift analysis, FGSM adversarial testing, and cross-strategy robustness comparison. |

## Environment

Built and run on **Kaggle** (free GPU tier — T4 x2 or P100).

1. Open a notebook on Kaggle (or upload it to a new Kaggle notebook).
2. **Settings → Accelerator → GPU T4 x2** (or P100 if offered).
3. Add the CheXpert dataset via **+ Add Input** in the Kaggle sidebar.
4. Run cells top to bottom. Each notebook auto-locates the dataset and any prior checkpoints
   under `/kaggle/input`, so no path editing should be needed.

To run locally instead of on Kaggle, install the dependencies below and point `DATA_ROOT` at
your local copy of the CheXpert dataset.

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
from the Stanford ML Group. The dataset is **not included in this repository**. Request access
and download it from the link above, then follow the notebook setup instructions to point the
code at it.

## Results

See the final summary tables and degradation curves inside
`LebNet_Robustness_COMPLETE.ipynb` for the full comparison across strategies, seeds, and
corruption severities.

## Checkpoints

Model checkpoints (`.pt` / `.pth`) are not tracked in this repository due to size. Re-run the
training notebooks to regenerate them, or host them separately (e.g. Kaggle Datasets, Google
Drive, Hugging Face Hub) and link them here once available.

## Citation

If you build on the original CheXpert methodology, please cite:

> Irvin, J., Rajpurkar, P., Ko, M., et al. (2019). *CheXpert: A Large Chest Radiograph Dataset
> with Uncertainty Labels and Expert Comparison.* AAAI.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The CheXpert
dataset itself is subject to Stanford ML Group's own license terms — see the dataset page
above.
