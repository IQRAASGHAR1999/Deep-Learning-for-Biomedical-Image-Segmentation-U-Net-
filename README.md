# Deep Learning for Biomedical Image Segmentation (U-Net)

Implements U-Net for biomedical image segmentation. Trains on cell nuclei or lung CT data, with full workflow from data prep to visualization.

## 📦 Structure

```
.
├── src/
│   ├── unet.py                # U-Net model
│   ├── train.py               # Training loop
│   ├── eval.py                # Evaluation/metrics
│   └── data_loader.py         # Data utilities
├── data/                      # Place downloaded datasets here
├── results/                   # Model checkpoints, segmentation outputs
├── requirements.txt
├── README.md
└── .gitignore
```

## 🏁 Getting Started

1. Download dataset (see below) and place in `data/`
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train:
   ```
   python src/train.py
   ```
4. Evaluate:
   ```
   python src/eval.py
   ```
5. Analyze results:
   - Open `notebooks/visualization.ipynb`

## 🔬 Recommended Datasets

- [Kaggle Data Science Bowl 2018 (Cell Nuclei)](https://www.kaggle.com/c/data-science-bowl-2018)
- [Lung CT Segmentation](https://www.kaggle.com/andrewmvd/lung-segmentation)

## 📊 Results

- Dice/IoU metrics per epoch
- Overlay plots in `results/`

---
