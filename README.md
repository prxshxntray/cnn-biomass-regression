# CSIRO Biomass Regression: Machine Learning Research

This repository contains research and implementation of Deep Learning models for estimating biomass from pasture images, developed as part of the CSIRO Image2Biomass competition.

## 📊 Performance Summary
Our research explored several architectures and advanced training strategies.

| Model | Public R² | Private R² | Feature Set |
| :--- | :--- | :--- | :--- |
| **ResNet18 Ensemble (Sub 2)** | **0.52** | **0.48** | 3-Fold, Target Scaling, CenterCrop |
| DenseNet121 (v25) | 0.40 | 0.33 | TTA, AMP, Huber Loss |
| Hybrid ViT | 0.42 | 0.45 | Transformer Neck |

---

## 📂 Key Notebooks
Instead of browsing through experimental logs, we have curated the primary research findings:

- [**Performance Analysis & Research Report**](notebooks/Performance_Analysis.ipynb): A deep dive into why certain models outperformed others, including visualizations and theoretical analysis.
- [**Top Performing Inference (ResNet18)**](notebooks/ResNet18_Sub2_v2.ipynb): Our most robust model utilizing a 3-fold ensemble technique.
- [**Advanced Training Pipeline (DenseNet121)**](notebooks/DenseNet121_TTA_AMP_Huber_v21.ipynb): Demonstrating the use of MixUp, Huber Loss, and Hybrid Learning Rate Schedulers.

## 📦 Project Structure
- `notebooks/`: Primary research notebooks.
- `notebooks/archive/`: Historical experiments and draft versions.
- `cnn-biomass-regression/`: Utility scripts and core workflow functions.
- `models/`: Weights and saved model checkpoints.

## 🛠️ Advanced Strategies Implemented
- **MixUp & Data Augmentation**: Improving convex behavior of predictions.
- **Weighted Huber Loss**: Robust regression against biomass outliers.
- **Test-Time Augmentation (TTA)**: Multi-view inference for variance reduction.
- **OneCycleLR & Plateau Scheduling**: Optimal convergence strategies.

---
*For more details on the theory and specific results, please refer to the [Performance Analysis](notebooks/Performance_Analysis.ipynb).*
