# EEG Seizure Prediction via Heterogeneity-Informed Deep Learning and Explainable AI

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

**Official implementation of "Investigating Intra-Patient Seizure Heterogeneity via EEG Fingerprinting and Explainable AI"**

*Submitted to IEEE Access (November 2025)*

</div>

---

## 📌 Overview

Patient-specific epileptic seizure prediction framework that quantifies and manages **intra-patient heterogeneity** through dual-validation methodology combining performance metrics and interpretability analysis.

### Key Contributions

- **89.3% Variance Reduction** in model stability (CV: 31.2% → 2.3%)
- **Dual-Validation Framework:** Performance + interpretability screening
- **t-SNE Fingerprinting:** 4-metric statistical validation
- **GradCAM++ Quantification:** 4 research-grade interpretability metrics
- **Leave-One-Seizure-Out (LOSO):** Temporal file pairing for clinical realism

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Sensitivity** | 94.3% ± 2.2% |
| **Specificity** | 96.4% ± 1.8% |
| **F1-Score** | 92.9% ± 2.1% |
| **AUC** | 0.983 ± 0.012 |
| **Variance Reduction** | **89.3%** |

---

## 📂 Repository Structure

EEG_SeizurePrediction/
│

├── 📁 GradCAM++ Images/ # Interpretability visualizations

├── 📁 Similarity_Matrix/ # t-SNE clustering results

├── 📁 T_SNE/ # Dimensionality reduction outputs

├── 🐍 EEG_preprocessing.py # STFT pipeline (raw EEG → spectrograms)

├── 🐍 EEG_annotation.py # Dataset labeling utilities

├── 🐍 EEG_model_training.py # Patient-specific LOSO training

├── 🐍 EEG_master_model.py # 512-dim neural fingerprint extraction

├── 🐍 EEG_TSNE.py # Seizure fingerprinting + validation

├── 🐍 EEG_GradCAM++.py # Interpretability quantification

├── 📄 Leave-One-out per fold validation metrics.docx

├── 📄 LICENSE

└── 📄 README.md

---

## 🚀 Quick Start

### Installation

git clone https://github.com/rkv2005/EEG_SeizurePrediction.git
cd EEG_SeizurePrediction
pip install -r requirements.txt


**Requirements:**
torch>=1.10.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
pandas>=1.3.0


### Dataset Download

**CHB-MIT Scalp EEG Database** (PhysioNet):

wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/


**Cohort:** 10 patients | 80 seizure events | 22-channel EEG | 256 Hz

---

## 🔧 Usage

### Step 1: Preprocessing

Convert raw EEG to time-frequency spectrograms:

python EEG_preprocessing.py --data_path /path/to/chb-mit --output_dir ./spectrograms


**Output:** 112×112 spectrograms per channel | STFT window: 128 samples | 50% overlap

---

### Step 2: Patient-Specific Training (LOSO)

Train with Leave-One-Seizure-Out cross-validation:

python EEG_model_training.py --patient chb01 --epochs 100 --batch_size 32


**Features:**
- Temporal file pairing (adjacent interictal segments)
- Focal Loss (α=0.65, γ=2.0)
- Early stopping (patience=20)
- Data balancing: 2.5:1 interictal:preictal

---

### Step 3: Neural Fingerprint Extraction

Generate 512-dimensional representations:

python EEG_master_model.py --patient chb01 --load_checkpoint ./models/chb01_best.pth


Outputs BiLSTM hidden states for clustering.

---

### Step 4: t-SNE Fingerprinting

Cluster seizures by electrophysiological patterns:

python EEG_TSNE.py --patient chb01 --n_clusters 3 --perplexity 5


**Validation Metrics:**
- ✅ Silhouette Score (cluster separation)
- ✅ Permutation Test (statistical significance)
- ✅ Adjusted Rand Index (reproducibility)
- ✅ KL Divergence (projection quality)

---

### Step 5: GradCAM++ Analysis

Visualize time-frequency biomarkers:

python EEG_GradCAM++.py --patient chb01 --seizure_id s1 --layer conv_block4


**Quantification Metrics:**
- Gini Coefficient (energy concentration)
- Energy in Top 10% (activation focus)
- Largest Component Ratio (spatial coherence)
- Signal-to-Noise Ratio (discriminability)

---

## 🧠 Methodology

### Pipeline Overview

LOSO Cross-Validation → Exposes heterogeneity through fold variance
↓

Statistical Screening → Chi-square (p<0.05) + z-score (z<-2.0)
↓

t-SNE Fingerprinting → Clusters seizures into subgroups
↓

GradCAM++ Validation → Filters diffuse, low-SNR activations
↓

Ablation Retraining → Removes unlearnable outliers (18/80)


### Decision Criterion

Seizures removed **ONLY IF** failing **BOTH:**
- Performance screening (z<-2.0 in sensitivity/F1)
- Interpretability validation (≥2 GradCAM++ metrics below threshold)

---

## 📊 Results

### Before vs. After Ablation

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Sensitivity | 66.1% ± 20.6% | **94.3% ± 2.2%** | ↑ 42.7% |
| Specificity | 94.2% ± 3.1% | **96.4% ± 1.8%** | ↑ 2.3% |
| F1-Score | 73.8% ± 16.4% | **92.9% ± 2.1%** | ↑ 25.9% |
| AUC | 0.921 ± 0.094 | **0.983 ± 0.012** | ↑ 6.7% |
| CV (Sensitivity) | 31.2% | **2.3%** | ↓ **89.3%** |

### Case Study: CHB06

**Demonstrates heterogeneity preservation:**
- ✅ **Retained:** 3 seizures from same file with different patterns (all pass metrics)
- ❌ **Removed:** 1 seizure with diffuse activation (fails all 4 metrics)
- **Proof:** System preserves diversity, removes only unlearnable outliers

---

## 📄 Citation

If you use this code, please cite:

@article{venkatesh2025seizure,
title={Investigating Intra-Patient Seizure Heterogeneity via EEG Fingerprinting and Explainable AI},
author={Venkatesh, Raghav Kishore and Pratyush, S and Shridevi, S},
journal={IEEE Access},
year={2025},
note={Under Review}
}


---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgments

- **Dataset:** CHB-MIT Database (PhysioNet)

---

## 📧 Contact

**Raghav Kishore Venkatesh**  
School of Electronics Engineering, VIT Chennai

- 📧 Email: [raghavkishore2005@gmail.com]
- 🐛 Issues: [GitHub Issues](https://github.com/rkv2005/EEG_SeizurePrediction/issues)

---

<div align="center">

**⚡ Status:** Code accompanies IEEE Access submission (Under Review, November 2025)

Made for reproducible seizure prediction research

</div>

