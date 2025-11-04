EEG Seizure Prediction via Heterogeneity-Informed Deep Learning
[![License: MIT](https://img.shields.://opensource.org/licenses/MIT
[![IEEE Access](https://img.shields.io/badge/Paper-IEEE%20Access-blueation of "Investigating Intra-Patient Seizure Heterogeneity via EEG Fingerprinting and Explainable AI" submitted to IEEE Access.

Overview
This repository provides a complete framework for patient-specific epileptic seizure prediction that explicitly quantifies and manages intra-patient seizure heterogeneity through dual-validation methodology combining performance metrics and interpretability analysis.

Key Features:

Hybrid CNN-BiLSTM architecture for EEG-based seizure prediction

Leave-One-Seizure-Out (LOSO) cross-validation for heterogeneity exposure

Statistical screening framework (Chi-square + z-score analysis)

t-SNE-based seizure fingerprinting with 4-metric validation

GradCAM++ interpretability quantification with 4 quality metrics

Heterogeneity-informed ablation achieving 89.3% variance reduction

Performance:

Sensitivity: 94.3%

Specificity: 96.4%

F1-Score: 92.9%

AUC: 0.983

Variance Reduction: CV 31.2% → 2.3% (89.3% improvement)

Repository Structure
text
EEG_SeizurePrediction/
├── GradCAM++ Images/          # Visualization outputs
├── Similarity_Matrix/          # t-SNE clustering results
├── T_SNE/                      # t-SNE projection outputs
├── EEG_GradCAM++.py           # GradCAM++ interpretability analysis
├── EEG_TSNE.py                # t-SNE fingerprinting implementation
├── EEG_annotation.py          # Dataset annotation utilities
├── EEG_master_model.py        # Master model training (512-dim fingerprints)
├── EEG_model_training.py      # Patient-specific LOSO training
├── EEG_preprocessing.py       # STFT preprocessing pipeline
├── Leave-One-out per fold validation metrics.docx  # Performance tables
├── LICENSE                     # MIT License
└── README.md                   # This file
Installation
Requirements
Python 3.8+

PyTorch 1.10+

CUDA 11.3+ (for GPU acceleration)

Setup
bash
git clone https://github.com/rkv2005/EEG_SeizurePrediction.git
cd EEG_SeizurePrediction

pip install -r requirements.txt
Dependencies:

text
torch>=1.10.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
Dataset
This project uses the CHB-MIT Scalp EEG Database from PhysioNet:

10 patients, 80 seizure events

22-channel EEG recordings

Sampling rate: 256 Hz

30-minute preictal windows, 5-minute seizure prediction horizon

Download:

bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
Inclusion Criteria:

≥3 seizures per patient

30-min preictal data availability

Temporal file proximity (≤24 hours between seizures)

Usage
1. Preprocessing
Convert raw EEG to time-frequency spectrograms:

bash
python EEG_preprocessing.py --data_path /path/to/chb-mit --output_dir ./spectrograms
Parameters:

STFT window: 128 samples (0.5 sec)

Overlap: 64 samples (50%)

NFFT: 256

Output: 112×112 spectrograms per channel

2. Patient-Specific LOSO Training
Train individual models with Leave-One-Seizure-Out validation:

bash
python EEG_model_training.py --patient chb01 --epochs 100 --batch_size 32
Key Features:

Temporal file pairing (adjacent interictal segments)

Focal Loss (α=0.65, γ=2.0)

Early stopping (patience=20, min_epochs=40)

Data balancing: 2.5:1 interictal:preictal ratio

3. Master Model Training
Generate 512-dimensional neural fingerprints:

bash
python EEG_master_model.py --patient chb01 --load_checkpoint ./models/chb01_best.pth
Outputs BiLSTM hidden states for t-SNE analysis.

4. t-SNE Fingerprinting
Cluster seizures based on neural representations:

bash
python EEG_TSNE.py --patient chb01 --n_clusters 3 --perplexity 5
Validation Metrics:

Silhouette Score (cluster separation)

Permutation Test (statistical significance)

Adjusted Rand Index (reproducibility)

KL Divergence (projection quality)

5. GradCAM++ Analysis
Visualize time-frequency biomarkers:

bash
python EEG_GradCAM++.py --patient chb01 --seizure_id s1 --layer conv_block4
Quantification Metrics:

Gini Coefficient (energy concentration)

Energy in Top 10% (activation focus)

Largest Component Ratio (spatial coherence)

Signal-to-Noise Ratio (discriminability)

Methodology Overview
Pipeline
LOSO Cross-Validation → Exposes heterogeneity through fold-to-fold variance

Statistical Screening → Chi-square test (p<0.05) + z-score analysis (z<-2.0)

t-SNE Fingerprinting → Clusters seizures into electrophysiological subgroups

GradCAM++ Validation → Filters diffuse, low-SNR activations (latent confounders)

Ablation Retraining → Removes unlearnable outliers (18/80 seizures)

Decision Criterion
Seizures removed ONLY if failing BOTH:

Performance screening (z<-2.0 in sensitivity/F1)

Interpretability validation (≥2 GradCAM++ metrics below threshold)

Results
Before vs. After Ablation
Metric	Before	After	Improvement
Sensitivity	66.1% ± 20.6%	94.3% ± 2.2%	+42.7%
Specificity	94.2% ± 3.1%	96.4% ± 1.8%	+2.3%
F1-Score	73.8% ± 16.4%	92.9% ± 2.1%	+25.9%
AUC	0.921 ± 0.094	0.983 ± 0.012	+6.7%
CV (Sens)	31.2%	2.3%	-89.3%
Case Study: CHB06 (Heterogeneity Preservation)
Retained: 3 seizures from same file with different patterns (all pass metrics)

Removed: 1 seizure with diffuse activation (fails all 4 metrics)

Proof: System preserves diversity, removes only unlearnable outliers

Supplementary Materials
Located in /Leave-One-out per fold validation metrics.docx:

Table 3: Statistical screening results (Chi-square + z-scores)

Table 4: t-SNE validation metrics per patient

Table 5: GradCAM++ metrics per seizure

Table 6: Before/after ablation comparison

Figure 2-5: t-SNE projections and GradCAM++ heatmaps

Citation
If you use this code, please cite:

text
@article{venkatesh2025seizure,
  title={Investigating Intra-Patient Seizure Heterogeneity via EEG Fingerprinting and Explainable AI},
  author={Venkatesh, Raghav Kishore and Pratyush, S and Shridevi, S},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
License
This project is licensed under the MIT License - see LICENSE file.

Acknowledgments
CHB-MIT Database: PhysioNet

Latent variable framework: Das et al. (2023)

Pattern stability concept: Pei et al. (2025)

Contact
For questions or collaboration:

Email: [your-email@vit.ac.in]

GitHub Issues: github.com/rkv2005/EEG_SeizurePrediction/issues

Status: Code release accompanies IEEE Access submission (under review, November 2025)

