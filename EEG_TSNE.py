import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import random
import os
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy.stats import f_oneway, kruskal
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
RANDOM_STATE = 42
BATCH_SIZE = 64
MASTER_MODEL_PATH = "/kaggle/working/master_model.pt"
TSNE_PLOT_PATH = "/kaggle/working/seizure_tsne_visualization.png"
SIMILARITY_MATRIX_PATH = "/kaggle/working/seizure_similarity_matrix.png"
FINGERPRINT_SAVE_PATH = "/kaggle/working/neural_fingerprints.csv"
METRICS_TABLE_PATH = "/kaggle/working/tsne_cluster_validation_metrics.csv"

def seed_everything(seed=RANDOM_STATE):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class EEGSegmentDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spectro = np.load(row['file_path'], mmap_mode='r')[row['segment_idx']]
        return torch.tensor(spectro, dtype=torch.float32)

class DeeperCNN_BiLSTM(nn.Module):
    def __init__(self, num_eeg_channels=22, num_classes=2, 
                 conv_dropout=0.2, lstm_dropout=0.3, classifier_dropout=0.3):
        super(DeeperCNN_BiLSTM, self).__init__()
        self.conv_block1 = self._create_conv_block(num_eeg_channels, 32, conv_dropout)
        self.conv_block2 = self._create_conv_block(32, 64, conv_dropout)
        self.conv_block3 = self._create_conv_block(64, 128, conv_dropout)
        self.conv_block4 = self._create_conv_block(128, 256, conv_dropout, pool=False)
        final_conv_height = 112 // (2**3)
        self.lstm_input_features = 256 * final_conv_height
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features, hidden_size=256, num_layers=2, 
            batch_first=True, bidirectional=True, dropout=lstm_dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * 256, 128), nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, num_classes)
        )
    def _create_conv_block(self, in_channels, out_channels, dropout_rate, pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        ]
        if pool: layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def get_fingerprint(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        _, (h_n, _) = self.lstm(x)
        final_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return final_hidden_state

def analyze_and_validate_fingerprints(model, device, cleaned_df):
    preictal_df = cleaned_df[cleaned_df['label'] == 1].copy()
    seizure_files = preictal_df['file_name'].unique()
    all_fingerprints = []
    seizure_labels = []

    print("Extracting neural fingerprints for each seizure...")
    for seizure_file in seizure_files:
        seizure_name = os.path.basename(seizure_file).replace('_stft_112x112_30sec.npy', '')
        print(f"  -> Processing: {seizure_name}")
        seizure_df = preictal_df[preictal_df['file_name'] == seizure_file]
        dataset = EEGSegmentDataset(seizure_df)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        seizure_segment_fingerprints = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                fingerprint_batch = model.get_fingerprint(xb)
                seizure_segment_fingerprints.append(fingerprint_batch.cpu().numpy())
        avg_fingerprint = np.mean(np.vstack(seizure_segment_fingerprints), axis=0)
        all_fingerprints.append(avg_fingerprint)
        seizure_labels.append(seizure_name)
    print("Fingerprint extraction complete.")

    fingerprint_df = pd.DataFrame(np.array(all_fingerprints), index=seizure_labels)
    fingerprint_df.to_csv(FINGERPRINT_SAVE_PATH)
    print(f"Neural fingerprints saved to {FINGERPRINT_SAVE_PATH}")

    # Cosine Similarity Matrix
    print("\nCalculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(fingerprint_df)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap='viridis', 
                xticklabels=seizure_labels, yticklabels=seizure_labels)
    plt.title('Cosine Similarity of Seizure Neural Fingerprints')
    plt.savefig(SIMILARITY_MATRIX_PATH)
    plt.close()
    print(f"Similarity matrix plot saved to {SIMILARITY_MATRIX_PATH}")

    # t-SNE and Clustering
    print("\nPerforming t-SNE dimensionality reduction and clustering...")
    fingerprints = fingerprint_df.values
    tsne = TSNE(n_components=2, perplexity=min(5, len(seizure_labels)-1), n_iter=1000, random_state=RANDOM_STATE)
    tsne_results = tsne.fit_transform(fingerprints)
    k = 3  # or set dynamically
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(tsne_results)
    cluster_labels = kmeans.labels_

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=cluster_labels, palette="Set2", s=200)
    for i, label in enumerate(seizure_labels):
        plt.text(tsne_results[i,0]+0.05, tsne_results[i,1]+0.05, label, fontsize=9)
    plt.title("t-SNE of Seizure Neural Fingerprints")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(TSNE_PLOT_PATH)
    plt.show()
    print(f"t-SNE visualization saved to {TSNE_PLOT_PATH}")

    # Silhouette Score
    sil_score = silhouette_score(tsne_results, cluster_labels)
    print(f"Silhouette Score: {sil_score:.3f}")

    # Permutation Test for Silhouette Score
    n_permutations = 1000
    perm_sil_scores = []
    for _ in range(n_permutations):
        perm_labels = shuffle(cluster_labels, random_state=_)
        perm_sil_scores.append(silhouette_score(tsne_results, perm_labels))
    perm_sil_scores = np.array(perm_sil_scores)
    p_value = (np.sum(perm_sil_scores >= sil_score) + 1) / (n_permutations + 1)
    print(f"Permutation test p-value: {p_value:.4f}")

    # Cluster Stability (Adjusted Rand Index)
    n_runs = 10
    all_labels = []
    for seed in range(n_runs):
        tsne_tmp = TSNE(n_components=2, perplexity=min(5, len(seizure_labels)-1), n_iter=1000, random_state=seed)
        tsne_tmp_results = tsne_tmp.fit_transform(fingerprints)
        kmeans_tmp = KMeans(n_clusters=k, random_state=seed).fit(tsne_tmp_results)
        all_labels.append(kmeans_tmp.labels_)
    ari_scores = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))
    mean_ari = np.mean(ari_scores)
    print(f"Cluster Stability (mean ARI): {mean_ari:.3f}")

    # Cluster Mean Comparison (ANOVA/F-test or Kruskal-Wallis in original space)
    grouped = [fingerprints[cluster_labels == i] for i in range(k)]
    anova_label = "ANOVA"
    min_points = min(len(g) for g in grouped)
    if min_points < 2:
        print("One or more clusters have only one point; using Kruskal-Wallis (nonparametric) test.")
        try:
            h_stat, p_kw = kruskal(*grouped)
            p_val_report = float(p_kw)
            anova_label = "Kruskal-Wallis"
        except Exception as e:
            print(f"Kruskal-Wallis test failed: {e}")
            p_val_report = np.nan
            anova_label = "None"
    else:
        try:
            f_stat, p_manova = f_oneway(*grouped)
            if isinstance(p_manova, np.ndarray):
                p_val_report = float(p_manova.item())
            else:
                p_val_report = float(p_manova)
        except Exception as e:
            print(f"ANOVA test failed: {e}")
            p_val_report = np.nan
            anova_label = "None"
    if not np.isnan(p_val_report):
        print(f"Cluster mean comparison ({anova_label}) p-value: {p_val_report:.4e}")
    else:
        print("Mean comparison test not possible for the current cluster groups.")

    # Table of Results
    results = {
        "Metric": [
            "Silhouette Score",
            "Permutation p-value",
            "Cluster Stability (ARI)",
            f"Cluster Mean {anova_label} p-value",
            "t-SNE KL Divergence"
        ],
        "Value": [
            f"{sil_score:.3f}",
            f"{p_value:.4f}",
            f"{mean_ari:.3f}",
            f"{p_val_report:.2e}" if not np.isnan(p_val_report) else "-",
            f"{tsne.kl_divergence_:.4f}"
        ]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_TABLE_PATH, index=False)
    print("\nCluster Validation Metrics Table:")
    print(results_df)

if __name__ == '__main__':
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading the trained Master Model...")
    model = DeeperCNN_BiLSTM()
    model.load_state_dict(torch.load(MASTER_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    # cleaned_df must be loaded in your environment
    analyze_and_validate_fingerprints(model, device, cleaned_df)
