import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from IPython.display import display, Image
from scipy.ndimage import zoom, label
import gc
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_STATE = 42
FOLD_MODEL_DIR = "/kaggle/input/patient-10/pytorch/default/1"
TARGET_LAYER_NAME = 'conv_block4' 
GRADCAM_SAVE_DIR = f"/kaggle/working/grad_cams_plus_plus_{TARGET_LAYER_NAME}/"
SPECTROGRAM_DIMS = (112, 112)

# ==============================================================================
# SETUP FUNCTIONS
# ==============================================================================

def seed_everything(seed=RANDOM_STATE):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# DATASET & MODEL DEFINITION
# ==============================================================================

class EEGSegmentDataset(Dataset):
    """Dataset class for loading EEG spectrogram segments."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spectro = np.load(row['file_path'], mmap_mode='r')[row['segment_idx']]
        return torch.tensor(spectro, dtype=torch.float32)

class DeeperCNN_BiLSTM(nn.Module):
    """Model architecture - matches the saved training models."""
    def __init__(self, num_eeg_channels=22, num_classes=2, 
                 conv_dropout=0.4, lstm_dropout=0.5, classifier_dropout=0.5):
        super(DeeperCNN_BiLSTM, self).__init__()
        self.conv_block1 = self._create_conv_block(num_eeg_channels, 32, conv_dropout)
        self.conv_block2 = self._create_conv_block(32, 64, conv_dropout)
        self.conv_block3 = self._create_conv_block(64, 128, conv_dropout)
        self.conv_block4 = self._create_conv_block(128, 256, conv_dropout, pool=False)

        final_conv_height = SPECTROGRAM_DIMS[0] // (2**3)
        self.lstm_input_features = 256 * final_conv_height

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features, 
            hidden_size=256, num_layers=2, batch_first=True, 
            bidirectional=True, dropout=lstm_dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * 256, 128), nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, num_classes)
        )

    def _create_conv_block(self, in_channels, out_channels, dropout_rate, pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        _, (h_n, _) = self.lstm(x)
        final_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.classifier(final_hidden_state)
        return output

# ==============================================================================
# CORE 4 GRADCAM++ METRICS (Only metrics used in paper)
# ==============================================================================

def extract_core_gradcam_metrics(heatmap):
    """
    Extract ONLY the 4 core metrics used in paper:
    1. Gini Coefficient (focus)
    2. Energy in Top 10% (concentration)
    3. Largest Component Ratio (spatial coherence)
    4. Signal-to-Noise Ratio (signal quality)
    
    Args:
        heatmap: 2D numpy array (frequency x time), normalized [0, 1]
    
    Returns:
        dict with 4 core metrics
    """
    flat_map = heatmap.flatten()
    
    # =========================================================================
    # METRIC 1: Gini Coefficient (focus/concentration)
    # =========================================================================
    sorted_values = np.sort(flat_map)
    n = len(sorted_values)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * np.sum(sorted_values) + 1e-10) - (n + 1) / n
    gini_coefficient = float(gini)
    
    # =========================================================================
    # METRIC 2: Energy in Top 10% (concentration in peak regions)
    # =========================================================================
    top10pct_n = max(1, int(len(flat_map) * 0.10))
    energy_in_top10pct = float(np.sum(np.sort(flat_map)[-top10pct_n:]) / (np.sum(flat_map) + 1e-10))
    
    # =========================================================================
    # METRIC 3: Largest Component Ratio (spatial coherence)
    # =========================================================================
    threshold_75 = np.percentile(flat_map, 75)
    hot_mask = (heatmap > threshold_75).astype(int)
    labeled_array, num_features = label(hot_mask)
    
    if num_features > 0 and np.sum(hot_mask) > 0:
        component_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
        largest_component_ratio = float(max(component_sizes) / np.sum(hot_mask))
    else:
        largest_component_ratio = 0.0
    
    # =========================================================================
    # METRIC 4: Signal-to-Noise Ratio (signal quality)
    # =========================================================================
    max_activation = float(np.max(flat_map))
    noise_level = np.percentile(flat_map, 25)
    signal_to_noise_ratio = float((max_activation - noise_level) / (noise_level + 1e-10))
    
    return {
        'gini_coefficient': gini_coefficient,
        'energy_in_top10pct': energy_in_top10pct,
        'largest_component_ratio': largest_component_ratio,
        'signal_to_noise_ratio': signal_to_noise_ratio
    }

# ==============================================================================
# CORRECTED GRADCAM++ GENERATION
# ==============================================================================

def generate_specialist_grad_cams(model_dir, all_segments_df, session_map, device):
    """
    Generates Grad-CAMs++ with FIXED implementation:
    - .eval() mode (no dropout randomness)
    - Explicit target class (preictal = class 1)
    - torch.no_grad() for consistency
    - Only 4 core metrics saved
    """
    os.makedirs(GRADCAM_SAVE_DIR, exist_ok=True)
    
    gradcam_metrics_list = []
    
    for fold_id, held_out_file in enumerate(session_map, 1):
        seizure_name = os.path.basename(held_out_file).replace('_stft_112x112_30sec.npy', '')
        print(f"\n{'='*80}")
        print(f"Processing Fold {fold_id}: {seizure_name}")
        print(f"{'='*80}")

        # Load fold-specific model
        model_path = os.path.join(model_dir, f"loso_model_fold_{fold_id}.pt")
        if not os.path.exists(model_path):
            print(f"  ❌ Model not found: {model_path}")
            continue
            
        specialist_model = DeeperCNN_BiLSTM(
            conv_dropout=0.4, 
            lstm_dropout=0.5, 
            classifier_dropout=0.5
        )
        specialist_model.load_state_dict(torch.load(model_path, map_location=device))
        specialist_model.to(device).eval()  # FIXED: Keep in eval mode

        # Get target layer for GradCAM
        target_layer = getattr(specialist_model, TARGET_LAYER_NAME)
        
        # Get preictal segments for this seizure
        seizure_df = all_segments_df[
            (all_segments_df['file_name'] == held_out_file) & 
            (all_segments_df['label'] == 1)
        ].copy()
        
        if seizure_df.empty:
            print(f"  ⚠️  No preictal data found")
            continue
            
        dataset = EEGSegmentDataset(seizure_df)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Initialize GradCAM++
        cam = GradCAMPlusPlus(model=specialist_model, target_layers=[target_layer])
        
        # FIXED: Force target to preictal class (class 1)
        targets = [ClassifierOutputTarget(1)]
        
        # Generate heatmaps for all segments
        all_heatmaps = []
        for i, spectrogram_tensor in enumerate(data_loader):
            spectrogram_tensor = spectrogram_tensor.to(device)
            
            # FIXED: Use eval mode + no_grad for consistent gradients
            with torch.no_grad():
                heatmap = cam(input_tensor=spectrogram_tensor, targets=targets)[0, :]
            
            all_heatmaps.append(heatmap)
            print(f"  Segment {i+1}/{len(dataset)} processed", end='\r')

        if not all_heatmaps:
            print(f"\n  ❌ No heatmaps generated")
            continue

        # Average across all segments
        avg_heatmap = np.mean(all_heatmaps, axis=0)
        
        # Resize to match spectrogram dimensions
        resized_heatmap = zoom(
            avg_heatmap, 
            [d / s for d, s in zip(SPECTROGRAM_DIMS, avg_heatmap.shape)], 
            order=3
        )

        # Normalize to [0, 1] for metrics calculation
        heatmap_min, heatmap_max = np.min(resized_heatmap), np.max(resized_heatmap)
        if heatmap_max > heatmap_min:
            normalized_heatmap = (resized_heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            normalized_heatmap = resized_heatmap

        # ==================================================================
        # EXTRACT 4 CORE METRICS
        # ==================================================================
        metrics = extract_core_gradcam_metrics(normalized_heatmap)
        
        # Add metadata
        metrics['patient'] = seizure_name.split('_')[0]
        metrics['fold'] = fold_id
        metrics['seizure_name'] = seizure_name
        
        # Display metrics
        print(f"\n  📊 Core Metrics:")
        print(f"     Gini Coefficient: {metrics['gini_coefficient']:.3f}")
        print(f"     Energy in Top 10%: {metrics['energy_in_top10pct']:.1%}")
        print(f"     Component Ratio: {metrics['largest_component_ratio']:.3f}")
        print(f"     Signal-to-Noise: {metrics['signal_to_noise_ratio']:.2f}")
        
        gradcam_metrics_list.append(metrics)

        # Save visualization (PNG)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(normalized_heatmap, cmap='jet', aspect='auto', origin='lower')
        ax.set_title(seizure_name, fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('Time Bins', fontsize=12)
        ax.set_ylabel('Frequency Bins', fontsize=12)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Activation', rotation=270, labelpad=15)
        plt.tight_layout()
        
        png_path = os.path.join(GRADCAM_SAVE_DIR, f'gradcamplusplus_{TARGET_LAYER_NAME}_{seizure_name}.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved PNG: {png_path}")
        display(Image(filename=png_path))
        plt.close(fig)
        
        # Clean up memory
        del specialist_model, cam, all_heatmaps
        torch.cuda.empty_cache()
        gc.collect()

    # ==================================================================
    # SAVE FINAL CSV (4 METRICS ONLY)
    # ==================================================================
    if gradcam_metrics_list:
        metrics_df = pd.DataFrame(gradcam_metrics_list)
        
        # Reorder columns for clarity
        column_order = ['patient', 'fold', 'seizure_name', 'gini_coefficient', 
                       'energy_in_top10pct', 'largest_component_ratio', 'signal_to_noise_ratio']
        metrics_df = metrics_df[column_order]
        
        csv_path = os.path.join(GRADCAM_SAVE_DIR, 'gradcam_core_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"✅ SAVED CORE METRICS CSV: {csv_path}")
        print(f"{'='*80}")
        print(f"\nSummary Statistics:")
        print(f"  Total folds processed: {len(metrics_df)}")
        print(f"  Mean Gini: {metrics_df['gini_coefficient'].mean():.3f}")
        print(f"  Mean Energy Top 10%: {metrics_df['energy_in_top10pct'].mean():.1%}")
        print(f"  Mean Component Ratio: {metrics_df['largest_component_ratio'].mean():.3f}")
        print(f"  Mean SNR: {metrics_df['signal_to_noise_ratio'].mean():.2f}")
        
        return metrics_df
    else:
        print("\n⚠️  No metrics extracted")
        return None

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Assume 'cleaned_df' and 'session_map' are pre-loaded
    
    print(f"\n🚀 Starting CORRECTED GradCAM++ generation...")
    print(f"   Target Layer: {TARGET_LAYER_NAME}")
    print(f"   Metrics: Gini, Energy Top 10%, Component Ratio, SNR")
    
    results_df = generate_specialist_grad_cams(FOLD_MODEL_DIR, cleaned_df, session_map, device)
    
    print("\n✅ GradCAM++ generation complete!")
    print(f"📁 Results saved to: {GRADCAM_SAVE_DIR}")
