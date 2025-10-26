import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import random
import os
import gc
import time

# --- CONFIGURATION & CONSTANTS ---

# Master Model Settings
# Determined from the average best epoch from the inner loops of your cross-validation.
OPTIMAL_EPOCHS = 42       
UNDERSAMPLE_RATIO = 3.0   
BATCH_SIZE = 64
RANDOM_STATE = 42         

# Paths
MODEL_SAVE_PATH = "/kaggle/working/master_model.pt"

# --- HELPER CLASSES AND FUNCTIONS (UNCHANGED) ---

def seed_everything(seed=RANDOM_STATE):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    """Standard Focal Loss implementation."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class EEGSegmentDataset(Dataset):
    """Dataset class for loading EEG spectrogram segments."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spectro = np.load(row['file_path'], mmap_mode='r')[row['segment_idx']]
        return torch.tensor(spectro, dtype=torch.float32), torch.tensor(int(row['label']), dtype=torch.long)

class DeeperCNN_BiLSTM(nn.Module):
    """The final model architecture with the successful, lower regularization."""
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
            nn.BatchNorm2d(out_channels), nn.ReLU()
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
        return self.classifier(final_hidden_state)

def random_undersample(df, ratio, random_state=RANDOM_STATE):
    minority = df[df['label'] == 1]
    majority = df[df['label'] == 0]
    n_majority = int(len(minority) * ratio)
    majority_downsampled = majority.sample(n=min(n_majority, len(majority)), random_state=random_state)
    return pd.concat([minority, majority_downsampled]).sample(frac=1, random_state=random_state)

# --- MAIN MASTER MODEL TRAINING SCRIPT ---
if __name__ == '__main__':
    seed_everything()
    
    # Assume cleaned_df is pre-loaded in your environment, containing 100% of the patient's data.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare the full dataset with undersampling
    print(f"Preparing full dataset of {len(cleaned_df)} segments...")
    train_df_balanced = random_undersample(cleaned_df, ratio=UNDERSAMPLE_RATIO)
    train_loader = DataLoader(EEGSegmentDataset(train_df_balanced), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    print(f"Training on {len(train_df_balanced)} balanced segments.")

    # 2. Initialize the model and optimizer
    model_params = {'num_eeg_channels': 22, 'num_classes': 2}
    optimizer_params = {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4}
    
    model = DeeperCNN_BiLSTM(**model_params).to(device)
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer_params['lr'], 
                                                    epochs=OPTIMAL_EPOCHS, steps_per_epoch=len(train_loader))
    criterion = FocalLoss().to(device)

    # 3. Train for the optimal number of epochs (no validation)
    print(f"Starting final training for a fixed {OPTIMAL_EPOCHS} epochs...")
    model.train()
    
    for epoch in range(OPTIMAL_EPOCHS):
        epoch_start_time = time.time()
        total_train_loss = 0
        
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item() * xb.size(0)
            
            elapsed = time.time() - epoch_start_time
            eta = (elapsed / (i + 1)) * (len(train_loader) - (i + 1))
            print(f"Epoch {epoch+1}/{OPTIMAL_EPOCHS} | Batch {i+1}/{len(train_loader)} | ETA: {eta:.0f}s", end='\r')
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{OPTIMAL_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Time: {time.time() - epoch_start_time:.1f}s")

    # 4. Save the final Master Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nMaster Model training complete. Model saved to: {MODEL_SAVE_PATH}")
    
    # Clean up memory
    del model, train_loader, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
