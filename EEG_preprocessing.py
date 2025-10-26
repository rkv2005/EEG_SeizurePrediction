import os
import numpy as np
import mne
from scipy.signal import stft
from PIL import Image

# --- GENERAL PARAMETERS ---
# Filtering
L_FREQ = 1.0
H_FREQ = 70.0
NOTCH_FREQ = 60.0

# Segmentation
SEGMENT_LENGTH_SEC = 30
OVERLAP_SEC = 25

# STFT
SFREQ = 256
NPERSEG = 128
NOVERLAP = 64
NFFT = 256
TARGET_SIZE = (112, 112)
FINAL_DTYPE = np.float32

# Canonical channel list for standardization
STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
]

# Patients to process (Using the recommended list)
patients_to_use = [
    'Patient_19'
]

def process_single_file(edf_path, output_dir):
    """
    End-to-end function with CORRECTED normalization to preprocess a single .edf file.
    """
    patient_name = os.path.basename(os.path.dirname(edf_path))
    filename = os.path.basename(edf_path).replace('.edf', '')

    # --- 1. Load, Filter, and Notch (Correct, Unchanged) ---
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, picks='eeg', fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=NOTCH_FREQ, picks='eeg', verbose=False)
    except Exception as e:
        print(f"Skipping file {edf_path} due to loading/filtering error: {e}")
        return

    # --- 2. Channel Standardization (Correct, Unchanged) ---
    t8p8_data = None
    raw_copy = raw.copy()
    if 'T8-P8-0' in raw_copy.ch_names and 'T8-P8-1' in raw_copy.ch_names:
        t8p8_data = (raw_copy.get_data(picks=['T8-P8-0'])[0] + raw_copy.get_data(picks=['T8-P8-1'])[0]) / 2
        raw_copy.drop_channels(['T8-P8-0', 'T8-P8-1'])
    elif 'T8-P8-0' in raw_copy.ch_names:
        raw_copy.rename_channels({'T8-P8-0': 'T8-P8'})
    elif 'T8-P8-1' in raw_copy.ch_names:
        raw_copy.rename_channels({'T8-P8-1': 'T8-P8'})
    available_channels = [ch for ch in STANDARD_CHANNELS if ch in raw_copy.ch_names]
    raw_available = raw_copy.copy().pick(available_channels)
    raw_available.set_eeg_reference(ref_channels='average', verbose=False)
    final_data_list = []
    for ch in STANDARD_CHANNELS:
        if ch == 'T8-P8' and t8p8_data is not None:
            final_data_list.append(t8p8_data)
        elif ch in raw_available.ch_names:
            final_data_list.append(raw_available.get_data(picks=[ch])[0])
        else:
            final_data_list.append(np.zeros(raw_available.n_times))
    final_data_array = np.array(final_data_list)

    # --- 3. Global Time-Series Normalization and Segmentation (Correct, Unchanged) ---
    mean = final_data_array.mean()
    std = final_data_array.std()
    if std == 0: std = 1e-8
    data_global_norm = (final_data_array - mean) / std
    segment_length_samples = int(SEGMENT_LENGTH_SEC * SFREQ)
    overlap_samples = int(OVERLAP_SEC * SFREQ)
    step_size = segment_length_samples - overlap_samples
    n_channels, n_samples = data_global_norm.shape
    segments = []
    for i in range(0, n_samples - segment_length_samples + 1, step_size):
        segment = data_global_norm[:, i:i+segment_length_samples]
        segments.append(segment)
    if not segments:
        print(f"No segments created for {edf_path}. Skipping.")
        return
    segments = np.array(segments)

    # --- 4. STFT Conversion and Resizing (CORRECTED LOGIC) ---

    # 4a. First, compute all raw log-spectrograms for the entire file
    all_log_spectrograms = []
    for seg in segments:
        seg_spectrograms = []
        for ch_data in seg:
            f, t, Zxx = stft(
                ch_data, fs=SFREQ, window='hann',
                nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, boundary=None
            )
            # We get the raw log-spectrogram here, no normalization yet
            spectrogram = np.log(np.abs(Zxx) + 1e-8)
            seg_spectrograms.append(spectrogram)
        all_log_spectrograms.append(np.array(seg_spectrograms))

    # 4b. Compute GLOBAL min and max across all spectrograms from this file
    all_log_spectrograms_np = np.array(all_log_spectrograms)
    global_min = all_log_spectrograms_np.min()
    global_max = all_log_spectrograms_np.max()

    # 4c. Now, normalize and resize all spectrograms using the GLOBAL stats
    final_spectrograms = []
    for spec_group in all_log_spectrograms: # spec_group is a (n_channels, freq, time) array
        # Apply global normalization
        if global_max - global_min > 0:
            norm_spec_group = (spec_group - global_min) / (global_max - global_min)
        else:
            norm_spec_group = np.zeros_like(spec_group) # Avoid division by zero

        # Resize each channel's spectrogram individually
        resized_channels = []
        for ch_spec in norm_spec_group:
            resized_ch = np.array(
                Image.fromarray(ch_spec).resize(TARGET_SIZE, Image.Resampling.BICUBIC)
            )
            resized_channels.append(resized_ch)
        final_spectrograms.append(np.array(resized_channels))

    final_spectrograms = np.array(final_spectrograms, dtype=FINAL_DTYPE)

    # --- 5. Save the final processed data (Unchanged) ---
    patient_output_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_output_dir, exist_ok=True)
    save_path = os.path.join(patient_output_dir, f"{filename}_stft_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}_{SEGMENT_LENGTH_SEC}sec.npy")
    np.save(save_path, final_spectrograms)

    print(f"Successfully processed {edf_path}. Saved STFT spectrograms with shape {final_spectrograms.shape} to {save_path}")

# --- MAIN EXECUTION LOOP (Unchanged) ---
base_dir = '/content/drive/MyDrive/Seizure_Dataset/Original_dataset'
processed_dir = '/content/drive/MyDrive/Seizure_Dataset/STFT_112x112_30sec'

print("Starting batch processing with corrected pipeline...")
for patient_name in patients_to_use:
    patient_dir = os.path.join(base_dir, patient_name)
    if not os.path.isdir(patient_dir):
        print(f"Directory not found for {patient_name}. Skipping.")
        continue
    for file_name in sorted(os.listdir(patient_dir)):
        if not file_name.endswith('.edf'):
            continue
        edf_path = os.path.join(patient_dir, file_name)
        process_single_file(edf_path, processed_dir)

print("\nAll spectrograms have been generated.")
