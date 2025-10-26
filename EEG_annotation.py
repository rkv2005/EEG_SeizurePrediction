import os
import numpy as np
import pandas as pd
import re

# ---- CONFIGURATION ----
SEGMENT_DURATION_SEC = 30
OVERLAP_SEC = 25
STEP_SIZE_SEC = SEGMENT_DURATION_SEC - OVERLAP_SEC
PREICTAL_WINDOW_SEC = 30 * 60
INTERICTAL_GAP_SEC = 4 * 3600

# --- YOUR IDEA IMPLEMENTED HERE ---
# This creates the buffer zone you suggested. Any segment ending within this
# many seconds of a seizure will not be considered preictal.
PREICTAL_BUFFER_SEC = 25

# --- Paths (remains the same) ---
STFT_DATA_DIR = '/content/drive/MyDrive/Seizure_Dataset/STFT_112x112_30sec'
SUMMARY_DIR = '/content/drive/MyDrive/Seizure_Dataset/Summary'
OUTPUT_DF_DIR = '/content/drive/MyDrive/Seizure_Dataset/Summary_30sec'
os.makedirs(OUTPUT_DF_DIR, exist_ok=True)

patients_to_use = []

# --- Summary File Parsing Functions (Unchanged) ---
def parse_time(timestr):
    h, m, s = map(int, timestr.strip().split(":"))
    return h * 3600 + m * 60 + s

def extract_time_from_line(line):
    return line.split(":", 1)[1].strip()

def parse_summary(summary_file):
    file_metadata = []
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("File Name:"):
            fname = line.split(":", 1)[1].strip()
            start_time = parse_time(extract_time_from_line(lines[i+1]))
            end_time = parse_time(extract_time_from_line(lines[i+2]))
            n_seizures = int(lines[i+3].split(":", 1)[1].strip())
            seizures = []
            for s in range(n_seizures):
                sz_start = int(lines[i+4+s*2].split(":", 1)[1].strip().split()[0])
                sz_end = int(lines[i+5+s*2].split(":", 1)[1].strip().split()[0])
                seizures.append((sz_start, sz_end))
            file_metadata.append({
                'filename': fname,
                'start_time': start_time,
                'end_time': end_time,
                'seizures': seizures
            })
            i += 4 + n_seizures * 2
        else:
            i += 1
    return file_metadata

# --- LABELING FUNCTION WITH YOUR CHANGE ---
def label_segments(file_metadata, num_segments, segment_duration, step_size):
    """
    Labels segments using your idea of a 25-second buffer zone.
    """
    file_labels = []
    seizures = file_metadata['seizures']

    if not seizures:
        for i in range(num_segments):
            seg_start = i * step_size
            seg_end = seg_start + segment_duration
            file_labels.append({
                'segment_idx': i, 'segment_start_sec': seg_start, 'segment_end_sec': seg_end,
                'label': 0, 'label_name': 'interictal', 'file_name': file_metadata['filename']
            })
        return file_labels

    for i in range(num_segments):
        seg_start = i * step_size
        seg_end = seg_start + segment_duration
        label, label_name = 4, 'grey'

        is_ictal = any(
            seg_start < sz_end and seg_end > sz_start
            for sz_start, sz_end in seizures
        )
        if is_ictal:
            label, label_name = 2, 'ictal'

        # --- THE ONE-LINE CHANGE IS HERE ---
        # A segment is preictal only if it ends at least 25 seconds BEFORE the seizure starts.
        elif any(
            (sz_start - PREICTAL_WINDOW_SEC <= seg_start) and (seg_end <= sz_start - PREICTAL_BUFFER_SEC)
            for sz_start, sz_end in seizures
        ):
            label, label_name = 1, 'preictal'

        elif all(
            seg_end <= sz_start - INTERICTAL_GAP_SEC or seg_start >= sz_end + INTERICTAL_GAP_SEC
            for sz_start, sz_end in seizures
        ):
            label, label_name = 0, 'interictal'

        file_labels.append({
            'segment_idx': i, 'segment_start_sec': seg_start, 'segment_end_sec': seg_end,
            'label': label, 'label_name': label_name, 'file_name': file_metadata['filename']
        })

    return file_labels

# --- Main Execution Logic (Unchanged) ---
def main():
    for patient_name in patients_to_use:
        match = re.match(r'Patient_(\d+)', patient_name)
        if match:
            chb_num = match.group(1).zfill(2)
            summary_file_name = f"chb{chb_num}-summary.txt"
        else:
            print(f"Skipping patient directory '{patient_name}' due to inconsistent naming.")
            continue

        summary_file_path = os.path.join(SUMMARY_DIR, summary_file_name)
        if not os.path.exists(summary_file_path):
            print(f"Summary file not found for {patient_name}: {summary_file_path}")
            continue

        file_metadata_list = parse_summary(summary_file_path)
        patient_segments = []
        patient_stft_dir = os.path.join(STFT_DATA_DIR, patient_name)
        if not os.path.isdir(patient_stft_dir):
            print(f"STFT directory not found for {patient_name}: {patient_stft_dir}")
            continue

        for stft_file_name in sorted(os.listdir(patient_stft_dir)):
            if not stft_file_name.endswith('.npy'):
                continue

            original_edf_name_match = re.match(r'(chb\d+_\d+)', stft_file_name)
            if not original_edf_name_match:
                continue
            original_edf_name = original_edf_name_match.group(1) + '.edf'

            file_meta = next((item for item in file_metadata_list if item["filename"] == original_edf_name), None)
            if not file_meta:
                continue

            stft_path = os.path.join(patient_stft_dir, stft_file_name)
            try:
                num_segments = np.load(stft_path, mmap_mode='r').shape[0]
            except Exception as e:
                print(f"Error loading {stft_path} to get shape: {e}")
                continue

            file_labels = label_segments(file_meta, num_segments, SEGMENT_DURATION_SEC, STEP_SIZE_SEC)
            patient_segments.extend(file_labels)

        if patient_segments:
            patient_df = pd.DataFrame(patient_segments)
            output_csv_path = os.path.join(OUTPUT_DF_DIR, f"{patient_name}_segments.csv")
            patient_df.to_csv(output_csv_path, index=False)
            print(f"Successfully saved segment labels for {patient_name} to {output_csv_path} with {len(patient_df)} segments.")
        else:
            print(f"No segments found for patient {patient_name}.")


if __name__ == "__main__":
    main()
