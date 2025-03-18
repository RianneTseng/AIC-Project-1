import os
import librosa
import numpy as np
import pandas as pd

# Set audio directory
AUDIO_DIR = "cleaned_audio"  # Replace with the path to your audio folder
OUTPUT_CSV = "cleaned_audio_features.csv"

# Define feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Compute MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Compute Chromagram features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Compute Spectral Contrast features
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)

    # Merge features
    features = np.hstack([mfccs_mean, chroma_mean, spec_contrast_mean])
    
    return features

# Assign labels (American = 0, British = 1)
def get_label(filename):
    if "american" in filename:
        return 0
    elif "british" in filename:
        return 1
    return -1  # Unknown label

# Start extracting features from all audio files (only process files starting with 'cleaned_')
data = []
file_count = 0  # Count processed files
for filename in os.listdir(AUDIO_DIR):
    if filename.startswith("cleaned_") and filename.endswith(".wav"):
        file_count += 1
        file_path = os.path.join(AUDIO_DIR, filename)
        print(f"Extracting features of {filename}")
        features = extract_features(file_path)
        
        # Check if feature length is correct
        if len(features) != 32:
            print(f"Feature length error! {filename} has only {len(features)} features")
            continue

        label = get_label(filename)
        data.append([filename] + list(features) + [label])

print(f"Total {file_count} audio files processed.")

# Create DataFrame and save as CSV
feature_columns = [f"mfcc_{i+1}" for i in range(13)] + \
                  [f"chroma_{i+1}" for i in range(12)] + \
                  [f"spectral_{i+1}" for i in range(7)] + ["label"]

df = pd.DataFrame(data, columns=["filename"] + feature_columns)

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nFeature extraction completed. Saved to {OUTPUT_CSV}")
