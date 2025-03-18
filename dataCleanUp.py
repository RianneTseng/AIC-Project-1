import os
import librosa
import librosa.display
import noisereduce as nr
import soundfile as sf
import webrtcvad
import wave
import struct

# Set audio directories
AUDIO_DIR = "youtube_audio"
OUTPUT_DIR = "cleaned_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check audio file validity
def check_audio_validity(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) == 0:  # Empty audio file
            return False
        if len(y) < sr * 2:  # Audio file is shorter than 2 seconds
            return False
        return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

# Remove background noise
def denoise_audio(file_path, output_path):
    y, sr = librosa.load(file_path, sr=16000)
    noise_sample = y[0:1000]  # Take the first 1000 samples as background noise
    y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    sf.write(output_path, y_denoised, sr)

# Remove silent segments
def remove_silence(input_wav, output_wav):
    vad = webrtcvad.Vad(2)  # Set sensitivity level (0-3)
    
    with wave.open(input_wav, "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

    if channels != 1:
        print(f"Skipping {input_wav} (Not mono-channel)")
        return False  # webrtcvad only supports mono-channel audio

    frame_duration = 30  # 30ms per frame
    frame_size = int(rate * frame_duration / 1000) * width  # Calculate the correct frame size

    voiced_frames = []
    for i in range(0, len(frames), frame_size):
        frame = frames[i: i + frame_size]  # Extract an audio segment
        if len(frame) < frame_size:
            break  # Skip the last segment if it is less than 30ms
        if vad.is_speech(frame, rate):
            voiced_frames.append(frame)

    # Skip if no speech is detected
    if not voiced_frames:
        print(f"Skipping {input_wav} (No speech detected)")
        return False

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(b"".join(voiced_frames))  # Convert back to bytes format output
    
    return True

# Start processing audio files
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        input_path = os.path.join(AUDIO_DIR, filename)
        temp_denoised = os.path.join(OUTPUT_DIR, f"denoised_{filename}")
        output_path = os.path.join(OUTPUT_DIR, f"cleaned_{filename}")

        # Check if the audio is valid
        if not check_audio_validity(input_path):
            print(f"Skipping {filename} (Invalid or too short)")
            continue

        # Remove background noise
        denoise_audio(input_path, temp_denoised)

        # Remove silent segments
        if not remove_silence(temp_denoised, output_path):
            print(f"Skipping {filename} (No valid speech detected)")
            os.remove(temp_denoised)  # Remove denoised but invalid audio
        else:
            print(f"Processed: {filename} -> {output_path}")

print("All audio files processed successfully!")
