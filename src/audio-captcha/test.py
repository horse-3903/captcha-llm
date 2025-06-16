import os
import random
from pydub import AudioSegment
from pydub.generators import WhiteNoise

# === Configuration ===
input_dir = "data/audio-captcha/ami-data-segments/ES2002a_0400"  # replace with your directory containing .wav files
output_path = "src/audio-captcha/noisy_output.wav"
noise_level_db = -40

# === Load a Random WAV File ===
wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
if not wav_files:
    raise FileNotFoundError("No .wav files found in the specified directory.")

selected_file = random.choice(wav_files)
input_path = os.path.join(input_dir, selected_file)
print(f"Selected file: {selected_file}")

# === Add White Noise ===
audio = AudioSegment.from_wav(input_path)
noise = WhiteNoise().to_audio_segment(duration=len(audio)).apply_gain(noise_level_db)
noisy_audio = audio.overlay(noise)

# === Export the Noisy Audio ===
noisy_audio.export(output_path, format="wav")
print(f"Noisy audio saved as: {output_path}")
