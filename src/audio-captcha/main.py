import os
import csv

import base64
import random

from pydub import AudioSegment
from pydub.generators import WhiteNoise

from openai import OpenAI
from dotenv import load_dotenv

# === Configuration ===
AUGMENTATION_MODE = "noise"  # Options: "none", "noise", "blend"

# === Load API Key ===
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# === Directories ===
segments_dir = "data/audio-captcha/ami-data-segments"
results_dir = "results/audio-captcha"
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, AUGMENTATION_MODE, "results_raw.csv")

# === Utilities ===
def encode_audio_to_base64(wav_path):
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def gpt4o_audio_infer(audio_b64):
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this audio about?"},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content

def compare_texts(reference, prediction):
    prompt = (
        "You are a helpful assistant. Given the following two texts:\n\n"
        f"Reference transcript:\n{reference}\n\n"
        f"LLM output:\n{prediction}\n\n"
        "Please provide a brief evaluation of how well the LLM output matches the reference transcript. "
        "Highlight any major differences, inaccuracies or omissions."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# === Augmentations ===
def add_noise(audio, noise_level_db=-20):
    noise = WhiteNoise().to_audio_segment(duration=len(audio)).apply_gain(noise_level_db)
    return audio.overlay(noise)

def blend_with_random_segment(audio, segment_folder, all_segments):
    others = [s for s in all_segments if s != segment_folder]
    if not others:
        return audio
    other = random.choice(others)
    other_path = os.path.join(segments_dir, other, f"{other}.wav")
    if not os.path.exists(other_path):
        return audio
    try:
        other_audio = AudioSegment.from_wav(other_path)
        other_audio = other_audio[:len(audio)]  # truncate if longer
        mixed = audio.overlay(other_audio - 10)  # lower volume of blend
        return mixed
    except Exception as e:
        print(f"Blend failed: {e}")
        return audio

def apply_augmentation(audio, segment_folder, all_segments):
    if AUGMENTATION_MODE == "noise":
        return add_noise(audio)
    elif AUGMENTATION_MODE == "blend":
        return blend_with_random_segment(audio, segment_folder, all_segments)
    return audio

# === Processing ===
def process_segment(segment_folder, all_segments):
    wav_path = os.path.join(segments_dir, segment_folder, f"{segment_folder}.wav")
    txt_path = os.path.join(segments_dir, segment_folder, f"{segment_folder}.txt")

    if not os.path.exists(wav_path) or not os.path.exists(txt_path):
        print(f"Missing files for {segment_folder}, skipping.")
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    try:
        audio = AudioSegment.from_wav(wav_path)
        audio_aug = apply_augmentation(audio, segment_folder, all_segments)

        # Save augmented version to temporary path
        tmp_path = "tmp.wav"
        audio_aug.export(tmp_path, format="wav")
        audio_b64 = encode_audio_to_base64(tmp_path)

        gpt_output = gpt4o_audio_infer(audio_b64)
        comparison = compare_texts(transcript, gpt_output)

        return {
            "segment": segment_folder,
            "transcript": transcript,
            "gpt_output": gpt_output,
            "comparison": comparison
        }
    except Exception as e:
        print(f"Error processing {segment_folder}: {e}")
        return None

def save_results_to_csv(rows, csv_file):
    header = ["segment", "transcript", "gpt_output", "comparison"]
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

# === Main ===
def main():
    all_segments = [d for d in os.listdir(segments_dir) if os.path.isdir(os.path.join(segments_dir, d))]
    all_segments.sort()

    for segment in all_segments[10:]:  # skip first 10 if needed
        print(f"Processing segment: {segment} (mode={AUGMENTATION_MODE})")
        data = process_segment(segment, all_segments)
        if data:
            save_results_to_csv([data], csv_path)

    print(f"Done. Results saved to {csv_path}")

if __name__ == "__main__":
    main()
