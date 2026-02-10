import os
from util import AudioCaptchaTester

# --------------------------
# Run the Main Process
# --------------------------
if __name__ == "__main__":
    # Set the CSV path, model list, and results path
    csv_path = "data/audio-captcha/extended.csv"  # Path to your CSV file
    model_list = [
        "google/gemini-3-flash-preview",
        "openai/gpt-audio-mini",
        "mistralai/voxtral-small-24b-2507"
    ]
    
    results_path = "results/audio-final-2/"
    # Choose the mode: "none", "background", "combined", or "gaussian"

    tester = AudioCaptchaTester(
        csv_path=csv_path,
        models=model_list,
        results_dir=results_path,
        mode="none",
        sample_limit=100,
        background_file="data/audio-captcha/bg-24k.wav",
        gaussian_level=1.70,
        background_boost=5.0,         # louder background
        combined_ratios=[0.7, 0.7],   # for overlapping
        concurrency_limit=100
    )

    for mode in ["none", "background", "combined", "gaussian"]:
        tester.mode = mode
        tester.set_results_dir(os.path.join(results_path, mode))
        tester.reset()
        tester.start()
