import os
from util import AudioCaptchaTester

# --------------------------
# Run the Main Process
# --------------------------
if __name__ == "__main__":
    # Set the CSV path, model list, and results path
    csv_path = "data/audio-captcha/extended.csv"  # Path to your CSV file
    model_list = [
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",

        "openai/gpt-4o-audio-preview",
        "mistralai/voxtral-small-24b-2507"
    ]
    
    results_path = "results/audio-final/none"
    # Choose the mode: "none", "background", "combined", or "gaussian"

    tester = AudioCaptchaTester(
        csv_path=csv_path,
        models=model_list,
        results_dir=results_path,
        mode="none",
        sample_limit=100,
        background_file="data/audio-captcha/bg-24k.wav",
        gaussian_level=1.50,
        background_boost=4.0,         # louder background
        combined_ratios=[0.6, 0.6],   # for overlapping
        concurrency_limit=100
    )

    for mode in ["none", "background", "combined", "gaussian"]:
        tester.mode = mode
        tester.set_results_dir(os.path.join(results_path, mode))
        tester.reset()
        tester.start()
