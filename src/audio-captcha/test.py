from util import preview_audio_modes

if __name__ == "__main__":
    preview_audio_modes(
        base_audio=r"C:\Users\chong\Desktop\Coding\Github\captcha-llm\data\audio-captcha\audio\id-174.wav",
        background_audio=r"C:\Users\chong\Desktop\Coding\Github\captcha-llm\data\audio-captcha\bg-24k.wav",
        output_dir=r"C:\Users\chong\Desktop\Coding\Github\captcha-llm\results\audio-samples",
        gaussian_level=1.5,
        overlap_count=2,
        overlap_ratios=[0.6, 0.6],
        background_boost=4.0,
    )

