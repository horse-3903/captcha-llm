# captcha-llm

Research code for evaluating modern LLMs on CAPTCHA-style tasks:
- ASCII CAPTCHA recognition (text and rendered-as-image)
- Audio CAPTCHA multiple-choice recognition
- OCR experiments (DeepSeek OCR)
- Fine-tune data generation for ASCII CAPTCHAs

This repository is organized as runnable scripts rather than a packaged Python library.

## Contents

- `src/ascii-captcha/` - ASCII CAPTCHA evaluation (text + image render) and utilities
- `src/audio-captcha/` - Audio CAPTCHA evaluation (audio transformations + OpenRouter inference)
- `src/fine-tune/` - ASCII CAPTCHA fine-tuning data generation
- `src/deepseek-ocr/` - OCR experiments and Docker helper
- `src/benchmarks/` - Small-batch timing benchmarks (generation + inference)
- `ollama/` - Experimental local/ollama scripts
- `fonts/` - Fonts used for rendering ASCII CAPTCHAs
- `data/` - Datasets (not committed)
- `results/` - Outputs (ignored)

## Requirements

- Python 3.10+ recommended
- Windows PowerShell (examples below use PowerShell)
- Optional GPU for heavy OCR or TTS workloads

Install dependencies from `requirements.txt` (top-level is a general set; submodules may require extras):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Some modules use optional libraries (e.g., `torch`, `scipy`, `pyarrow`, `tqdm`, `Pillow`). Install what your workflow requires.

## Environment variables

Audio and ASCII model queries are routed via OpenRouter by default.

- `OPENROUTER_API_KEY` (required for audio and ASCII inference)

Set in `.env` or your shell environment.

## ASCII CAPTCHA evaluation

Main entrypoint:

- `src/ascii-captcha/main.py`

Key behavior:
- Generates ASCII CAPTCHA samples from `data/ascii-captcha` (expects `.txt` files)
- Can render ASCII art as images before sending to models
- Saves per-model raw CSVs under `results/.../raw/`

Example (edit the model list and parameters in `main.py`):

```powershell
python src\ascii-captcha\main.py
```

## Audio CAPTCHA evaluation

Main entrypoint:

- `src/audio-captcha/main.py`

Key behavior:
- Uses a CSV such as `data/audio-captcha/extended.csv`
- Applies optional transformations: `none`, `background`, `combined`, `gaussian`
- Sends audio to OpenRouter models
- Writes raw per-model results to `results/.../raw/`

Example (edit the model list and parameters in `main.py`):

```powershell
python src\audio-captcha\main.py
```

## Fine-tune data generation (ASCII)

Generate ASCII CAPTCHA fine-tune data as Parquet:

- `src/fine-tune/generate_data.py`

```powershell
python src\fine-tune\generate_data.py
```

Output goes to `data/ascii-captcha-ft/`.

## DeepSeek OCR experiments

Scripts:

- `src/deepseek-ocr/src/main.py`
- `src/deepseek-ocr/Dockerfile`

These are research experiments and not productionized. Use as needed.

## Benchmarks (small-batch timing)

Timing script:

- `src/benchmarks/collect_timings.py`

This measures:
- ASCII CAPTCHA generation time (figlet + render)
- Audio CAPTCHA generation time (none/gaussian/background/combined)
- Optional audio inference timing (small batch)

Run:

```powershell
python src\benchmarks\collect_timings.py --ascii-samples 25 --audio-samples 10
```

Enable audio inference timing (requires `OPENROUTER_API_KEY`):

```powershell
python src\benchmarks\collect_timings.py --run-audio-inference
```

Outputs:
- `results/benchmarks/captcha_generation_times.csv`
- `results/benchmarks/audio_inference/timings_summary.csv`

## Results and data

Generated results and datasets are intentionally ignored in git:
- `results/`
- `data/`
- `*.whl`, `*.zip`

If you need to keep outputs, store them outside the repo or remove the ignore rules intentionally.

## Notes

- This repo is research-oriented; scripts are configured in-code rather than via CLI flags.
- Model lists and parameters should be edited in the corresponding `main.py` files.
- Many results are large; keep them out of git unless you are using Git LFS deliberately.

## License

See `LICENSE`.
