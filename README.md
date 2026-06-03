<div align="center">

# captcha-llm

Research framework for evaluating LLM performance on CAPTCHA-solving tasks

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/github/license/horse-3903/captcha-llm?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/horse-3903/captcha-llm?style=flat-square)](../../commits)
[![Stars](https://img.shields.io/github/stars/horse-3903/captcha-llm?style=flat-square)](../../stargazers)

</div>

---

## Overview

`captcha-llm` is a research codebase that systematically evaluates modern large language models on a variety of CAPTCHA-style challenges, including ASCII text CAPTCHAs, rendered image CAPTCHAs, and audio CAPTCHAs. Models are queried via OpenRouter, results are logged as CSVs, and the repository also includes tooling for fine-tune data generation and small-batch timing benchmarks.

## Features

- **ASCII CAPTCHA evaluation** — test LLMs on text and image-rendered ASCII CAPTCHAs
- **Audio CAPTCHA evaluation** — multiple-choice audio recognition with optional noise transforms (`none`, `background`, `gaussian`, `combined`)
- **Fine-tune data generation** — produce Parquet datasets of ASCII CAPTCHA samples for supervised fine-tuning
- **DeepSeek OCR experiments** — standalone OCR scripts with a Docker helper
- **Benchmarking suite** — measures CAPTCHA generation and inference latency across modalities

## Tech Stack

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-FF6B35?style=for-the-badge&logo=openai&logoColor=white)](https://openrouter.ai)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

## Getting Started

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (required for model inference)
- Optional: GPU for heavy OCR or TTS workloads

### Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Some modules use optional libraries (`torch`, `scipy`, `pyarrow`, `tqdm`, `Pillow`). Install only what your workflow requires.

### Environment Variables

```powershell
$env:OPENROUTER_API_KEY = "your-key-here"
```

Or add to a `.env` file at the project root.

## Usage

### ASCII CAPTCHA Evaluation

Edit the model list and parameters inside `src/ascii-captcha/main.py`, then run:

```powershell
python src\ascii-captcha\main.py
```

Generates ASCII CAPTCHA samples from `data/ascii-captcha/` (expects `.txt` files), optionally renders them as images, and saves per-model CSV results under `results/.../raw/`.

### Audio CAPTCHA Evaluation

Edit the model list and parameters inside `src/audio-captcha/main.py`, then run:

```powershell
python src\audio-captcha\main.py
```

Reads from a CSV such as `data/audio-captcha/extended.csv`, applies the configured audio transformation, queries OpenRouter, and writes raw results to `results/.../raw/`.

### Fine-Tune Data Generation

```powershell
python src\fine-tune\generate_data.py
```

Outputs a Parquet dataset to `data/ascii-captcha-ft/`.

### Benchmarks

```powershell
# Generation timing only
python src\benchmarks\collect_timings.py --ascii-samples 25 --audio-samples 10

# Include audio inference timing (requires OPENROUTER_API_KEY)
python src\benchmarks\collect_timings.py --run-audio-inference
```

Outputs:
- `results/benchmarks/captcha_generation_times.csv`
- `results/benchmarks/audio_inference/timings_summary.csv`

## Project Structure

```
captcha-llm/
├── src/
│   ├── ascii-captcha/     # ASCII CAPTCHA evaluation (text + image render)
│   ├── audio-captcha/     # Audio CAPTCHA evaluation with noise transforms
│   ├── fine-tune/         # Fine-tune dataset generation
│   ├── deepseek-ocr/      # OCR experiments + Dockerfile
│   └── benchmarks/        # Generation & inference timing benchmarks
├── fonts/                 # Fonts used to render ASCII CAPTCHAs
├── ollama/                # Experimental local/Ollama scripts
├── data/                  # Datasets (not committed)
└── results/               # Outputs (git-ignored)
```

> **Note:** `results/` and `data/` are intentionally excluded from git. Store outputs outside the repo or remove the ignore rules deliberately if you need to track them.

## License

MIT — see [LICENSE](LICENSE)
