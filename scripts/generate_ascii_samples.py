"""
Generate a small set of ASCII CAPTCHA samples using pyfiglet.

Each sample is saved as a plain-text file whose stem is the ground-truth label.
This mirrors the format of data/ascii-captcha/ used by the evaluation pipeline.

Usage:
    python scripts/generate_ascii_samples.py --n 20 --out examples/ascii_samples/
"""
import argparse
import os
import random
import string
import sys

try:
    import pyfiglet
except ImportError:
    print("pyfiglet is required: pip install pyfiglet")
    sys.exit(1)


FONTS = ["standard", "banner", "big", "block", "bubble", "digital", "ivrit", "mini", "script", "shadow"]
CHARSET = string.ascii_uppercase + string.digits


def random_label(length: int = 6) -> str:
    return "".join(random.choices(CHARSET, k=length))


def render_ascii(label: str, font: str = "standard") -> str:
    try:
        return pyfiglet.figlet_format(label, font=font)
    except pyfiglet.FontNotFound:
        return pyfiglet.figlet_format(label, font="standard")


def main():
    parser = argparse.ArgumentParser(description="Generate ASCII CAPTCHA samples")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="examples/ascii_samples", help="Output directory")
    parser.add_argument("--length", type=int, default=6, help="Character length of each label")
    parser.add_argument("--font", type=str, default=None, help="pyfiglet font (default: random per sample)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    generated = 0
    for _ in range(args.n):
        label = random_label(args.length)
        font = args.font or random.choice(FONTS)
        ascii_art = render_ascii(label, font)

        out_path = os.path.join(args.out, f"{label}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(ascii_art)
        generated += 1

    print(f"Generated {generated} ASCII CAPTCHA samples in {args.out}/")


if __name__ == "__main__":
    main()
