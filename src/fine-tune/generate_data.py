import os
import io
import math
import pyfiglet
import random
import string
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
import pyarrow as pa
import pyarrow.parquet as pq


def _text_to_image_cached(
    text: str,
    font_path: str,
    font_size: int,
    font_cache: dict,
) -> Image.Image:
    """Render ASCII text as a monospaced image with per-process font cache."""
    font_key = (font_path, font_size)
    font = font_cache.get(font_key)
    if font is None:
        font = ImageFont.truetype(font_path, font_size)
        font_cache[font_key] = font

    lines = text.splitlines()
    line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
    total_height = int(sum(line_heights) + 10)
    max_width = int(max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in lines) + 10)

    image = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(image)

    y = 5
    for line, h in zip(lines, line_heights):
        draw.text((5, y), line, fill="black", font=font)
        y += h
    return image


def img_to_png_bytes(img: Image.Image) -> bytes:
    """Convert PIL image to raw PNG bytes for Parquet storage."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_captcha(length: int):
    char_set = string.ascii_uppercase + string.digits
    return ''.join(random.choices(char_set, k=length))


def _init_worker(font_path: str, font_size: int):
    global _FONT_PATH, _FONT_SIZE, _FONT_CACHE
    _FONT_PATH = font_path
    _FONT_SIZE = font_size
    _FONT_CACHE = {}


def _make_sample(args):
    if args["captcha_length"] is None:
        length = random.randint(7, 15)
    else:
        length = args["captcha_length"]

    ascii_text = generate_captcha(length)
    font = "ascii___"

    ascii_art = pyfiglet.figlet_format(ascii_text, font=font)
    ascii_img = _text_to_image_cached(
        ascii_art, args["font_path"], args["font_size"], _FONT_CACHE
    )
    ascii_img_bytes = img_to_png_bytes(ascii_img)

    return {
        "ascii_text": ascii_text,
        "ascii_img": ascii_img_bytes,
    }


def main(num_samples=200_000, captcha_length=7, num_workers=None, chunk_size=1000):
    out_dir = os.path.join("data", "ascii-captcha-ft")
    os.makedirs(out_dir, exist_ok=True)

    font_path = os.path.join("fonts", "courier.ttf")
    font_size = 16

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    rows = []
    worker_args = {
        "captcha_length": captcha_length,
        "font_path": font_path,
        "font_size": font_size,
    }

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(font_path, font_size),
    ) as pool:
        for row in tqdm(
            pool.imap_unordered(_make_sample, [worker_args] * num_samples, chunksize=chunk_size),
            total=num_samples,
        ):
            rows.append(row)

    # Convert to Arrow Table
    table = pa.Table.from_pylist(rows)

    # Save as a Parquet dataset
    out_path = os.path.join(out_dir, "ascii_dataset_1.parquet")
    pq.write_table(table, out_path)

    print(f"Saved {num_samples} samples → {out_path}")


if __name__ == "__main__":
    main()
