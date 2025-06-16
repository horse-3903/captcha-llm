import sys
sys.path.append("src/ascii-captcha")

import os
import random
import string
import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pyfiglet
from tqdm import tqdm
from PIL import Image
from util import text_to_image

font_lst = [
    'charact1', 'charact2', 'charact3', 'charact4', 'charact5', 'charact6', 'characte', 'ascii___', 'bubble__', 'char2___',
    'com_sen_', 'demo_1__', 'ebbs_1__', 'ebbs_2__', 'e__fist_', 'fbr2____', 'filter',
    'fraktur', 'georgi16', 'georgia11', 'gothic__', 'new_asci',
    'nscript', 'nvscript', 'o8', 'radical_', 'roman', 'roman___',
    'space_op', 't__of_ap', 'ucf_fan_', 'utopiab', 'utopiabi',
    'xhelvb', 'xhelvbi', 'xsansbi', 'xsbookb', 'xsbookbi', 'xtimes', 'xttyb'
]

def random_captcha_text(length=10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def captcha_to_ascii(captcha: str, font: str):
    assert font in font_lst
    return pyfiglet.figlet_format(captcha, font=font)

def generate_images(font: str, output_dir: str, num_samples: int = 1000):
    output_dir_font = os.path.join(output_dir, font)
    os.makedirs(output_dir_font, exist_ok=True)

    existing_files = glob.glob(os.path.join(output_dir_font, "*.png"))
    existing = set(os.path.splitext(os.path.basename(f))[0] for f in existing_files)
    num_existing = len(existing)
    num_to_generate = num_samples - num_existing

    if num_to_generate <= 0:
        print(f"[✓] Font '{font}' already has {num_samples} samples. Skipping.")
        return

    print(f"[→] Generating {num_to_generate} samples for font '{font}'...")

    for _ in tqdm(range(num_to_generate), desc=f"Font {font}", leave=False):
        captcha_text = random_captcha_text()
        while captcha_text in existing:
            captcha_text = random_captcha_text()

        ascii_text = captcha_to_ascii(captcha_text, font)
        text_path = os.path.join(output_dir_font, f"{captcha_text}.txt")

        with open(text_path, "w+") as f:
            f.write(ascii_text)

        image = text_to_image(ascii_text, font_size=32)
        image_path = os.path.join(output_dir_font, f"{captcha_text}.png")

        image.save(image_path)
        existing.add(captcha_text)

async def generate_ascii_dataset_async(output_dir: str, num_samples: int = 1000, max_workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, generate_images, font, output_dir, num_samples)
            for font in font_lst
        ]
        await asyncio.gather(*tasks)

def main():
    output_dir = "data/ascii-captcha-image"
    asyncio.run(generate_ascii_dataset_async(output_dir, num_samples=5000))
    # generate_images("univers", output_dir=os.path.join(output_dir, "univers"), num_samples=10)

if __name__ == "__main__":
    main()