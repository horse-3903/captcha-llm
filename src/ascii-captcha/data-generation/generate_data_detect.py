import sys
sys.path.append("src/ascii-captcha")

import os
import json
import glob
import random
import string
import hashlib
from tqdm import tqdm
from PIL import Image
import pyfiglet
from util import text_to_image  # Ensure this returns a cropped PIL image of the text

# Font configuration
font_lst = [
    'charact1', 'charact2', 'charact3', 'charact4', 'charact5', 'charact6', 'characte', 'ascii___', 'bubble__', 'char2___',
    'com_sen_', 'demo_1__', 'ebbs_1__', 'ebbs_2__', 'e__fist_', 'fbr2____', 'filter',
    'fraktur', 'georgi16', 'georgia11', 'gothic__', 'new_asci',
    'nscript', 'nvscript', 'o8', 'radical_', 'roman', 'roman___',
    'space_op', 't__of_ap', 'ucf_fan_', 'utopiab', 'utopiabi',
    'xhelvb', 'xhelvbi', 'xsansbi', 'xsbookb', 'xsbookbi', 'xtimes', 'xttyb'
]

# Dataset configuration
base_output_dir = "data/ascii-captcha-image-doctr/detect"
canvas_size = (900, 600)  # Width x Height

def random_captcha_text(length=10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def captcha_to_ascii(captcha: str, font: str):
    return pyfiglet.figlet_format(captcha, font=font)

def hash_file_sha256(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def save_image_and_label(captcha_text, ascii_img, image_path, canvas_size):
    canvas = Image.new("RGB", canvas_size, (255, 255, 255))

    max_offset_x = canvas_size[0] - ascii_img.width
    max_offset_y = canvas_size[1] - ascii_img.height
    offset_x = random.randint(0, max(0, max_offset_x))
    offset_y = random.randint(0, max(0, max_offset_y))

    canvas.paste(ascii_img, (offset_x, offset_y))
    canvas.save(image_path)

    polygon = [
        [offset_x, offset_y],
        [offset_x + ascii_img.width, offset_y],
        [offset_x + ascii_img.width, offset_y + ascii_img.height],
        [offset_x, offset_y + ascii_img.height]
    ]

    return canvas.size, polygon, hash_file_sha256(image_path)

def generate_dataset(total_samples, output_dir):
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    existing_files = set(os.path.basename(f) for f in glob.glob(os.path.join(image_dir, "*.png")))
    labels = {}

    num_existing = len(existing_files)
    num_to_generate = total_samples - num_existing
    print(f"Found {num_existing} samples. Generating {num_to_generate} more...")

    used_captchas = set()
    for _ in tqdm(range(num_to_generate), desc="Generating missing samples"):
        font = random.choice(font_lst)
        font_size = random.randint(16, 24)
        captcha_text = random_captcha_text()

        while captcha_text in used_captchas:
            captcha_text = random_captcha_text()
        used_captchas.add(captcha_text)

        ascii_text = captcha_to_ascii(captcha_text, font)
        ascii_img = text_to_image(ascii_text, font_size=font_size)

        filename = f"{captcha_text}_{font}_{font_size}.png"
        image_path = os.path.join(image_dir, filename)

        img_dims, polygon, sha256 = save_image_and_label(captcha_text, ascii_img, image_path, canvas_size)
        labels[filename] = {
            "img_dimensions": img_dims,
            "img_hash": sha256,
            "polygons": [polygon],
            "font": font,
            "font_size": font_size
        }

    print(f"✅ Dataset ready at: {output_dir}")
    return labels

def split_dataset(labels, output_dir, split_ratio=0.2):
    filenames = list(labels.keys())
    random.shuffle(filenames)
    split_idx = int(len(filenames) * (1 - split_ratio))

    train_names = filenames[:split_idx]
    val_names = filenames[split_idx:]

    for split_name, split_files in zip(["train", "val"], [train_names, val_names]):
        split_dir = os.path.join(output_dir, split_name)
        image_dir = os.path.join(split_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        split_labels = {}
        for fname in split_files:
            src_img_path = os.path.join(output_dir, "images", fname)
            dst_img_path = os.path.join(image_dir, fname)
            if not os.path.exists(dst_img_path):
                os.link(src_img_path, dst_img_path)
            split_labels[fname] = labels[fname]

        with open(os.path.join(split_dir, "labels.json"), "w") as f:
            json.dump(split_labels, f, indent=2)

        print(f"✅ {split_name.capitalize()} set: {len(split_files)} samples → {split_dir}")

def main():
    TOTAL_SAMPLES = 5000
    labels = generate_dataset(TOTAL_SAMPLES, base_output_dir)
    split_dataset(labels, base_output_dir, split_ratio=0.2)

if __name__ == "__main__":
    main()