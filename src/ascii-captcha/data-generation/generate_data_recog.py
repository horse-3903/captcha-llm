import sys
sys.path.append("src/ascii-captcha")

import os
import json
import random
import string
from tqdm import tqdm
from PIL import Image
import pyfiglet
from util import text_to_image  # Your helper

# font_lst = [
#     'charact1', 'charact2', 'charact3', 'charact4', 'charact5', 'charact6', 'characte', 'ascii___', 'bubble__', 'char2___',
#     'com_sen_', 'demo_1__', 'ebbs_1__', 'ebbs_2__', 'e__fist_', 'fbr2____', 'filter',
#     'fraktur', 'georgi16', 'georgia11', 'gothic__', 'new_asci',
#     'nscript', 'nvscript', 'o8', 'radical_', 'roman', 'roman___',
#     'space_op', 't__of_ap', 'ucf_fan_', 'utopiab', 'utopiabi',
#     'xhelvb', 'xhelvbi', 'xsansbi', 'xsbookb', 'xsbookbi', 'xtimes', 'xttyb'
# ]

base_output_dir = "data/ascii-captcha-image-doctr/recog"
image_size = (128, 64)  # Wider canvas
train_val_split_ratio = 0.2  # 20% for validation

def random_captcha_text(length=5):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def captcha_to_ascii(captcha: str, font: str):
    return pyfiglet.figlet_format(captcha, font=font)

def resize_and_pad(image, size, pad_color=(255, 255, 255)):
    image.thumbnail(size, Image.Resampling.NEAREST)
    new_img = Image.new("RGB", size, pad_color)
    offset_x = (size[0] - image.width) // 2
    offset_y = (size[1] - image.height) // 2
    new_img.paste(image, (offset_x, offset_y))
    return new_img

def generate_recog_dataset(total_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    labels = {}
    used_texts = set()

    images_temp_dir = os.path.join(output_dir, "all_images")
    os.makedirs(images_temp_dir, exist_ok=True)

    for _ in tqdm(range(total_samples), desc="Generating samples"):
        captcha_text = random_captcha_text()
        while captcha_text in used_texts:
            captcha_text = random_captcha_text()
        used_texts.add(captcha_text)

        # font = random.choice(font_lst)
        font = "ascii___"
        font_size = 8

        ascii_text = captcha_to_ascii(captcha_text, font)
        ascii_img = text_to_image(ascii_text, font_size=font_size)
        final_img = resize_and_pad(ascii_img, image_size)

        captcha_text = captcha_text.upper() if captcha_text.islower() else captcha_text.lower()

        filename = f"{captcha_text}.png"
        image_path = os.path.join(images_temp_dir, filename)
        final_img.save(image_path)

        labels[filename] = captcha_text

    # Split filenames into train/val
    filenames = list(labels.keys())
    random.shuffle(filenames)
    split_idx = int(len(filenames) * (1 - train_val_split_ratio))

    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]

    # Create directories for train and val
    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        split_dir = os.path.join(output_dir, split_name)
        images_dir = os.path.join(split_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        split_labels = {}
        for fname in split_files:
            src = os.path.join(images_temp_dir, fname)
            dst = os.path.join(images_dir, fname)
            if not os.path.exists(dst):
                os.link(src, dst)  # Hard link to save space
            split_labels[fname] = labels[fname]

        # Save labels json
        labels_path = os.path.join(split_dir, "labels.json")
        with open(labels_path, "w") as f:
            json.dump(split_labels, f, indent=2)

        print(f"✅ {split_name.capitalize()} set created with {len(split_files)} samples at: {split_dir}")

    # Optionally remove the temp folder if no longer needed
    import shutil
    shutil.rmtree(images_temp_dir)

def main():
    TOTAL_SAMPLES = 10000
    generate_recog_dataset(TOTAL_SAMPLES, base_output_dir)

if __name__ == "__main__":
    main()
