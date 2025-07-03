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
from util import text_to_image

# Configuration
# FONT_LIST = [
#     'charact1', 'charact2', 'charact3', 'charact4', 'charact5', 'charact6', 
#     'characte', 'ascii___', 'bubble__', 'char2___', 'com_sen_', 'demo_1__', 
#     'ebbs_1__', 'ebbs_2__', 'e__fist_', 'fbr2____', 'filter', 'fraktur', 
#     'georgi16', 'georgia11', 'gothic__', 'new_asci', 'nscript', 'nvscript', 
#     'o8', 'radical_', 'roman', 'roman___', 'space_op', 't__of_ap', 'ucf_fan_', 
#     'utopiab', 'utopiabi', 'xhelvb', 'xhelvbi', 'xsansbi', 'xsbookb', 
#     'xsbookbi', 'xtimes', 'xttyb'
# ]

FONT_LIST = ['georgi16', 'georgia11']

# Dataset paths
BASE_OUTPUT_DIR = "data/ascii-captcha-image-doctr"
RECOG_IMAGE_SIZE = (128, 64)  # Wider canvas for recognition
DETECT_CANVAS_SIZE = (900, 600)  # For detection
TRAIN_VAL_SPLIT_RATIO = 0.2  # 20% for validation

def random_captcha_text(length=5, uppercase=False):
    chars = string.ascii_uppercase + string.digits if uppercase else string.ascii_lowercase + string.digits
    return ''.join(random.choices(chars, k=length))

def captcha_to_ascii(captcha: str, font: str):
    return pyfiglet.figlet_format(captcha, font=font)

def hash_file_sha256(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def resize_and_pad(image, size, pad_color=(255, 255, 255)):
    """For recognition dataset"""
    image.thumbnail(size, Image.Resampling.NEAREST)
    new_img = Image.new("RGB", size, pad_color)
    offset_x = (size[0] - image.width) // 2
    offset_y = (size[1] - image.height) // 2
    new_img.paste(image, (offset_x, offset_y))
    return new_img

def save_detection_image_and_label(captcha_text, ascii_img, image_path, canvas_size):
    """For detection dataset"""
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

def generate_recognition_dataset(total_samples, output_dir):
    """Generate dataset for text recognition"""
    output_dir = os.path.join(output_dir, "recog")
    os.makedirs(output_dir, exist_ok=True)

    labels = {}
    used_texts = set()

    images_temp_dir = os.path.join(output_dir, "all_images")
    os.makedirs(images_temp_dir, exist_ok=True)

    for _ in tqdm(range(total_samples), desc="Generating recognition samples"):
        captcha_text = random_captcha_text()
        while captcha_text in used_texts:
            captcha_text = random_captcha_text()
        used_texts.add(captcha_text)

        font = random.choice(FONT_LIST)
        font_size = 8

        ascii_text = captcha_to_ascii(captcha_text, font)
        ascii_img = text_to_image(ascii_text, font_size=font_size)
        final_img = resize_and_pad(ascii_img, RECOG_IMAGE_SIZE)

        # Randomly decide to uppercase/lowercase for variety
        captcha_text = captcha_text.upper() if random.random() > 0.5 else captcha_text.lower()

        filename = f"{captcha_text}.png"
        image_path = os.path.join(images_temp_dir, filename)
        final_img.save(image_path)

        labels[filename] = captcha_text

    # Split into train/val
    split_and_save_dataset(labels, images_temp_dir, output_dir, "recognition")
    return labels

def generate_detection_dataset(total_samples, output_dir):
    """Generate dataset for text detection"""
    output_dir = os.path.join(output_dir, "detect")
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    existing_files = set(os.path.basename(f) for f in glob.glob(os.path.join(image_dir, "*.png")))
    labels = {}

    num_existing = len(existing_files)
    num_to_generate = total_samples - num_existing
    print(f"Found {num_existing} samples. Generating {num_to_generate} more...")

    used_captchas = set()
    for _ in tqdm(range(num_to_generate), desc="Generating detection samples"):
        font = random.choice(FONT_LIST)
        font_size = random.randint(16, 24)
        captcha_text = random_captcha_text(length=10, uppercase=True)

        while captcha_text in used_captchas:
            captcha_text = random_captcha_text(length=10, uppercase=True)
        used_captchas.add(captcha_text)

        ascii_text = captcha_to_ascii(captcha_text, font)
        ascii_img = text_to_image(ascii_text, font_size=font_size)

        filename = f"{captcha_text}_{font}_{font_size}.png"
        image_path = os.path.join(image_dir, filename)

        img_dims, polygon, sha256 = save_detection_image_and_label(
            captcha_text, ascii_img, image_path, DETECT_CANVAS_SIZE
        )
        labels[filename] = {
            "img_dimensions": img_dims,
            "img_hash": sha256,
            "polygons": [polygon],
            "font": font,
            "font_size": font_size
        }

    print(f"✅ Detection dataset ready at: {output_dir}")
    split_and_save_dataset(labels, image_dir, output_dir, "detection")
    return labels

def split_and_save_dataset(labels, images_dir, output_dir, dataset_type):
    """Split dataset into train/val and save with appropriate structure"""
    filenames = list(labels.keys())
    random.shuffle(filenames)
    split_idx = int(len(filenames) * (1 - TRAIN_VAL_SPLIT_RATIO))

    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        split_dir = os.path.join(output_dir, split_name)
        
        if dataset_type == "recognition":
            images_dest_dir = os.path.join(split_dir, "images")
            os.makedirs(images_dest_dir, exist_ok=True)
            
            split_labels = {}
            for fname in split_files:
                src = os.path.join(images_dir, fname)
                dst = os.path.join(images_dest_dir, fname)
                if not os.path.exists(dst):
                    os.link(src, dst)  # Hard link to save space
                split_labels[fname] = labels[fname]
                
            # Save labels json
            labels_path = os.path.join(split_dir, "labels.json")
            with open(labels_path, "w") as f:
                json.dump(split_labels, f, indent=2)
                
        elif dataset_type == "detection":
            images_dest_dir = os.path.join(split_dir, "images")
            os.makedirs(images_dest_dir, exist_ok=True)
            
            split_labels = {}
            for fname in split_files:
                src = os.path.join(images_dir, fname)
                dst = os.path.join(images_dest_dir, fname)
                if not os.path.exists(dst):
                    os.link(src, dst)
                split_labels[fname] = labels[fname]
                
            labels_path = os.path.join(split_dir, "labels.json")
            with open(labels_path, "w") as f:
                json.dump(split_labels, f, indent=2)

        print(f"✅ {split_name.capitalize()} set created with {len(split_files)} samples at: {split_dir}")

    # Clean up temporary directory for recognition dataset
    if dataset_type == "recognition":
        import shutil
        shutil.rmtree(images_dir)

def main():
    # Generate datasets
    print("Generating recognition dataset...")
    generate_recognition_dataset(10000, BASE_OUTPUT_DIR)
    
    print("\nGenerating detection dataset...")
    generate_detection_dataset(10000, BASE_OUTPUT_DIR)

if __name__ == "__main__":
    main()