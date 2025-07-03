from PIL import Image
import os

dir_path = "data/ascii-captcha-image-doctr/recog/train/images"
sizes = [Image.open(os.path.join(dir_path, f)).size for f in os.listdir(dir_path) if f.endswith(".png")]
print("Max:", max(sizes), "Min:", min(sizes), "Median:", sorted(sizes)[len(sizes)//2])
