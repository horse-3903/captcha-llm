import re
import os
import glob
import random
from tqdm import tqdm
from util import text_to_image

data_path = "data/ascii-captcha/"
output_path = "data/ascii-captcha-misc/"
data_path_list = glob.glob(os.path.join(data_path, "**", "*.txt"), recursive=True)

selected = random.sample(data_path_list, 20)

for path in tqdm(selected):
    text_data = open(path).read()
    img_data = text_to_image(text_data)

    value = re.search(r'(?<=\\)[A-Za-z0-9]+(?=\.txt$)', path).group() # type: ignore
    img_path = os.path.join(output_path, f"{value}.jpg")

    img_data.save(img_path)