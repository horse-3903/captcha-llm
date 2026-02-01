import os
import torch
from transformers import AutoModel, AutoTokenizer

image_file = 'data/ascii-captcha-misc/QNRINV1DL7.jpg'

output_path = 'results/deepseek-ocr'
output_path_list = os.listdir(output_path)
output_path_final = os.path.join(output_path, f"test-{len(output_path_list)+1}")
os.makedirs(output_path_final, exist_ok=True)

prompt = (
    "<image>\n"
    "Read this ASCII-art CAPTCHA made of '#' symbols. "
    "Identify each cluster as a character and combine them left to right.\n"
    "Output only the decoded 10-character text.\n"
)

with open(os.path.join(output_path_final, "prompt.txt"), "w+") as f:
    f.write(prompt)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name='deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# infer(self, tokenizer, prompt='', image_file='', output_path=' ', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False):

# Tiny: base_size=512, image_size=512, crop_mode=False
# Small: base_size=640, image_size=640, crop_mode=False
# Base: base_size=1024, image_size=1024, crop_mode=False
# Large: base_size=1280, image_size=1280, crop_mode=False

# Gundam: base_size=1024, image_size=640, crop_mode=True

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path=output_path_final, base_size=1024, image_size=640, crop_mode=True, save_results=True, test_compress=True)
