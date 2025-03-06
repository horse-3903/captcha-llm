import os
import random

from tqdm import tqdm

from util import test_captcha, save_result

def main():
    prompt = 2 # 1, 2 or 3
    
    data = os.listdir("data")
    model_lst = ["meta-llama/llama-3.2-11b-vision-instruct:free", "meta-llama/llama-3.2-90b-vision-instruct:free"]
    random.shuffle(model_lst)
    
    image_file_lst = random.sample(data, 500)
    image_file_lst = ["data/" + file_name for file_name in image_file_lst]
    
    for model in model_lst:
        print(f"Testing model: {model}")
        for image_file_name in tqdm(image_file_lst):
            actual_solution, predicted_solution = test_captcha(prompt, image_file_name, model)
            
            if predicted_solution == "ERROR_RATE_LIMIT":
                continue
            
            save_result(f"results/llama-{prompt}/raw", model, actual_solution, predicted_solution)

if __name__ == "__main__":
    main()