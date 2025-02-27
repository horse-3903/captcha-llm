import os
import random

from tqdm import tqdm

from util import test_captcha, save_result

def main():
    prompt = 1 # 1 or 2
    
    data = os.listdir("data")
    model_lst = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    random.shuffle(model_lst)

    image_file_lst = random.sample(data, 500)
    image_file_lst = ["data/" + file_name for file_name in image_file_lst]
    
    for model in model_lst:
        print(f"Testing model: {model}")
        for image_file_name in tqdm(image_file_lst):
            actual_solution, predicted_solution = test_captcha(prompt, image_file_name, model)
            
            if predicted_solution == "ERROR_RATE_LIMIT":
                continue
            
            save_result(f"results/openai-{prompt}/", model, actual_solution, predicted_solution)

if __name__ == "__main__":
    main()