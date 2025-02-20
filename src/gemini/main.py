import os
import random

from tqdm import tqdm

from util import test_captcha, save_result

def main():
    data = os.listdir("data")
    model_lst = ["gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

    image_file_lst = random.sample(data, 500)
    image_file_lst = ["data/" + file_name for file_name in image_file_lst]
    
    for model in model_lst:
        print(f"Testing model: {model}")
        for image_file_name in tqdm(image_file_lst):
            actual_solution, predicted_solution = test_captcha(image_file_name, model)
            
            if predicted_solution == "ERROR_RATE_LIMIT":
                continue
            
            save_result("results/gemini/", model, actual_solution, predicted_solution)

if __name__ == "__main__":
    main()