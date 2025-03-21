import os
import csv

import random
from tqdm import tqdm

from util import crosscheck_captcha, save_result

def main():
    prompt = 1 # 1, 2 or 3
    iteration = 1
    
    model_lst = ["gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b"] # , "gemini-1.5-pro", "gemini-2.0-flash"]
    random.shuffle(model_lst)
    
    data = []
    final_result_dir = []
    
    res_par_dir = "./results/part-2/gemini-3-1/clean"
    result_dir = os.listdir(res_par_dir)
    
    final_result_dir = [f"{res_par_dir}/{d}" for d in result_dir]
        
    for res in final_result_dir:
        with open(res) as f:
            reader = csv.reader(f)
            headers = next(reader)
            data += [*reader]
            
    data = [*filter(lambda x: x, data)]
    data = random.sample(data, 500)

    for model in model_lst:
        for actual_solution, _, pred_solution, _, _, _ in tqdm(data):
            if not os.path.exists(f"./data/{actual_solution}.png"):
                continue
            
            res = crosscheck_captcha(3, actual_solution, pred_solution, model)
            
            if not res:
                continue
            
            save_result(f"./results/part-2/gemini-{prompt}-{iteration}/raw/", f"{model}_results.csv", data=[res])

if __name__ == "__main__":
    main()