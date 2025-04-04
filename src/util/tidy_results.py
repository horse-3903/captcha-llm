import os
import csv

from clean_results import clean_data
from sum_results import summarise_data

def tidy_data(parent_path):
    raw_path = f"{parent_path}/raw/"
    clean_path = f"{parent_path}/clean/"
    
    os.makedirs(clean_path, exist_ok=True)
    
    clean_data(raw_path, clean_path)
    summarise_data(parent_path, clean_path)
    
def overall_summary(parent_path):
    file_path = f"{parent_path}/summary.csv"
    data = []
    header = []
    
    for provider in os.listdir(parent_path):
        if not os.path.exists(os.path.join(parent_path, provider, "summary.csv")):
            continue
        
        with open(os.path.join(parent_path, provider, "summary.csv")) as f:
            reader = csv.reader(f)
            header = next(reader)
            
            data.extend([*reader])
        
    with open(file_path, "w+") as f:
        writer = csv.writer(f)
        writer.writerows([header, *data])
            
def combine_summary(parent_path):
    file_path = f"{parent_path}/summary.csv"
    
    with open(file_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [*reader]
        
    data = [row for row in data if row]
    data = [[row[0], *map(float, row[1:])] for row in data]
        
    new_data = {}
    final_data = []
    
    for model, result, sim, sim_ncs in data:
        if model not in new_data:
            new_data[model] = (result, sim, sim_ncs, 1)
        else:
            new_data[model] = (result + new_data[model][0], sim + new_data[model][1], sim_ncs + new_data[model][2],  + new_data[model][3] + 1)
    
    for model, res in new_data.items():
        final_data.append((model, *[float(i)/float(res[3]) for i in res[:3]]))
    
    with open(file_path, "w+") as f:
        writer = csv.writer(f)
        writer.writerows([header, *sorted(final_data)])
    
def main():
    parent_path_lst = ["results/new/part-1/prompt-1/", "results/new/part-1/prompt-2/", "results/new/part-1/prompt-3/"]
    
    # for parent_path in parent_path_lst:
        # for provider in os.listdir(parent_path):            
            # if os.listdir(os.path.join(parent_path, provider)):
            #     print(os.path.join(parent_path, provider))
            #     tidy_data(os.path.join(parent_path, provider))
        
        # overall_summary(parent_path)
    # overall_summary("results/new/part-1/")
    # combine_summary("results/new/part-1/")
    
if __name__ == "__main__":
    main()