import os
import csv

csv.field_size_limit(2**31 - 1)

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
            print(data)
        
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
    count_idx = header.index("Count") if "Count" in header else None
    numeric_indices = [i for i in range(1, len(header)) if i != count_idx]

    new_data = {}

    for row in data:
        model = row[0]
        try:
            weight = float(row[count_idx]) if count_idx is not None else 1.0
        except (ValueError, TypeError, IndexError):
            weight = 1.0

        if model not in new_data:
            new_data[model] = {"sums": {i: 0.0 for i in numeric_indices}, "count": 0.0}

        for i in numeric_indices:
            try:
                new_data[model]["sums"][i] += float(row[i]) * weight
            except (ValueError, TypeError, IndexError):
                continue

        new_data[model]["count"] += weight

    final_data = []
    for model, agg in new_data.items():
        count = agg["count"]
        row = [model]
        for i in range(1, len(header)):
            if i == count_idx:
                row.append(count)
            else:
                avg = agg["sums"][i] / count if count else 0.0
                row.append(avg)
        final_data.append(row)

    with open(file_path, "w+") as f:
        writer = csv.writer(f)
        writer.writerows([header, *sorted(final_data)])
    
def main():
    # parent_path_lst = ["results/new/part-1/prompt-1/", "results/new/part-1/prompt-2/", "results/new/part-1/prompt-3/"]
    
    # for parent_path in parent_path_lst:
    #     for provider in os.listdir(parent_path):            
    #         if os.listdir(os.path.join(parent_path, provider)):
    #             print(os.path.join(parent_path, provider))
    #             tidy_data(os.path.join(parent_path, provider))

    # parent_path = "results/unified/image"
    
    tidy_data("results/ascii-final-1/image")
    tidy_data("results/ascii-final-1/text")
    
    # summarise_data(parent_path, f"{parent_path}/clean")
    # combine_summary(parent_path)
    
if __name__ == "__main__":
    main()
