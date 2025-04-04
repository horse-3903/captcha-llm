import os
import csv

def summarise_data(parent_path, clean_path):
    assert os.path.exists(parent_path)
    assert os.path.exists(clean_path)
    
    files = os.listdir(clean_path)
    result = {}
    
    for file in files:
        path = f"{clean_path}/{file}"
        with open(path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            header = list(next(reader))
            rows = list(reader)
            
        header = header[2:]
        rows = [r[2:] for r in rows if r]
        mean = []
        
        for i in range(len(header)):
            col = [float(r[i]) for r in rows]
            mean.append(sum(col) / len(col))
            
        summary = dict(zip(header, mean))
        
        result[file.split("_results")[0]] = summary
        
    with open(f"{parent_path}/summary.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Result", "Similarity", "Similarity (Non Case-Sensitive)"])
        
        for key, value in result.items():
            writer.writerow([key, *value.values()])
            
def main():
    parent_path = "results/gemini-3/"
    clean_path = f"{parent_path}/clean/"
    
    summarise_data(parent_path, clean_path)
    
if __name__ == "__main__":
    main()