import os
import csv

def summarise(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = list(next(reader))
        rows = list(reader)
        
    header = header[2:]
    rows = [r[2:] for r in rows if r]
    mean = []
    
    for i in range(len(header)):
        col = [float(r[i]) for r in rows]
        mean.append(sum(col) / len(col))
        
    return dict(zip(header, mean))

def main():
    clean_path = "results/gemini/clean/"
    parent_path = "results/gemini/"

    files = os.listdir(clean_path)
    result = {}
    
    for file in files:
        summary = summarise(f"{clean_path}/{file}")
        result[file.split("_results")[0]] = summary
        
    with open(f"{parent_path}/summary.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Result", "Similarity", "Similarity (Non Case-Sensitive)"])
        
        for key, value in result.items():
            writer.writerow([key, *value.values()])
    
if __name__ == "__main__":
    main()