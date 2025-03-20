import os
import csv

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def clean_data(raw_path, clean_path):    
    assert os.path.exists(raw_path)
    assert os.path.exists(clean_path)
    
    files = os.listdir(raw_path)

    for file in files:
        with open(f"{raw_path}/{file}", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
        
        header = data[0]
        data = data[1:]
        
        data = [*filter(lambda x: x, data)]
        data = [[r.strip("\n").encode("ascii", errors="ignore").decode() for r in row] for row in data]
        for i, v in enumerate(data):
            if len(v) != 3:
                print(i, v)
        data = [[row[0], row[1], row[2], int(row[0]==row[2]), similar(row[0], row[2]), similar(row[0].lower(), row[2].lower())] for row in data]
            
        with open(f"{clean_path}/{file}", "w+") as f:
            writer = csv.writer(f)
            writer.writerow(header+["Result", "Similarity", "Similarity (Non Case-Sensitive)"])
            writer.writerows(data)

def main():
    parent_path = "results/gemini-3/"
    raw_path = f"{parent_path}/raw/"
    clean_path = f"{parent_path }/clean/"

    os.makedirs(clean_path, exist_ok=True)
    
    clean_data(raw_path, clean_path)
  
if __name__ == "__main__":
    main()