import os
import csv
from tqdm import tqdm
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def clean_data(raw_path, clean_path):    
    assert os.path.exists(raw_path)
    assert os.path.exists(clean_path)
    
    files = os.listdir(raw_path)

    for file in files:
        print(f"Processing {file}...")
        with open(f"{raw_path}/{file}", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
        
        header = data[0]
        data = data[1:]
        
        # Clean rows (remove empty, force ASCII)
        cleaned = [
            [cell.strip().encode("ascii", errors="ignore").decode() for cell in row]
            for row in data if row
        ]

        new_data = []
        for row in tqdm(cleaned, desc=f"Cleaning {file}", ncols=80):
            try:
                if len(row) >= 2:
                    actual, predicted = row[0], row[1]
                elif len(row) == 3:
                    actual, predicted = row[0], row[2]
                else:
                    continue

                # sim = similar(actual, predicted)
                sim_lower = similar(actual.lower(), predicted.lower())
                result = int(actual == predicted)
                new_data.append([*row, result, sim_lower])
            except Exception:
                continue
        
        os.makedirs(clean_path, exist_ok=True)
        with open(f"{clean_path}/{file}", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header + ["Result", "Similarity"])
            writer.writerows(new_data)

def main():
    parent_path = "results/gemini-3/"
    raw_path = f"{parent_path}/raw/"
    clean_path = f"{parent_path}/clean/"

    os.makedirs(clean_path, exist_ok=True)
    clean_data(raw_path, clean_path)
  
if __name__ == "__main__":
    main()
