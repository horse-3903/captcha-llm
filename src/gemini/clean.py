import os
import csv

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():
    raw_path = "results/gemini/raw/"
    clean_path = "results/gemini/clean/"

    files = os.listdir(raw_path)

    for file in files:
        with open(f"{raw_path}/{file}", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
        
        header = data[0]
        data = data[1:]
        
        data = [*filter(lambda x: x, data)]
        data = [[r.strip("\n").encode("ascii", errors="ignore").decode() for r in row] for row in data]
        data = [[row[0], row[1], int(row[0]==row[1]), similar(row[0], row[1]), similar(row[0].lower(), row[1].lower())] for row in data]
            
        with open(f"{clean_path}/{file}", "w+") as f:
            writer = csv.writer(f)
            writer.writerow(header+["Similarity", "Similarity (Non Case-Sensitive)"])
            writer.writerows(data)
            
if __name__ == "__main__":
    main()