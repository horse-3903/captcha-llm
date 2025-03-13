import os
import csv

def main():
    files = sorted(os.listdir("results/gemini"))

    for file in files:
        file = "results/gemini/" + file
        
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
            print(len(data))
            
if __name__ == "__main__":
    main()