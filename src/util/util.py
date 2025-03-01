import os

from clean import clean_data
from summarise import summarise_data

def clean_and_summarise(parent_path):
    raw_path = f"{parent_path}/raw/"
    clean_path = f"{parent_path}/clean/"
    
    os.makedirs(clean_path, exist_ok=True)
    
    clean_data(raw_path, clean_path)
    summarise_data(parent_path, clean_path)
    
def main():
    parent_path = "results/openai-2/"
    
    clean_and_summarise(parent_path)
    
if __name__ == "__main__":
    main()