import os
import csv

# Define the base directory
base_dir = "./results/part-1/"

# Iterate through every subdirectory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)

    # Ensure it's a directory
    if os.path.isdir(subdir_path):
        raw_path = os.path.join(subdir_path, "raw")

        # Ensure the "raw" subdirectory exists
        if os.path.isdir(raw_path):
            print(raw_path)
            # Iterate through all CSV files
            for file in os.listdir(raw_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(raw_path, file)
                    print(file_path)

                    # Read the CSV manually to handle inconsistencies
                    with open(file_path, "r", newline="", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = [row[:2] + row[3:] if len(row) > 2 else row for row in reader]

                    # Write back the modified CSV
                    with open(file_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)

print("Processing complete!")
