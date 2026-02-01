import os
import csv

def summarise_data(parent_path, clean_path):
    assert os.path.exists(parent_path)
    assert os.path.exists(clean_path)
    
    files = os.listdir(clean_path)
    result = {}
    
    for file in files:
        path = os.path.join(clean_path, file)
        with open(path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = list(next(reader))
            except StopIteration:
                print(f"⚠️ Skipping empty file: {file}")
                continue
            rows = list(reader)
        
        # Clean and trim
        header = header[2:]
        rows = [r[2:] for r in rows if len(r) >= len(header) + 2]

        if not rows:
            print(f"⚠️ No data rows found in {file}, skipping.")
            continue
        
        mean = []
        for i in range(len(header)):
            try:
                # Filter out non-numeric values gracefully
                col = [float(r[i]) for r in rows if r[i].strip() != ""]
                if len(col) == 0:
                    mean.append(0.0)
                    continue
                mean.append(sum(col) / len(col))
            except (ValueError, IndexError) as e:
                print(f"⚠️ Skipping column {header[i]} in {file}: {e}")
                mean.append(0.0)
        
        summary = dict(zip(header, mean))
        summary["Count"] = len(rows)
        result[file.split("_results")[0]] = summary
    
    # Write the summary CSV
    out_path = os.path.join(parent_path, "summary.csv")
    with open(out_path, "w+", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        response_col = None
        if result:
            sample = next(iter(result.values()))
            response_col = next(
                (col for col in sample.keys() if "response" in col.lower()), None
            )

        header_cols = ["Model"]
        if response_col:
            header_cols.append("Avg Response Time (s)")
        header_cols += ["Result", "Similarity", "Count"]
        writer.writerow(header_cols)
        
        for key, value in result.items():
            row = [key]
            if response_col:
                row.append(value.get(response_col, 0.0))
            row.extend([value.get("Result", 0.0), value.get("Similarity", 0.0), value.get("Count", 0)])
            writer.writerow(row)
    
    print(f"✅ Summary written to {out_path}")

def main():
    parent_path = "results/gemini-3/"
    clean_path = os.path.join(parent_path, "clean/")
    summarise_data(parent_path, clean_path)
    
if __name__ == "__main__":
    main()
