import os
import json
import cv2
import numpy as np

def display_polygons(label_json_path, image_root_dir, max_images=10):
    # Load labels
    with open(label_json_path, 'r') as f:
        labels = json.load(f)

    for idx, (filename, meta) in enumerate(labels.items()):
        image_path = os.path.join(image_root_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Failed to load {image_path}")
            continue

        # Draw polygons
        polygons = meta['polygons']
        if isinstance(polygons, dict):  # multiple classes
            all_polys = [pt for polylist in polygons.values() for pt in polylist]
        else:  # single class
            all_polys = polygons

        for poly in all_polys:
            pts = cv2.convexHull(np.array(poly, dtype=np.int32)).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Optional: display image dimensions and hash
        img_dims = meta.get("img_dimensions")
        img_hash = meta.get("img_hash")
        cv2.putText(image, f"{filename}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if img_dims:
            cv2.putText(image, f"Size: {img_dims}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Polygons", image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

        if idx + 1 >= max_images:
            break

    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    split_dir = "data/ascii-captcha-image-doctr/detect/train"  # or "val"
    display_polygons(
        label_json_path=os.path.join(split_dir, "labels.json"),
        image_root_dir=os.path.join(split_dir, "images"),
        max_images=20
    )
