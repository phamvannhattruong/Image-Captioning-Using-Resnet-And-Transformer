import csv
import json
import os

def convert_csv_to_json(csv_file, output_file):
    captions_dict = {}

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f,delimiter="|")
        for row in reader:
            image = row["image_name"]
            caption = row[" comment"]

            if image not in captions_dict:
                captions_dict[image] = []
            captions_dict[image].append(caption)

    images = []
    for image, captions in captions_dict.items():
        images.append({
            "file_name": image,
            "comment": captions,
        })

    json_data = {"images": images}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"Đã tạo file JSON: {output_file}")


if __name__ == "__main__":
    csv_file = "flickr30k_images/results.csv"
    output_file = "configure/captions.json"
    convert_csv_to_json(csv_file, output_file)
