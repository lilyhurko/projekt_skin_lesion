import pandas as pd
import os
import shutil
from tqdm import tqdm

def setup_mcml_data():
    base_path = "data/processed"
    raw_img_path_1 = "data/raw/HAM10000_images_part_1"
    raw_img_path_2 = "data/raw/HAM10000_images_part_2"
    metadata = pd.read_csv("data/raw/HAM10000_metadata.csv")

   
    
    levels = {
        "level_1": {"healthy": ["nv"], "unhealthy": ["mel", "bkl", "bcc", "akiec", "vasc", "df"]},
        "level_2": {"eczema": ["bkl", "df", "vasc"], "melanoma": ["mel", "bcc", "akiec"]},
        "level_3": {"benign": ["akiec"], "malignant": ["mel", "bcc"]} 
        }

    print(" Rozpoczynam pełną segregację danych dla 3 poziomów...")

    for level, mapping in levels.items():
        for class_name, dx_list in mapping.items():
            class_dir = os.path.join(base_path, level, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            subset = metadata[metadata['dx'].isin(dx_list)]
            
            for img_id in tqdm(subset['image_id'], desc=f"Kopiowanie {level}/{class_name}"):
                found = False
                for part in [raw_img_path_1, raw_img_path_2]:
                    src = os.path.join(part, f"{img_id}.jpg")
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(class_dir, f"{img_id}.jpg"))
                        found = True
                        break

    print(" Wszystkie poziomy (1, 2, 3) zostały uzupełnione!")

if __name__ == "__main__":
    setup_mcml_data()