import os
import glob
import cv2
from src.preprocessing import remove_hair
from tqdm import tqdm 

def clean_dataset(base_path):
    image_paths = glob.glob(os.path.join(base_path, "**/*.jpg"), recursive=True)
    print(f"🧹 Rozpoczynam czyszczenie {len(image_paths)} obrazów...")

    for img_path in tqdm(image_paths):
        cleaned_img = remove_hair(img_path)
        
        if cleaned_img is not None:
            cleaned_bgr = cv2.cvtColor(cleaned_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, cleaned_bgr)

if __name__ == "__main__":
    clean_dataset("data/processed")
    print("✅ Wszystkie zdjęcia zostały wyczyszczone i są gotowe do treningu!")