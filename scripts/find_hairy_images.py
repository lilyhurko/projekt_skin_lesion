import cv2
import numpy as np
import os
import glob

def calculate_hair_density(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    hair_pixels = np.sum(mask == 255)
    total_pixels = img.shape[0] * img.shape[1]
    density = (hair_pixels / total_pixels) * 100
    
    return density

raw_data_path = "data/raw/HAM10000_images_part_*"
images = glob.glob(os.path.join(raw_data_path, "*.jpg"))

results = []
print(f"Rozpoczynam analizę {len(images)} zdjęć... To może chwilę potrwać.")

for img_path in images:
    density = calculate_hair_density(img_path)
    if density > 5: 
        results.append((os.path.basename(img_path), density))

results.sort(key=lambda x: x[1], reverse=True)

print("\nTOP 10 NAJBARDZIEJ OWŁOSIONYCH ZDJĘĆ:")
print(f"{'Image ID':<20} | {'Hair Density (%)':<20}")
print("-" * 45)
for i in range(min(10, len(results))):
    print(f"{results[i][0]:<20} | {results[i][1]:.2f}%")