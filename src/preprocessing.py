import cv2
import numpy as np

def remove_hair(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    

    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
   
    dst = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
    
    final_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    
    return final_img

def save_comparison(original_path, output_path):
    import matplotlib.pyplot as plt
    
    original = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
    cleaned = remove_hair(original_path)
    
    if cleaned is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Oryginał (z artefaktami)")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cleaned)
        plt.title("Po preprocessingu (Inpainting)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Porównanie zapisane w: {output_path}")