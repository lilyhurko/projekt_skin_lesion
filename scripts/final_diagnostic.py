import torch
import os
import sys
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.getcwd())

from src.model import get_model_for_level
from src.preprocessing import remove_hair

def load_model(level, device):
    """Pomocnicza funkcja do ładowania wag modelu"""
    model = get_model_for_level(level).to(device)
    model_path = f"models/model_{level}.pth"
    if not os.path.exists(model_path):
        print(f"BŁĄD: Nie znaleziono pliku {model_path}!")
        return None
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_mcml_diagnostic(img_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n--- 🏥 ROZPOCZYNAM DIAGNOSTYKĘ MCML ---")
    print(f"📁 Plik: {os.path.basename(img_path)}")

    # 1. Preprocessing (Twój as w rękawie - Inpainting)
    cleaned_img = remove_hair(img_path)
    if cleaned_img is None:
        return "Błąd wczytywania zdjęcia."
    
    img_pil = Image.fromarray(cleaned_img)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # LEVEL 1: Healthy vs Unhealthy
        m1 = load_model("level_1", device)
        out1 = torch.softmax(m1(input_tensor), dim=1)
        if torch.argmax(out1) == 0:
            return f"WYNIK: Zdrowa skóra / Znamiona łagodne (Pewność: {out1[0][0]*100:.2f}%)"

        # LEVEL 2: Eczema vs Melanoma-type
        m2 = load_model("level_2", device)
        out2 = torch.softmax(m2(input_tensor), dim=1)
        if torch.argmax(out2) == 0:
            return f" WYNIK: Zmiana zapalna / Eczema (Pewność: {out2[0][0]*100:.2f}%)"

        # LEVEL 3: Benign vs Malignant
        m3 = load_model("level_3", device)
        out3 = torch.softmax(m3(input_tensor), dim=1)
        if torch.argmax(out3) == 1:
            return f" WYNIK KRYTYCZNY: Podejrzenie CZERNIAKA (Pewność: {out3[0][1]*100:.2f}%)"
        else:
            return f"ℹ WYNIK: Zmiana barwnikowa łagodna (Pewność: {out3[0][0]*100:.2f}%)"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        if os.path.exists(test_path):
            wynik = run_mcml_diagnostic(test_path)
            print(f"\n{wynik}")
        else:
            print(f" Nie znaleziono pliku: {test_path}")
    else:
        print(" Użycie: PYTHONPATH=. python scripts/final_diagnostic.py <sciezka_do_zdjecia>")