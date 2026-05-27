import os
import sys
# Dodajemy ścieżkę do sys.path, żeby na pewno widział folder src
sys.path.append(os.getcwd())

try:
    from src.preprocessing import save_comparison
    print("✅ Zaimportowano moduły poprawnie.")
except ImportError as e:
    print(f"❌ Błąd importu: {e}")
    exit()

# 1. ID zdjęcia
img_id = "ISIC_0032214" 

# 2. Sprawdzanie ścieżek
path_part1 = f"data/raw/HAM10000_images_part_1/{img_id}.jpg"
path_part2 = f"data/raw/HAM10000_images_part_2/{img_id}.jpg"

original_path = ""
if os.path.exists(path_part1):
    original_path = path_part1
    print(f"📂 Znaleziono zdjęcie w part_1: {path_part1}")
elif os.path.exists(path_part2):
    original_path = path_part2
    print(f"📂 Znaleziono zdjęcie w part_2: {path_part2}")
else:
    print(f"❌ BŁĄD: Nie znaleziono pliku {img_id}.jpg w data/raw!")
    # Wyświetlmy co jest w folderze, żeby sprawdzić literówki
    print("Zawartość part_1 (pierwsze 5 plików):", os.listdir("data/raw/HAM10000_images_part_1")[:5])
    exit()

# 3. Tworzenie folderu reports/figures jeśli go nie ma
os.makedirs("reports/figures", exist_ok=True)

output_path = "reports/figures/preprocessing_result.png"

print(f"⌛ Rozpoczynam przetwarzanie i zapisywanie do {output_path}...")
save_comparison(original_path, output_path)
print("🏁 Skrypt zakończył działanie.")