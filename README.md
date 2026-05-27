# Multi-Class Multi-Level (MCML) Skin Lesion Classification System

Projekt realizuje nowoczesny, wielopoziomowy system wspomagania decyzji klinicznych (CAD) w diagnostyce zmian skórnych. Architektura systemu bazuje na koncepcji hierarchicznej "dziel i zwyciężaj" (Divide and Conquer) opisanej w pracy Hameed et al. (2020), jednak została zmodernizowana poprzez zastosowanie sieci **EfficientNet-B0**, algorytmów automatycznego inpaintingu owłosienia oraz modułów wyjaśnialnej sztucznej inteligencji (**XAI / Grad-CAM**).

---

##  Główne Cechy Systemu

- **Hierarchiczna Klasyfikacja MCML:** Zamiast płaskiej klasyfikacji wieloklasowej, problem został podzielony na trzy wyspecjalizowane poziomy decyzyjne, co pozwala na optymalizację metryk krytycznych pod kątem bezpieczeństwa onkologicznego.
- **Zaawansowany Preprocessing:** Automatyczna eliminacja artefaktów (owłosienia) za pomocą algorytmu inpaintingu Telea (OpenCV) oraz dynamiczne usuwanie czarnych ramek aparatów i kółek pomocniczych.
- **Bezpieczeństwo Pro-Kliniczne:** Zastosowanie wag klasowych (Class Weights 1:5) w celu maksymalizacji czułości (*Sensitivity*) dla stanów złośliwych i eliminacji błędów typu *False Negative*.
- **Wyjaśnialność Modelu (XAI):** Integracja map aktywacji Grad-CAM w celu wizualnej weryfikacji obszarów, na których sieć neuronowa skupia uwagę podczas wnioskowania.

---

## Architektura Systemu (MCML)

Działanie systemu podzielone jest na 3 niezależne logicznie poziomy (*Levels*):

1. **Level 1 (Healthy vs Unhealthy):** Bramka bezpieczeństwa odrzucająca zdrową skórę (`nv` - znamiona melanocytowe) i kierująca wszelkie patologie do dalszej analizy.
2. **Level 2 (Eczema vs Melanoma-type):** Różnicowanie pomiędzy stanami zapalnymi (typu egzema) a zmianami o charakterze barwnikowym/nowotworowym.
3. **Level 3 (Benign vs Malignant):** Poziom ekspercki różnicujący stopień złośliwości. Rak podstawnokomórkowy (**BCC**) oraz rogowiec słoneczny (**AKIEC**) zostały sklasyfikowane jako `Malignant` (wraz z czerniakiem `mel`), co podnosi czujność diagnostyczną systemu.

---

##  Zbiór Danych i Struktura Projektu

Projekt wykorzystuje zbiór **HAM10000** (10 015 obrazów dermoskopowych). Ze względów licencyjnych oraz z uwagi na rozmiar plików, surowe dane nie są dołączone do repozytorium.

---

### Wymagana struktura katalogów:
```text
projekt_skin_lesion/
├── data/
│   ├── raw/
│   │   └── HAM10000_images_part_1/    # Tutaj umieść pobrane zdjęcia (.jpg)
│   └── processed/                     # Generowane automatycznie przez skrypty
├── models/                            # Tutaj zostaną zapisane wagi (.pth)
├── scripts/
│   ├── final_diagnostic.py            # Skrypt ostatecznej diagnozy pacjenta
│   ├── clean_all_data.py              # Pipeline inpaintingu i czyszczenia
│   ├── setup_project.py               # Przygotowanie struktury hierarchicznej
│   └── train.py                       # Skrypt treningowy dla poziomów MCML
├── .gitignore
└── README.md
```
---

##  Instrukcja Wdrożenia i Uruchomienia
### 1. Przygotowanie Środowiska Wirtualnego
Projekt wymaga Pythona w wersji 3.13 (zalecane środowisko wirtualne izolowane od globalnych pakietów Conda/base):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Uwaga: W przypadku problemów z instalacją pakietu tqdm wewnątrz środowiska wirtualnego, należy użyć instalacji celowanej:

### 2. Przygotowanie Danych i Mapowanie Hierarchiczne
Pobierz zdjęcia HAM10000 i umieść je w folderze `data/raw/HAM10000_images_part_1`/`, a następnie uruchom skrypt segregujący je do struktur MCML:
```bash
python scripts/setup_project.py
```
### 3. Pipeline Czyszczenia (Inpainting)
Uruchom proces automatycznego usuwania włosów oraz artefaktów tła algorytmem Telea:
```bash
PYTHONPATH=. python scripts/clean_all_data.py
```
### 4. Trening Modelu
Wytrenuj poszczególne poziomy hierarchiczne (skrypt automatycznie wykorzystuje akcelerację sprzętową, np. Apple Silicon MPS / CUDA):
```bash
PYTHONPATH=. python scripts/train.py
```
### 5. Diagnostyka i Testowanie (Wnioskowanie)
Aby przeprowadzić pełną, wielopoziomową diagnozę na pojedynczym surowym zdjęciu, użyj skryptu `final_diagnostic.py`:
```bash
PYTHONPATH=. python scripts/final_diagnostic.py data/raw/HAM10000_images_part_1/ISIC_0024331.jpg
```
---

## Wyniki i Metryki Badawcze

Wdrożenie wag klasowych oraz zaawansowanego preprocessingu pozwoliło na eliminację zjawiska biasu (płaskiego dopasowania) w stronę klasy dominującej i zmusiło model do realnej nauki struktur dermoskopowych.

| Poziom | Zadanie Klasyfikacji | Dokładność (Accuracy) | Czułość (Sensitivity) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Level 1** | Healthy vs Unhealthy | 73.01% | **90.42%** | ✅ Stabilny |
| **Level 2** | Eczema vs Melanoma-type | 82.28% | **90.78%** | ✅ Stabilny |
| **Level 3** | Benign vs Malignant | 87.59% | **66.48%** | ✅ Stabilny |

###  Wnioski z Metryk
- **Wysoka czułość na poziomie Level 1 (90.42%) oraz Level 2 (90.78%)** gwarantuje, że system skutecznie wychwytuje anomalie i nie pomija wczesnych stadiów nowotworowych, co ma kluczowe znaczenie w badaniach przesiewowych.
- **Redefinicja Poziomu 3:** Przeniesienie raka podstawnokomórkowego (BCC) do grupy `Malignant` na poziomie Level 3 wymusiło na modelu znacznie rygorystyczniejszą ocenę tekstur oraz asymetrii zmiany, podnosząc onkologiczną czujność systemu.

---

##  Wyjaśnialna AI (XAI) i Walidacja

W celu zapewnienia pełnej transparentności decyzji medycznych, system został zintegrowany z mapami aktywacji **Grad-CAM** (Gradient-weighted Class Activation Mapping).

Podczas gdy model bazowy (*Baseline*) wykazywał silną tendencję do nadmiarowej aktywacji filtrów konwolucyjnych na artefaktach owłosienia i krawędziach zdjęć (osiągając przez to niską czułość kliniczną), zastosowanie inpaintingu wymusiło całkowite przesunięcie uwagi sieci. 

W obecnej wersji najwyższe aktywacje wag sieci **EfficientNet-B0** koncentrują się ściśle w centralnych obszarach zmian chorobowych oraz na ich bezpośrednich granicach. Odzwierciedla to rzeczywiste kryteria medyczne stosowane przez dermatologów podczas oceny struktur barwnikowych.