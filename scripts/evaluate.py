import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.model import get_model_for_level
from src.dataset import get_dataloaders
from src.gradcam import GradCAM
from sklearn.metrics import confusion_matrix

def evaluate_and_visualize(level_name, data_path, model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, test_loader, class_names = get_dataloaders(data_path)
    
    model = get_model_for_level(level_name).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) 
    specificity = tn / (tn + fp)

    print(f"\nWYNIKI DLA {level_name.upper()}:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Sensitivity (Czułość): {sensitivity*100:.2f}%")
    print(f"Specificity (Swoistość): {specificity*100:.2f}%")


    target_layer = model.network.features[8]
    grad_cam = GradCAM(model, target_layer)
    
    img_batch, label_batch = next(iter(test_loader))
    input_tensor = img_batch[0].unsqueeze(0).to(device)
    
    mask = grad_cam.generate(input_tensor)
    
    img_np = img_batch[0].permute(1, 2, 0).numpy()
    img_np = (img_np * 0.229) + 0.485 
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.uint8(255 * img_np), 0.6, heatmap, 0.4, 0)

    plt.imsave("reports/figures/gradcam_result_level3.png", overlay)
    print("Mapa Grad-CAM została zapisana w folderze reports/figures/")

if __name__ == "__main__":
    evaluate_and_visualize("level_2", "data/processed/level_3", "models/model_level_3.pth")