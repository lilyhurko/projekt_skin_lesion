import torch
import torch.nn as nn
import torch.optim as optim
from src.model import get_model_for_level
from src.dataset import get_dataloaders
import os

def train_mcml_level(level_name, data_path, epochs=15, batch_size=32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nRozpoczynam trening dla: {level_name.upper()}")
    print(f" Urządzenie: {device}")

    train_loader, test_loader, class_names = get_dataloaders(data_path, batch_size)
    print(f" Klasy: {class_names}")

    model = get_model_for_level(level_name).to(device)
    weights = torch.tensor([1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Acc: {train_acc:.2f}%")

    os.makedirs("models", exist_ok=True)
    save_path = f"models/model_{level_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f" Model zapisany w: {save_path}")

if __name__ == "__main__":
    train_mcml_level("level_3", "data/processed/level_3", epochs=15)