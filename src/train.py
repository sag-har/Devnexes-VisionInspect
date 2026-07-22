import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from prepare_data import audit_and_split_data
from dataset import VisionInspectDataset, get_transforms
from model import DefectClassificationModel
import os

def train_pipeline(epochs=5, batch_size=32, lr=0.001):
    # 1. Fetch data split dictionaries
    train_records, val_records, _ = audit_and_split_data()
    
    if not train_records:
        return
        
    # 2. Extract image paths and labels
    train_paths, train_labels = [r['path'] for r in train_records], [r['label'] for r in train_records]
    val_paths, val_labels = [r['path'] for r in val_records], [r['label'] for r in val_records]
    
    # 3. Instantiate transformed datasets
    train_transform, val_transform = get_transforms(img_size=224)
    train_dataset = VisionInspectDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = VisionInspectDataset(val_paths, val_labels, transform=val_transform)
    
    # 4. Create DataLoader streams
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 5. Runtime environment initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training computational engine initialized on: {device}")
    
    model = DefectClassificationModel(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    os.makedirs("models", exist_ok=True)
    
    # 6. Core execution loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = (correct / total) * 100
        print(f"Epoch [{epoch+1}/{epochs}] ↳ Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        # Save checkpoints when accuracy beats previous peak
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/defect_model_best.pth")
            print("💾 Best performance weights saved successfully.")

if __name__ == "__main__":
    train_pipeline(epochs=3)