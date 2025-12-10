import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE=(128, 128)
BATCH_SIZE= 32
EPOCHS= 2 # Increased epochs
TRAIN_DIR="Training"
TEST_DIR="Testing"

"""Chooses best hardware to run cnn"""
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

"""Transform the images"""
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])
"""A full training dataset we will split into (train/val)"""
full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
class_names = full_train_dataset.classes
num_classes = len(class_names)
print("Class names:", class_names)
"""80%train and 20%testing """
val_size= int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
"""Test the Dataset"""
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#plot confusion matrix, with trues=validation labels, preds=predictions, title=plot title
def plot_cm(trues, preds, title, classes):
    print("Unique labels in y_true:", np.unique(trues))
    print("Unique predictions:", np.unique(preds))
    print("classes mapping:", classes)
    cm=100*confusion_matrix(trues, preds, normalize='true')
    plt.figure(figsize=(4,5))
    ax=sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    for text in ax.texts:
        text.set_text(text.get_text() + '%')
    plt.title(title, fontsize=16)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.tight_layout()
    os.makedirs("conf_mats", exist_ok=True)
    plt.savefig(f'conf_mats/{title.replace(" ", "_")}.png', dpi=300)

"""THE MODEL - IMPROVED"""
class BrainCNN(nn.Module):


    def __init__(self, num_classes: int):
        super(BrainCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 -> 64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16
            
            # Block 4 (New)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8
        )
        # After 4 pools: 128x128 -> 8x8; channels = 256
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
model = BrainCNN(num_classes=num_classes).to(device)
print(model)
MODEL_PATH = "../models/brain_cnn_4class.pth" # Moved to models folder
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Increased initial LR slightly
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

"""Training and Validation loop"""
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    preds_full=[]
    labels_full=[]
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            preds_full.append(preds.cpu())
            labels_full.append(labels.cpu())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    preds_np=torch.cat(preds_full).numpy()
    labels_np=torch.cat(labels_full).numpy()
    return epoch_loss, epoch_acc, preds_np, labels_np

best_val_acc = 0.0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, preds_np, labels_np     = evaluate(model, val_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
          f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
    plot_cm(labels_np, preds_np, f"Val Confusion Matrix Epoch {epoch+1}", class_names)
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved with Val Acc: {val_acc:.4f}")

print(f"Best Validation Accuracy: {best_val_acc:.4f}")

"""4 class test evaluation """
# Load best model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
all_outputs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        all_outputs.append(outputs.cpu())
        all_labels.append(labels)

all_outputs = torch.cat(all_outputs, dim=0)
all_labels = torch.cat(all_labels, dim=0)

probs = torch.softmax(all_outputs, dim=1).numpy()
y_pred = np.argmax(probs, axis=1)
y_true = all_labels.numpy()

print("\n4-class Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

plot_cm(y_true, y_pred, "Test Confusion Matrix", class_names)

"""Now map to Healthy vs Unhealthy Brain"""
idx_notumor = class_names.index("notumor")  #index of healthy class

y_true_binary = (y_true != idx_notumor).astype(int)  #0=healthy, 1=unhealthy
y_pred_binary = (y_pred != idx_notumor).astype(int)

binary_names = ["healthy", "unhealthy"]

print("\nBINARY Classification Report (healthy vs unhealthy):")
print(classification_report(y_true_binary, y_pred_binary, target_names=binary_names))
plot_cm(y_true_binary, y_pred_binary, "Binary Test Confusion Matrix", binary_names)

