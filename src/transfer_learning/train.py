import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from pathlib import Path

import torchvision.transforms as transforms

from model import get_model
# def get_model(num_classes):
#     return nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(224 * 224 * 3, 128),
#         nn.ReLU(),
#         nn.Linear(128, num_classes)
#     )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 10

def get_dataloaders(batch_size=32):

    BASE_DIR = Path(__file__).resolve().parents[2]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
    BASE_DIR / "data" / "train",
    transform=transform
)

    val_dataset = datasets.ImageFolder(
        BASE_DIR / "data" / "validation",
        transform=transform
    )

    # pour tester sans le vrai model donc avec peu d'image pour voir le fonctionnement du pipeline -> rajouter .dataset dans le return avant .classes
    # train_dataset = Subset(train_dataset, range(200))
    # val_dataset = Subset(val_dataset, range(50))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, len(train_dataset.classes) 


def train_model(lr=0.001, optimizer_name="adam", batch_size=32):
    train_loader, val_loader, num_classes = get_dataloaders(batch_size=batch_size)
    model = get_model(num_classes=num_classes).to(device)
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):

        # TRAIN
        model.train()
        correct = 0
        total = 0
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # VALIDATION
        model.eval()
        correct = 0
        total = 0
        running_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_loss = running_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    return model, train_losses, val_losses, train_accuracies, val_accuracies