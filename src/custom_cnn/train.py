import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.custom_cnn.model import ButterflyCNN
from src.evaluation.metrics import calculate_metrics, print_metrics
from src.evaluation.plots import plot_learning_curves, plot_confusion_matrix, compare_models_results

# config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_RUNS = 5
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
TEST_DIR = 'data/test'
RESULTS_DIR = 'results/custom_runs'
PLOTS_DIR = 'results/plots'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

def evaluate_on_test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds)

def train_and_evaluate():
    """
    Entraîne le CNN personnalisé 5 fois et calcule les métriques d'évaluation.
    """
    print(f"Utilisation de l'appareil : {device}")
    
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = train_dataset.classes
    num_classes = len(classes)
    
    all_accuracies = []
    
    for i in range(NUM_RUNS):
        print(f"\n{'='*20}")
        print(f"Run {i+1}/{NUM_RUNS}")
        print(f"{'='*20}")
        
        model = ButterflyCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        history = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{EPOCHS} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'best_model_run_{i+1}.pth'))
        
        history_path = os.path.join(RESULTS_DIR, f'history_run_{i+1}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
            
        model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f'best_model_run_{i+1}.pth')))
        y_true, y_pred = evaluate_on_test(model, test_loader, device)
        
        metrics = calculate_metrics(y_true, y_pred)
        all_accuracies.append(metrics['accuracy'])
        
        print(f"\nRésultats Run {i+1}:")
        print_metrics(metrics)
        
        plot_learning_curves(
            history, 
            save_path=os.path.join(PLOTS_DIR, f'learning_curves_custom_run_{i+1}.png'),
            title_suffix=f"(Custom CNN Run {i+1})"
        )
        
        if i == NUM_RUNS - 1:
            plot_confusion_matrix(
                y_true, y_pred, classes, 
                save_path=os.path.join(PLOTS_DIR, 'confusion_matrix_custom_final.png'),
                title=f"Matrice de Confusion Finale (Run {i+1})"
            )

    avg_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    print(f"\n{'#'*30}")
    print(f"RÉSULTATS FINAUX (CUSTOM CNN)")
    print(f"Accuracy moyenne sur {NUM_RUNS} runs: {avg_acc:.4f} (+/- {std_acc:.4f})")
    print(f"{'#'*30}")
    
    return avg_acc

if __name__ == "__main__":
    avg_acc = train_and_evaluate()
