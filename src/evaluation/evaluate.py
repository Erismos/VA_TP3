import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.custom_cnn.model import ButterflyCNN
from src.transfer_learning.model import TransferModel
from src.evaluation.metrics import calculate_metrics, print_metrics
from src.evaluation.plots import plot_confusion_matrix, compare_models_results

# config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DIR = 'data/test'
TRAIN_DIR = 'data/train'
CUSTOM_MODEL_PATH = 'results/custom_runs/best_model_run_1.pth'
TRANSFER_MODEL_PATH = 'results/transfer_model_run_1.pth'
PLOTS_DIR = 'results/plots'

os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_on_test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds)

def main():
    print(f"Utilisation de l'appareil : {device}")
    
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(TEST_DIR) or not os.path.exists(TRAIN_DIR):
        print("Erreur: Dossiers data/test ou data/train manquants.")
        return

    full_test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    full_train_dataset = datasets.ImageFolder(TRAIN_DIR)
    
    classes = full_test_dataset.classes
    num_classes = len(classes)
    
    train_subset = torch.utils.data.Subset(full_train_dataset, range(5000))
    trained_classes_indices = sorted(list(set([full_train_dataset.samples[i][1] for i in train_subset.indices])))
    trained_classes_names = [classes[i] for i in trained_classes_indices]
    
    print(f"Nombre de classes identifiées dans le subset (5000 images): {len(trained_classes_indices)}")

    indices_test = [i for i, (_, label) in enumerate(full_test_dataset.samples) if label in trained_classes_indices]
    test_subset = torch.utils.data.Subset(full_test_dataset, indices_test)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    results = {}

    # custom CNN
    if os.path.exists(CUSTOM_MODEL_PATH):
        print(f"\nÉvaluation de Custom CNN...")
        model = ButterflyCNN(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location=device))
        
        y_true, y_pred = evaluate_on_test(model, test_loader, device)
        metrics = calculate_metrics(y_true, y_pred)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        results['Custom CNN'] = metrics['accuracy']
        
        plot_confusion_matrix(
            y_true, y_pred, trained_classes_names, 
            save_path=os.path.join(PLOTS_DIR, 'cm_custom_subset.png'),
            title="Matrice de Confusion - Custom CNN (Subset 5000)"
        )

    # transfer learning
    if os.path.exists(TRANSFER_MODEL_PATH):
        print(f"\nÉvaluation de Transfer Learning...")
        model = TransferModel(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(TRANSFER_MODEL_PATH, map_location=device))
        
        y_true, y_pred = evaluate_on_test(model, test_loader, device)
        metrics = calculate_metrics(y_true, y_pred)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        results['Transfer Learning'] = metrics['accuracy']
        
        plot_confusion_matrix(
            y_true, y_pred, trained_classes_names, 
            save_path=os.path.join(PLOTS_DIR, 'cm_transfer_subset.png'),
            title="Matrice de Confusion - Transfer Learning (Subset 5000)"
        )

    # comparaison finale
    if len(results) > 1:
        print("\n" + "="*30)
        print("COMPARAISON FINALE (SUBSET 5000)")
        print("="*30)
        for name, acc in results.items():
            print(f"{name:18}: {acc:.4f}")
        
        compare_models_results(results, save_path=os.path.join(PLOTS_DIR, 'comparison_subset_5000.png'))

if __name__ == "__main__":
    main()
