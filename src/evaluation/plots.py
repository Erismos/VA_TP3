import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_learning_curves(history, save_path=None, title_suffix=""):
    """
    Trace les courbes d'apprentissage (accuracy et loss).
    
    Args:
        history: Dictionnaire contenant 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
        save_path: Chemin pour sauvegarder le plot.
        title_suffix: Suffixe à ajouter au titre.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Training and Validation Accuracy {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Training and Validation Loss {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None, title="Matrice de Confusion"):
    """
    Trace la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def compare_models_results(results_dict, save_path=None):
    """
    Compare les résultats de plusieurs modèles.
    results_dict: { 'model_name': accuracy_score, ... }
    """
    plt.figure(figsize=(10, 6))
    names = list(results_dict.keys())
    values = list(results_dict.values())
    
    sns.barplot(x=names, y=values)
    plt.title("Comparaison des Accuracy des Modèles")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
