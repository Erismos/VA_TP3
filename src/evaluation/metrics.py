from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import numpy as np
import json

def calculate_metrics(y_true, y_pred):
    """
    Calcule les métriques d'évaluation principales.
    
    Args:
        y_true: Les étiquettes réelles.
        y_pred: Les étiquettes prédites.
        
    Returns:
        dict: Un dictionnaire contenant les différentes métriques.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics

def print_metrics(metrics):
    """
    Affiche les métriques de manière lisible.
    """
    print(f"Accuracy Globale: {metrics['accuracy']:.4f}")
    print(f"Coefficient Kappa: {metrics['kappa']:.4f}")
    print("\nClassification Report:")
    print(json.dumps(metrics['classification_report'], indent=4))
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
