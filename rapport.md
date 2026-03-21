# Vision Artificielle - TP3
> Clément Auvray & Anna-Eve Mercier & Flavien Baron & Ewan Schwaller & Laure Warlop

## Introduction

La classification d'images repose sur deux approches principales : le transfer learning, qui adapte un modèle pré-entraîné au problème spécifique, et le CNN personnalisé, entraîné de zéro. Ce TP compare ces deux stratégies sur un problème de classification de papillons.

## Choix du Dataset et Preprocessing

### Sélection du Dataset

Le dataset Butterfly Image Classification (Kaggle) est composé de 6499 images étiquetées répartis en 75 espèces de papillons. La distribution est équilibrée : 71-131 images par classe (moyenne 86.7), avec un ratio de déséquilibre de 1.85x.

### Stratification et Split des Données

Un split stratifié 70/15/15 a été appliqué avec seed=42 pour assurer la reproducibilité et la comparabilité entre les modèles. Le split respecte la distribution des classes : 4509 images d'entraînement, 940 de validation, 1050 de test. Les images sont organisées par classe (dossiers) dans chaque sous-ensemble.

### Data Augmentation

L'augmentation a été appliquée exclusivement au set d'entraînement pour maintenir l'intégrité de la validation/test. Les transformations incluent : rotations ±15°, flips horizontaux, zoom 80%, brightness/contrast 1.1x. Ces augmentations ont produit ~27054 images (6x multiplicateur). Validation et test restent non-augmentés pour une évaluation honnête.

## Entraînement Transfer Learning

### Pipeline d'entraînement

Le pipeline d'entraînement a été implémenté avec PyTorch. À chaque epoch, une phase d'entraînement et une phase de validation sont exécutées. Les métriques suivantes sont enregistrées à chaque epoch : perte (loss) et précision (accuracy) pour les deux phases.

### Hyperparamètres testés

5 configurations ont été testées afin de comparer l'effet des hyperparamètres sur les performances :

| Run | Learning Rate | Optimizer | Batch Size |
|-----|--------------|-----------|------------|
| 1   | 0.001        | Adam      | 32         |
| 2   | 0.0001       | Adam      | 32         |
| 3   | 0.001        | SGD       | 32         |
| 4   | 0.001        | Adam      | 64         |
| 5   | 0.01         | SGD       | 64         |

Les optimizers testés sont Adam et SGD (momentum=0.9). Le nombre d'epochs est fixé à 10 pour tous les runs. La fonction de perte utilisée est la CrossEntropyLoss, adaptée à la classification multi-classes (75 espèces).

### Sauvegarde des résultats

Pour chaque run, le modèle entraîné est sauvegardé au format `.pth` et l'historique complet (losses et accuracies) est sauvegardé en `.json` pour permettre la comparaison et la reproductibilité des résultats.

### Résultats des 5 runs

| Run | LR     | Optimizer | Batch Size | Val Accuracy |
|-----|--------|-----------|------------|--------------|
| 1   | 0.001  | Adam      | 32         |         |
| 2   | 0.0001 | Adam      | 32         |        |
| 3   | 0.001  | SGD       | 32         |        |
| 4   | 0.001  | Adam      | 64         |        |
| 5   | 0.01   | SGD       | 64         |        |

**Mean Validation Accuracy : **

La configuration Run X (,,) obtient les meilleures performances. 