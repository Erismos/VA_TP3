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