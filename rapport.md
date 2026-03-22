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

# Entraînement Transfer Learning

## Configuration expérimentale

Initialement, nous avons utilisé une architecture ResNet50 avec :
- 10 époques par configuration  
- 100% des données d’entraînement  

Cependant, ce choix entraînait un temps de calcul supérieur à 24 heures, ce qui était beaucoup trop long pour notre contexte.

Nous avons donc décidé d’adapter notre approche :
- Utilisation de ResNet18, un modèle plus léger  
- Réduction à 5 époques par configuration  
- Utilisation de 5000 images d’entraînement uniquement

Ces modifications ont permis de réduire considérablement le temps d’entraînement tout en conservant des performances satisfaisantes.

## Pipeline d'entraînement

Le pipeline d'entraînement a été implémenté avec PyTorch. À chaque epoch, une phase d'entraînement et une phase de validation sont exécutées. Les métriques suivantes sont enregistrées à chaque epoch : perte (loss) et précision (accuracy) pour les deux phases.

## Hyperparamètres testés

10 configurations ont été testées afin de comparer l'effet des hyperparamètres sur les performances :

| Run | Learning Rate | Optimizer | Batch Size |fine_tune  |
|-----|---------------|-----------|------------|-----------|
| 1   | 0.001         | Adam      | 32         | False     |
| 2   | 0.0001        | Adam      | 32         | False     |
| 3   | 0.001         | SGD       | 32         | False     |
| 4   | 0.001         | Adam      | 64         | False     |
| 5   | 0.01          | SGD       | 64         | False     |
| 6   | 0.001         | Adam      | 32         | True      |
| 7   | 0.0001        | Adam      | 32         | True      |
| 8   | 0.001         | SGD       | 32         | True      |
| 9   | 0.001         | Adam      | 64         | True      |
| 10  | 0.01          | SGD       | 64         | True      |

Les optimizers testés sont Adam et SGD (momentum=0.9). La fonction de perte utilisée est la CrossEntropyLoss, adaptée à la classification multi-classes (75 espèces).

## Impact du Fine-Tuning

Dans nos expérimentations, nous avons comparé deux approches de transfer learning :

- Freeze complet (sans fine-tuning) : seules les dernières couches (classification) sont entraînées  
- Fine-tuning partiel : certaines couches du modèle pré-entraîné sont dégelées et réentraînées  

### Observations

Nous avons constaté que :

- Le freeze complet permet un entraînement plus rapide et plus stable au début  
- Le fine-tuning permet généralement d’obtenir de meilleures performances, mais :
  - il est plus lent  
  - il peut parfois entraîner un surapprentissage (overfitting)  
  - il nécessite un bon réglage du learning rate  

Dans certains runs, le fine-tuning a permis d’augmenter légèrement l’accuracy de validation (jusqu’à ~0.80), tandis que dans d’autres cas, les performances restaient similaires voire légèrement inférieures.


### Explication

Cela s’explique par le fait que :

- Le modèle pré-entraîné (ResNet18) contient déjà des features générales efficaces  
- Si le dataset est relativement limité (5000 images), le fine-tuning peut :
  - soit améliorer l’adaptation  
  - soit dégrader les performances en surajustant  


Ainsi l’utilisation du fine-tuning représente un compromis :

- utile pour améliorer les performances  
- mais plus coûteux en temps et plus sensible aux hyperparamètres  

Dans notre cas, les deux approches donnent des résultats comparables, ce qui montre que le freeze complet est déjà une solution solide, tandis que le fine-tuning peut apporter un gain supplémentaire dans certaines configurations.

## Sauvegarde des résultats

Pour chaque run, le modèle entraîné est sauvegardé au format `.pth` et l'historique complet (losses et accuracies) est sauvegardé en `.json` afin de permettre la comparaison et la reproductibilité des résultats.

## Résultats des 10 runs

| Run | Learning Rate | Optimizer | Batch Size |fine_tune  | Val Accuracy |
|-----|---------------|-----------|------------|-----------|--------------|
| 1   | 0.001         | Adam      | 32         | False     | 0.765        |
| 2   | 0.0001        | Adam      | 32         | False     | 0.705        |
| 3   | 0.001         | SGD       | 32         | False     | 0.735        |
| 4   | 0.001         | Adam      | 64         | False     | 0.760        |
| 5   | 0.01          | SGD       | 64         | False     | 0.770        |
| 6   | 0.001         | Adam      | 32         | True      | 0.765        |
| 7   | 0.0001        | Adam      | 32         | True      | 0.805        |
| 8   | 0.001         | SGD       | 32         | True      | 0.795        |
| 9   | 0.001         | Adam      | 64         | True      | 0.745        |
| 10  | 0.01          | SGD       | 64         | True      | 0.805        |

Mean Validation Accuracy : 0.765

La configuration Run 7 (lr=0.0001, Adam, batch_size=32) et Run 10 (lr=0.01, SGD, batch_size=64) obtiennent les meilleures performances avec une accuracy de validation de 0.805.

## Discussion

Malgré la réduction de la taille du modèle et du nombre de données, les résultats restent corrects et relativement stables, avec une accuracy moyenne de 76.5%.

Le temps total d’entraînement pour ces expériences est d’environ 3h30, ce qui représente une amélioration très importante par rapport à la configuration initiale.

Ces résultats montrent que :
- Un modèle plus léger comme ResNet18 est suffisant pour ce problème  
- Une partie des données (5000 images) permet déjà d’obtenir des performances acceptables  
- L’utilisation de l’ensemble complet des données aurait probablement permis d’améliorer encore les résultats  

Cependant, les contraintes de temps de calcul rendaient cette option difficilement réalisable.

## Conclusion du Transfer learning

Cette approche représente un bon compromis entre :
- performances  
- temps d’entraînement  
- ressources disponibles  

Même avec un dataset réduit et moins d’époques, le modèle atteint des performances satisfaisantes, ce qui valide notre stratégie.

# Entraînement CNN personnalisé

## Configuration expérimentale

En plus du transfer learning, on a aussi entraîné un CNN fait par nous pour cette tâche. L’idée était surtout de :
- tester une architecture qu’on contrôle complètement
- voir plus clairement l’impact des choix de conception sur la classification des 75 espèces.

Le modèle est entraîné from scratch, donc sans poids pré-entraînés, ce qui nous permet d’analyser directement ce qu’il apprend sur notre dataset.

## Choix d’architecture

Le réseau suit une structure en 5 blocs:
- Bloc 1 : 3 -> 64 canaux
- Bloc 2 : 64 -> 128 canaux
- Bloc 3 : 128 -> 256 canaux
- Bloc 4 : 256 -> 512 canaux
- Bloc 5 : 512 -> 512 canaux

Les quatre premiers blocs contiennent un max pooling. Ça réduit progressivement la taille des images intermédiaires et augmente le champ réceptif. Pour les papillons, ce choix nous paraît pertinent car il permet de capturer à la fois :
- des détails locaux (textures, motifs d’ailes)
- des structures plus globales (forme générale, organisation des couleurs)

On utilise des convolutions 3x3 avec padding=1 dans tous les blocs, ce qui donne un bon compromis entre la performance et le temps de calcul.

## Stabilisation de l’apprentissage

Chaque bloc convolutionnel est suivi d’une Batch Normalization et d’une activation ReLU.

Ce choix nous a aidés à :
- stabiliser les distributions d’activations au cours de l’entraînement
- accélérer la convergence
- limiter les problèmes de gradients instables sur un réseau profond

L’utilisation systématique de ReLU garde aussi une architecture simple et plutot efficace en pratique.

## Tête de classification

Après les blocs convolutionnels, le modèle utilise :
- un Adaptive Average Pooling (sortie 1x1)
- une couche fully connected 512 -> 1024
- une couche de sortie 1024 -> 75 classes

L’AdaptiveAvgPool2d permet de résumer l’information spatiale avant la classification, avec une tête de réseau plus compacte qu’un flatten direct.

La couche intermédiaire à 1024 neurones donne assez de capacité pour séparer des espèces qui se ressemblent visuellement.

## Régularisation et généralisation

Deux couches de Dropout (p=0.5) sont appliquées dans la tête de classification.

Ce choix est important dans notre cas car :
- le nombre de classes est élevé
- certaines espèces présentent des ressemblances importantes
- notre modèle from scratch est plus exposé au surapprentissage

## Initialisation des poids

On a utilisé une initialisation classique adaptée au type de couches pour éviter un démarrage instable de l’entraînement.
En pratique, ça aide le modèle à converger plus proprement dès les premières époques et à obtenir des résultats plus réguliers.

## Justification globale

Au final, ce CNN personnalisé représente un bon compromis entre :
- capacité de représentation (profondeur et nombre de canaux)
- robustesse de l’entraînement (BatchNorm, ReLU, initialisation adaptée)
- généralisation (Dropout)
- coût de calcul raisonnable (progression structurée des blocs + pooling)

En résumé, l’architecture est assez expressive pour faire de la classification fine de papillons, tout en restant raisonnable à entraîner dans le cadre du TP.


