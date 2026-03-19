# Data Module

## Commands

| Command | Description |
|---------|-------------|
| `python src/data/main.py` | Full data preparation pipeline: download dataset, analyze dataset, create stratified 70/15/15 split, apply augmentation |
| `python src/data/dataset.py` | Download and analyze dataset structure from CSV files and generate statistics |
| `python src/data/split_dataset.py` | Create train/validation/test splits with stratification |
| `python src/data/preprocessing.py` | Apply preprocessing and augmentation to training data |

## Classes

### DatasetAnalyzer
Analyzes dataset structure using CSV metadata files. Reads training labels from CSV and generates statistics.
- Methods: `analyze()`, `save_analysis()`

### DatasetSplitter
Creates stratified train/validation/test splits (70/15/15) from labeled training CSV.
- Methods: `split_stratified()`, `save_split_mapping()`

### ImagePreprocessor
Preprocesses images with normalization and augmentation (rotation, flip, zoom, brightness).
- Methods: `load_and_normalize()`, `apply_augmentations()`, `save_preprocessed_image()`

### PreprocessingPipeline
Complete preprocessing pipeline orchestrating image preprocessing with CSV labels.
- Methods: `preprocess_dataset()`, `load_preprocessed_dataset()`
