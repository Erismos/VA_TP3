"""Main pipeline for CSV-labeled dataset: analyze, split, then augment train.

This script orchestrates the complete data preparation workflow:
1. Analyze raw train/test structure from CSV files
2. Create train/validation/test split (70/15/15) from labeled training CSV only
3. Apply augmentation only on train split
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import DatasetAnalyzer, download_dataset
from data.preprocessing import PreprocessingPipeline
from data.split_dataset import DatasetSplitter


def main():
    """Execute complete data preparation pipeline."""
    
    print("VA_TP3 - Data Preparation Pipeline")
    
    # Configuration
    train_path = "data/raw/butterfly/train"
    test_path = "data/raw/butterfly/test"
    train_labels_csv = "data/raw/butterfly/Training_set.csv"
    test_csv = "data/raw/butterfly/Testing_set.csv"
    train_split_dir = "data/train"
    
    if not Path(train_path).exists():
        download_dataset()
    
    # =========================================================================
    # Phase 1: Analyze dataset
    # =========================================================================
    print("\nDataset Analysis")
    
    try:
        analyzer = DatasetAnalyzer(
            train_dir=train_path,
            test_dir=test_path,
            train_labels_csv=train_labels_csv,
            test_csv=test_csv,
            expected_classes=75,
        )
        stats = analyzer.analyze()
        analyzer.save_analysis("dataset_analysis.json")
        print("Dataset analysis completed")
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return False
    
    # =========================================================================
    # Phase 2: Create train/validation/test split (70/15/15)
    # =========================================================================
    print("\nCreate train/validation/test split")
    
    try:
        splitter = DatasetSplitter(
            train_dir=train_path,
            train_labels_csv=train_labels_csv,
            output_dir="data",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            ignore_unlabeled_test=True,
        )

        splitter.execute(copy_files=True)
        print("Dataset split completed")
        print("Mapping saved: split_mapping.json")
        
    except Exception as e:
        print(f"Error during split: {e}")
        return False
    
    # =========================================================================
    # Phase 3: Apply augmentation only on train split
    # =========================================================================
    print("\nAugmentation on train split")
    
    try:
        pipeline = PreprocessingPipeline(
            train_dir=train_path,
            train_labels_csv=train_labels_csv,
            target_size=(224, 224),
        )

        stats = pipeline.augment_split_train(split_train_dir=train_split_dir)
        print("Train augmentation completed")
        print(f"Original train images: {stats['original_train_images']}")
        print(f"Augmented images created: {stats['augmented_created']}")

        
    except Exception as e:
        print(f"Error during augmentation: {e}")
        return False
    
    print("Data preparation pipeline completed successfully")
    
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
