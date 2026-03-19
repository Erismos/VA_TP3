"""Create and load a stratified 70/15/15 split from Training_set.csv labels."""

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


class DatasetSplitter:
    """Create train/validation/test splits from labeled training CSV only."""

    def __init__(
        self,
        train_dir="data/raw/butterfly/train",
        train_labels_csv="data/raw/butterfly/Training_set.csv",
        output_dir="data",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        ignore_unlabeled_test=True,
    ):
        self.train_dir = Path(train_dir)
        self.train_labels_csv = Path(train_labels_csv)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.ignore_unlabeled_test = ignore_unlabeled_test

        self.split_mapping = {"train": [], "validation": [], "test": []}
        self.missing_train_files = []
        random.seed(random_seed)

    def _load_labeled_images(self):
        """Read Training_set.csv and group images by class."""
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")
        if not self.train_labels_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {self.train_labels_csv}")

        images_by_class = defaultdict(list)

        with open(self.train_labels_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"].strip()
                label = row["label"].strip()
                img_path = self.train_dir / filename

                if img_path.exists():
                    images_by_class[label].append(
                        {
                            "class": label,
                            "path": str(img_path),
                            "filename": filename,
                        }
                    )
                else:
                    self.missing_train_files.append(filename)

        return images_by_class

    def split_stratified(self):
        """Perform stratified 70/15/15 split using only labeled images."""
        print("\nCreating stratified split from Training_set.csv")
        print(
            f"Ratios -> train: {self.train_ratio:.2f}, "
            f"validation: {self.val_ratio:.2f}, test: {self.test_ratio:.2f}"
        )
        print(f"Seed: {self.random_seed}")

        if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) > 1e-9:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        if self.ignore_unlabeled_test:
            print("Unlabeled raw/test images are ignored for split creation.")

        images_by_class = self._load_labeled_images()

        for class_name, images in sorted(images_by_class.items()):
            random.shuffle(images)
            total = len(images)

            train_count = int(total * self.train_ratio)
            val_count = int(total * self.val_ratio)
            test_count = total - train_count - val_count

            train_items = images[:train_count]
            val_items = images[train_count : train_count + val_count]
            test_items = images[train_count + val_count :]

            self.split_mapping["train"].extend(train_items)
            self.split_mapping["validation"].extend(val_items)
            self.split_mapping["test"].extend(test_items)

        total_train = len(self.split_mapping["train"])
        total_val = len(self.split_mapping["validation"])
        total_test = len(self.split_mapping["test"])
        total = total_train + total_val + total_test

        print("\nSplit summary")
        print(f"Train:      {total_train:5d} ({(total_train/total)*100:5.1f}%)")
        print(f"Validation: {total_val:5d} ({(total_val/total)*100:5.1f}%)")
        print(f"Test:       {total_test:5d} ({(total_test/total)*100:5.1f}%)")
        print(f"Total:      {total:5d}")

        if self.missing_train_files:
            print(f"Missing train images referenced by CSV: {len(self.missing_train_files)}")

    def copy_to_directories(self):
        """Copy split images to data/train, data/validation, data/test folders."""
        print("\nCopying split files to output directories")

        split_to_folder = {
            "train": "train",
            "validation": "validation",
            "test": "test",
        }

        for split_name, folder_name in split_to_folder.items():
            split_root = self.output_dir / folder_name
            split_root.mkdir(parents=True, exist_ok=True)

            for item in self.split_mapping[split_name]:
                class_name = item["class"]
                src = Path(item["path"])
                dst_dir = split_root / class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / item["filename"]
                shutil.copy2(src, dst)

            print(f"  Copied {len(self.split_mapping[split_name])} images -> {split_root}")

    def save_split_mapping(self, output_file="split_mapping.json"):
        """Save split mapping to JSON for reproducibility."""
        mapping_data = {
            "random_seed": self.random_seed,
            "train_ratio": self.train_ratio,
            "validation_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "ignored_unlabeled_raw_test": self.ignore_unlabeled_test,
            "missing_train_files": self.missing_train_files,
            "splits": self.split_mapping,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2)

        print(f"\nSplit mapping saved: {output_file}")
        return output_file

    def execute(self, copy_files=True):
        """Run split workflow and optionally materialize folders."""
        self.split_stratified()
        if copy_files:
            self.copy_to_directories()
        self.save_split_mapping()
        print("\nSplit creation completed")


class SplitLoader:
    """Load saved train/validation/test mapping."""

    def __init__(self, split_file="split_mapping.json"):
        self.split_file = split_file
        self.mapping = None
        if Path(split_file).exists():
            self.load()

    def load(self):
        with open(self.split_file, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

    def get_split(self, split_type="train"):
        if self.mapping is None:
            raise ValueError("Split mapping not loaded")
        return self.mapping["splits"][split_type]

    def get_all_splits(self):
        if self.mapping is None:
            raise ValueError("Split mapping not loaded")
        return self.mapping["splits"]


if __name__ == "__main__":
    import sys

    splitter = DatasetSplitter(random_seed=42)
    try:
        splitter.execute(copy_files=True)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\nDataset split creation completed")
    print("Mapping saved: split_mapping.json")
