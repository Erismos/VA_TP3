"""Dataset loader and analyzer for CSV-labeled train/test image datasets."""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import kagglehub


class DatasetAnalyzer:
    """Analyze dataset structure using CSV metadata files.

    Train labels are read from Training_set.csv (filename,label).
    Test images are read from Testing_set.csv (filename).
    """

    def __init__(
        self,
        train_dir="data/raw/butterfly/train",
        test_dir="data/raw/butterfly/test",
        train_labels_csv="data/raw/butterfly/Training_set.csv",
        test_csv="data/raw/butterfly/Testing_set.csv",
        expected_classes=75,
    ):
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.train_labels_csv = Path(train_labels_csv)
        self.test_csv = Path(test_csv)
        self.expected_classes = expected_classes

        self.images_by_class = defaultdict(list)
        self.test_images = []
        self.missing_train_files = []
        self.missing_test_files = []

    def analyze(self):
        """Analyze dataset from CSV files and print key statistics."""
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
        if not self.train_labels_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {self.train_labels_csv}")
        if not self.test_csv.exists():
            raise FileNotFoundError(f"Testing CSV not found: {self.test_csv}")

        print("\nAnalyzing dataset from CSV files...")

        with open(self.train_labels_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"].strip()
                label = row["label"].strip()
                img_path = self.train_dir / filename

                if img_path.exists():
                    self.images_by_class[label].append(str(img_path))
                else:
                    self.missing_train_files.append(filename)

        with open(self.test_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"].strip()
                img_path = self.test_dir / filename

                if img_path.exists():
                    self.test_images.append(str(img_path))
                else:
                    self.missing_test_files.append(filename)

        return self._print_statistics()

    def _print_statistics(self):
        total_train_images = sum(len(imgs) for imgs in self.images_by_class.values())
        total_test_images = len(self.test_images)
        total_images = total_train_images + total_test_images

        num_classes = len(self.images_by_class)
        class_counts = {cls: len(imgs) for cls, imgs in self.images_by_class.items()}

        min_imgs = min(class_counts.values()) if class_counts else 0
        max_imgs = max(class_counts.values()) if class_counts else 0
        avg_imgs = (total_train_images / num_classes) if num_classes else 0.0
        imbalance_ratio = (max_imgs / min_imgs) if min_imgs else 0.0

        print(f"Total images: {total_images}")
        print(f"Train images (labeled): {total_train_images}")
        print(f"Test images (unlabeled): {total_test_images}")
        print(f"Number of classes: {num_classes}")
        print(f"Expected classes: {self.expected_classes}")
        print(f"Class count check: {'OK' if num_classes == self.expected_classes else 'MISMATCH'}")
        print(f"Min images/class: {min_imgs}")
        print(f"Max images/class: {max_imgs}")
        print(f"Avg images/class: {avg_imgs:.1f}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}x")

        if self.missing_train_files:
            print(f"Missing train images referenced by CSV: {len(self.missing_train_files)}")
        if self.missing_test_files:
            print(f"Missing test images referenced by CSV: {len(self.missing_test_files)}")

        print("\nDataset status: OK")

        return {
            "total_images": total_images,
            "train_images": total_train_images,
            "test_images": total_test_images,
            "num_classes": num_classes,
            "expected_classes": self.expected_classes,
            "images_per_class": class_counts,
            "min_images_per_class": min_imgs,
            "max_images_per_class": max_imgs,
            "avg_images_per_class": avg_imgs,
            "imbalance_ratio": imbalance_ratio,
            "missing_train_files": len(self.missing_train_files),
            "missing_test_files": len(self.missing_test_files),
        }

    def save_analysis(self, output_file="dataset_analysis.json"):
        """Save dataset analysis as JSON."""
        data = {
            "train_dir": str(self.train_dir),
            "test_dir": str(self.test_dir),
            "train_labels_csv": str(self.train_labels_csv),
            "test_csv": str(self.test_csv),
            "num_classes": len(self.images_by_class),
            "classes": {
                cls: {
                    "count": len(imgs),
                    "images": imgs,
                }
                for cls, imgs in self.images_by_class.items()
            },
            "test_set_images": self.test_images,
            "missing_train_files": self.missing_train_files,
            "missing_test_files": self.missing_test_files,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Analysis saved: {output_file}")
        return output_file


def load_image_paths(
    train_dir="data/raw/butterfly/train",
    train_labels_csv="data/raw/butterfly/Training_set.csv",
):
    """Load labeled train image paths from Training_set.csv."""
    images_by_class = defaultdict(list)
    train_dir = Path(train_dir)

    with open(train_labels_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            label = row["label"].strip()
            img_path = train_dir / filename
            if img_path.exists():
                images_by_class[label].append(str(img_path))

    return dict(images_by_class)


def download_dataset(dataset_name="phucthaiv02/butterfly-image-classification"):
    """Download dataset from Kaggle and move it under data/raw/butterfly."""
    print(f"Downloading dataset: {dataset_name}")
    dataset_path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset path: {dataset_path}")

    target_path = Path("data/raw/butterfly")
    if target_path.exists():
        print(f"Target path already exists: {target_path}")
        return str(target_path)

    os.makedirs(target_path.parent, exist_ok=True)
    os.rename(dataset_path, target_path)
    print(f"Dataset moved to: {target_path}")
    return str(target_path)


if __name__ == "__main__":
    if not Path("data/raw/butterfly").exists():
        download_dataset()
    analyzer = DatasetAnalyzer()
    analyzer.analyze()
    analyzer.save_analysis()
