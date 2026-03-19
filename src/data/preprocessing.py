"""Image preprocessing for CSV-labeled training images."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageEnhance


class ImagePreprocessor:
    """Preprocesses images: normalization and optional augmentation."""

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.processed_count = 0

    def load_and_normalize(self, image_path):
        """Load an image and normalize pixels to [0, 1]."""
        img = Image.open(image_path)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img_array = np.array(img, dtype=np.float32) / 255.0
        self.processed_count += 1
        return img_array

    def save_preprocessed_image(self, image_array, output_path):
        """Save preprocessed image array to disk."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        img_uint8 = (image_array * 255).astype(np.uint8)
        img = Image.fromarray(img_uint8)
        img.save(output_path)

    def apply_augmentations(self, image_array, augmentation_list=None):
        """Apply optional augmentations."""
        if augmentation_list is None:
            augmentation_list = ["rotation", "flip_h", "zoom", "brightness_contrast"]

        augmented_images = [image_array]

        if "rotation" in augmentation_list:
            img_pil = Image.fromarray((image_array * 255).astype(np.uint8))

            rotated_pos = img_pil.rotate(15, resample=Image.Resampling.BILINEAR)
            augmented_images.append(np.array(rotated_pos) / 255.0)

            rotated_neg = img_pil.rotate(-15, resample=Image.Resampling.BILINEAR)
            augmented_images.append(np.array(rotated_neg) / 255.0)

        if "flip_h" in augmentation_list:
            augmented_images.append(np.fliplr(image_array))

        if "flip_v" in augmentation_list:
            augmented_images.append(np.flipud(image_array))

        if "zoom" in augmentation_list:
            h, w = image_array.shape[:2]
            crop_size = int(h * 0.8)
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2

            cropped = image_array[start_h : start_h + crop_size, start_w : start_w + crop_size]
            img_pil = Image.fromarray((cropped * 255).astype(np.uint8))
            img_pil = img_pil.resize((w, h), Image.Resampling.LANCZOS)
            augmented_images.append(np.array(img_pil) / 255.0)

        if "brightness_contrast" in augmentation_list:
            img_pil = Image.fromarray((image_array * 255).astype(np.uint8))

            bright = ImageEnhance.Brightness(img_pil).enhance(1.1)
            augmented_images.append(np.array(bright) / 255.0)

            contrast = ImageEnhance.Contrast(img_pil).enhance(1.1)
            augmented_images.append(np.array(contrast) / 255.0)

        return augmented_images


class PreprocessingPipeline:
    """Preprocessing pipeline using Training_set.csv labels."""

    def __init__(
        self,
        train_dir="data/raw/butterfly/train",
        train_labels_csv="data/raw/butterfly/Training_set.csv",
        output_dir=None,
        target_size=(224, 224),
    ):
        self.train_dir = Path(train_dir)
        self.train_labels_csv = Path(train_labels_csv) if train_labels_csv else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.target_size = target_size
        self.preprocessor = ImagePreprocessor(target_size)

    def preprocess_dataset(self, create_augmentations=False):
        """Preprocess labeled training set from CSV rows."""
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")
        if self.train_labels_csv is None or not self.train_labels_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {self.train_labels_csv}")
        if self.output_dir is None:
            raise ValueError("output_dir must be provided for preprocess_dataset")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\nProcessing labeled training dataset...")

        stats = {
            "total_processed": 0,
            "classes_processed": defaultdict(int),
            "missing_files": 0,
        }

        with open(self.train_labels_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"].strip()
                class_name = row["label"].strip()
                img_path = self.train_dir / filename

                if not img_path.exists():
                    stats["missing_files"] += 1
                    continue

                img_array = self.preprocessor.load_and_normalize(str(img_path))

                output_class_dir = self.output_dir / class_name
                output_class_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_class_dir / filename
                self.preprocessor.save_preprocessed_image(img_array, str(output_path))

                stats["classes_processed"][class_name] += 1

                if create_augmentations:
                    augmented = self.preprocessor.apply_augmentations(img_array)
                    stem = Path(filename).stem
                    suffix = Path(filename).suffix

                    for i, aug_img in enumerate(augmented[1:], 1):
                        aug_path = output_class_dir / f"{stem}_aug{i}{suffix}"
                        self.preprocessor.save_preprocessed_image(aug_img, str(aug_path))

        stats["total_processed"] = self.preprocessor.processed_count

        print("\nPreprocessing completed")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Missing files: {stats['missing_files']}")
        print(f"Output: {self.output_dir}")

        return stats

    def augment_split_train(self, split_train_dir="data/train", augmentation_list=None):
        """Apply augmentation only on the train split directory.

        Validation and test splits are not touched by this method.
        """
        split_train_dir = Path(split_train_dir)
        if not split_train_dir.exists():
            raise FileNotFoundError(f"Train split directory not found: {split_train_dir}")

        print("\nApplying augmentation on train split only...")

        stats = {
            "original_train_images": 0,
            "augmented_created": 0,
            "classes_processed": defaultdict(int),
        }

        for class_dir in sorted(split_train_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            image_files = [
                p
                for p in sorted(class_dir.glob("*"))
                if p.is_file()
                and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                and "_aug" not in p.stem
            ]

            for img_file in image_files:
                img_array = self.preprocessor.load_and_normalize(str(img_file))
                augmented = self.preprocessor.apply_augmentations(
                    img_array, augmentation_list=augmentation_list
                )

                for i, aug_img in enumerate(augmented[1:], 1):
                    aug_path = class_dir / f"{img_file.stem}_aug{i}{img_file.suffix}"
                    self.preprocessor.save_preprocessed_image(aug_img, str(aug_path))
                    stats["augmented_created"] += 1

                stats["original_train_images"] += 1
                stats["classes_processed"][class_name] += 1


        print("\nTrain augmentation completed")
        print(f"Original train images processed: {stats['original_train_images']}")
        print(f"Augmented images created: {stats['augmented_created']}")

        return stats


if __name__ == "__main__":
    import sys

    pipeline = PreprocessingPipeline(output_dir="data/train_preprocessed")
    try:
        pipeline.preprocess_dataset(create_augmentations=False)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\nPreprocessing complete!")
