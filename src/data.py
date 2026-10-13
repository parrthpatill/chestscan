"""ChestScan image loading, augmentation, and tf.data pipelines."""

import csv
from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def load_manifest(manifest_path, dataset_root):
    """Load CSV rows and resolve image paths against the dataset root."""
    dataset_root = Path(dataset_root)
    required = {
        "path", "label", "patient_id",
        "split_original", "split_new", "test_cohort",
    }

    with open(manifest_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest is missing required columns.")

        rows = list(reader)

    for row in rows:
        parts = Path(row["path"]).parts

        # Recover the path relative to the actual dataset root.
        split_index = next(
            (i for i, part in enumerate(parts)
             if part in {"train", "val", "test"}),
            None,
        )
        if split_index is None:
            raise ValueError(f"Cannot resolve image path: {row['path']}")

        resolved = dataset_root.joinpath(*parts[split_index:])

        if not resolved.is_file():
            raise FileNotFoundError(f"Image not found: {resolved}")

        row["path"] = str(resolved)

    return rows


def decode_image(path, label):
    """Decode JPEG, convert to RGB, resize, and scale to [0, 255]."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.float32)


def build_augmentation():
    """Create augmentation layers for training images only."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.0278),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="training_augmentation",
    )


def make_dataset(rows, training=False, batch_size=BATCH_SIZE):
    """Create a batched, prefetched dataset from manifest rows."""
    if not rows:
        raise ValueError("Cannot create a dataset from zero rows.")

    paths = [row["path"] for row in rows]
    labels = [
        1 if row["label"] == "PNEUMONIA" else 0
        for row in rows
    ]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(
        decode_image,
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(rows),
            seed=42,
            reshuffle_each_iteration=True,
        )

        augmentation = build_augmentation()

        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def build_datasets(manifest_path, dataset_root):
    """Build train, validation, full-test, and clean-test datasets."""
    rows = load_manifest(manifest_path, dataset_root)

    train_rows = [r for r in rows if r["split_new"] == "train"]
    val_rows = [r for r in rows if r["split_new"] == "val"]
    test_rows = [r for r in rows if r["split_new"] == "test"]

    clean_test_rows = [
        r for r in test_rows
        if "test_clean" in r["test_cohort"].split(";")
    ]

    return {
        "train": make_dataset(train_rows, training=True),
        "val": make_dataset(val_rows),
        "test_full": make_dataset(test_rows),
        "test_clean": make_dataset(clean_test_rows),
    }
