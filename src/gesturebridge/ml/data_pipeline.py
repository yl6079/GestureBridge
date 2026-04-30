from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def load_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"path", "label", "class_name"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Manifest {csv_path} is missing required columns: {expected}")
    return df


def load_class_names(labels_path: Path) -> list[str]:
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _decode_and_resize(image_path: tf.Tensor, image_size: int) -> tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    return image


def _augment(image: tf.Tensor) -> tf.Tensor:
    # ASL signs are chirality-sensitive (J vs L, the trajectory of J/Z) so
    # horizontal flip is OFF — that was a real bug in the previous pipeline.
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.image.random_hue(image, max_delta=0.05)
    # Small random crop+pad simulates the "hand isn't perfectly centered"
    # case we hit on the C270 webcam, without needing MediaPipe at train time.
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    pad = tf.cast(tf.cast(h, tf.float32) * 0.08, tf.int32)
    image = tf.image.resize_with_crop_or_pad(image, h + pad, w + pad)
    image = tf.image.random_crop(image, size=[h, w, 3])
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image


def preprocess_for_mobilenet(image: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.mobilenet_v3.preprocess_input(image)


def build_dataset(
    csv_path: Path,
    image_size: int,
    batch_size: int,
    training: bool,
    shuffle_seed: int,
) -> tf.data.Dataset:
    manifest = load_manifest(csv_path)
    paths = manifest["path"].to_numpy(dtype=str)
    labels = manifest["label"].to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=shuffle_seed,
            reshuffle_each_iteration=True,
        )

    def _parse(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = _decode_and_resize(path, image_size=image_size)
        if training:
            image = _augment(image)
        image = preprocess_for_mobilenet(image)
        return image, label

    dataset = dataset.map(_parse, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def representative_dataset_from_csv(
    csv_path: Path,
    image_size: int,
    sample_count: int,
) -> tf.lite.RepresentativeDataset:
    manifest = load_manifest(csv_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {csv_path}")
    sample_manifest = manifest.sample(n=min(sample_count, len(manifest)), random_state=42)
    sample_paths = sample_manifest["path"].tolist()

    def _generator():
        for path in sample_paths:
            image = _decode_and_resize(tf.constant(path), image_size=image_size)
            image = preprocess_for_mobilenet(image)
            image = tf.expand_dims(image, axis=0)
            yield [tf.cast(image, tf.float32)]

    return _generator

