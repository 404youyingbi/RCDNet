import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import tensorflow as tf


def list_image_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images matched pattern: {pattern}")
    return files


def validate_paired_files(raw_files: Sequence[str], corrected_files: Sequence[str]) -> None:
    if len(raw_files) != len(corrected_files):
        raise ValueError(
            "Input/target file counts differ: "
            f"{len(raw_files)} input files vs {len(corrected_files)} target files."
        )


def _read_png(path: tf.Tensor) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    return tf.cast(image, tf.float32)


def _normalize_to_minus_one_one(image: tf.Tensor) -> tf.Tensor:
    return (image / 127.5) - 1.0


def _stateless_random_crop_pair(
    raw_img: tf.Tensor,
    corrected_img: tf.Tensor,
    crop_size: int,
    seed: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    stacked_images = tf.stack([raw_img, corrected_img], axis=0)
    cropped_images = tf.image.stateless_random_crop(
        stacked_images,
        size=[2, crop_size, crop_size, 3],
        seed=seed,
    )
    return cropped_images[0], cropped_images[1]


def _stateless_augment_pair(
    raw_img: tf.Tensor,
    corrected_img: tf.Tensor,
    seed: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    flip_lr = tf.random.stateless_uniform(shape=[], seed=seed + tf.constant([1, 0], tf.int32)) > 0.5
    flip_ud = tf.random.stateless_uniform(shape=[], seed=seed + tf.constant([2, 0], tf.int32)) > 0.5
    rot_k = tf.random.stateless_uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32,
        seed=seed + tf.constant([3, 0], tf.int32),
    )

    def flip_left_right_pair() -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.image.flip_left_right(raw_img), tf.image.flip_left_right(corrected_img)

    def keep_pair() -> Tuple[tf.Tensor, tf.Tensor]:
        return raw_img, corrected_img

    raw_img, corrected_img = tf.cond(flip_lr, flip_left_right_pair, keep_pair)

    def flip_up_down_pair() -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.image.flip_up_down(raw_img), tf.image.flip_up_down(corrected_img)

    def keep_pair_after_lr() -> Tuple[tf.Tensor, tf.Tensor]:
        return raw_img, corrected_img

    raw_img, corrected_img = tf.cond(flip_ud, flip_up_down_pair, keep_pair_after_lr)
    raw_img = tf.image.rot90(raw_img, k=rot_k)
    corrected_img = tf.image.rot90(corrected_img, k=rot_k)
    return raw_img, corrected_img


def load_and_preprocess_image(
    raw_img_path: tf.Tensor,
    corrected_img_path: tf.Tensor,
    crop_size: int = 256,
    seed: Optional[tf.Tensor] = None,
    augment: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    raw_img = _read_png(raw_img_path)
    corrected_img = _read_png(corrected_img_path)

    if seed is None:
        seed = tf.constant([100, 0], dtype=tf.int32)

    raw_img, corrected_img = _stateless_random_crop_pair(raw_img, corrected_img, crop_size, seed)
    if augment:
        raw_img, corrected_img = _stateless_augment_pair(raw_img, corrected_img, seed)

    return _normalize_to_minus_one_one(raw_img), _normalize_to_minus_one_one(corrected_img)


def load_image_test(image_path: tf.Tensor, crop_margin: int = 0) -> tf.Tensor:
    image = _read_png(image_path)

    if crop_margin > 0:
        original_shape = tf.shape(image)
        new_height = original_shape[0] - 2 * crop_margin
        new_width = original_shape[1] - 2 * crop_margin
        image = tf.image.crop_to_bounding_box(image, crop_margin, crop_margin, new_height, new_width)

    return _normalize_to_minus_one_one(image)


def get_datasets(
    raw_image_path: str,
    corrected_image_path: str,
    batch_size: int = 1,
    crop_size: int = 256,
    seed: int = 100,
    augment: bool = True,
) -> tf.data.Dataset:
    raw_files = list_image_files(raw_image_path)
    corrected_files = list_image_files(corrected_image_path)
    validate_paired_files(raw_files, corrected_files)

    dataset = tf.data.Dataset.from_tensor_slices((raw_files, corrected_files))
    dataset = dataset.shuffle(buffer_size=len(raw_files), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.enumerate()

    def map_fn(index: tf.Tensor, paths: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        example_seed = tf.stack([tf.cast(seed, tf.int32), tf.cast(index % (2**31 - 1), tf.int32)])
        return load_and_preprocess_image(paths[0], paths[1], crop_size, example_seed, augment)

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_datasets_metrics(
    raw_image_path: str,
    corrected_image_path: str,
    crop_margin: int = 0,
    batch_size: int = 1,
) -> tf.data.Dataset:
    raw_files = list_image_files(raw_image_path)
    corrected_files = list_image_files(corrected_image_path)
    validate_paired_files(raw_files, corrected_files)

    raw_dataset = tf.data.Dataset.from_tensor_slices(raw_files).map(
        lambda x: load_image_test(x, crop_margin),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    corrected_dataset = tf.data.Dataset.from_tensor_slices(corrected_files).map(
        lambda x: load_image_test(x, crop_margin),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = tf.data.Dataset.zip((raw_dataset, corrected_dataset))
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def get_file_names(pattern: str) -> List[str]:
    return [Path(path).name for path in list_image_files(pattern)]
