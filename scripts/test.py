import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from tqdm import tqdm

import data_loading as dl
from model.arch import RCDNet

DATASET_CHOICES = ("LOLv1", "LOLv2_Real", "LOLv2_Synthetic")


def get_time() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_test_paths(dataset: str) -> Tuple[str, str]:
    if dataset == "LOLv1":
        return "./data/LOLv1/Test/input/*.png", "./data/LOLv1/Test/target/*.png"
    if dataset == "LOLv2_Real":
        return "./data/LOLv2/Real_captured/Test/Low/*.png", "./data/LOLv2/Real_captured/Test/Normal/*.png"
    if dataset == "LOLv2_Synthetic":
        return "./data/LOLv2/Synthetic/Test/Low/*.png", "./data/LOLv2/Synthetic/Test/Normal/*.png"
    raise ValueError(f"Unsupported dataset: {dataset}")


def start_test(dataset: str, weights: str, output_dir: str) -> None:
    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")

    print(f"RCDNet evaluation | dataset: {dataset}")
    raw_test_path, corrected_test_path = get_test_paths(dataset)
    file_names = dl.get_file_names(raw_test_path)

    print(f"({get_time()}) Loading test dataset.")
    test_dataset = dl.get_datasets_metrics(raw_test_path, corrected_test_path, crop_margin=0, batch_size=1)
    num_elements = int(tf.data.experimental.cardinality(test_dataset).numpy())
    if num_elements <= 0:
        raise ValueError("The test dataset is empty or has unknown cardinality.")
    print(f"({get_time()}) Test samples: {num_elements}")

    model = RCDNet(filters=32, num_rab_blocks=4)
    model(tf.zeros((1, 256, 256, 3), dtype=tf.float32), training=False)
    model.load_weights(str(weights_path))

    result_dir = Path(output_dir) / dataset
    result_dir.mkdir(parents=True, exist_ok=True)

    psnr_metric = tf.keras.metrics.Mean()
    ssim_metric = tf.keras.metrics.Mean()

    for sample_index, (raw_image, corrected_image) in enumerate(tqdm(test_dataset, total=num_elements)):
        generated_image = model(raw_image, training=False)
        corrected_image = (corrected_image + 1.0) / 2.0
        generated_image = (generated_image + 1.0) / 2.0

        generated_image = tf.clip_by_value(generated_image, 0.0, 1.0)
        psnr_metric.update_state(tf.reduce_mean(tf.image.psnr(corrected_image, generated_image, max_val=1.0)))
        ssim_metric.update_state(tf.reduce_mean(tf.image.ssim(corrected_image, generated_image, max_val=1.0)))

        save_path = result_dir / file_names[sample_index]
        tf.keras.utils.save_img(str(save_path), generated_image.numpy()[0], scale=True)

    print(f"PSNR: {float(psnr_metric.result().numpy()):.6f}", flush=True)
    print(f"SSIM: {float(ssim_metric.result().numpy()):.6f}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RCDNet.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights.")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory for enhanced images.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    start_test(args.dataset, args.weights, args.output_dir)
