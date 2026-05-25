import argparse
import datetime as dt
import os
import random
import sys
from pathlib import Path
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf

import data_loading as dl
from model.arch import RCDNet
from model.losses import load_vgg, loss
from model.scheduler import CosineDecayWithRestartsLearningRateSchedule

DATASET_CHOICES = ("LOLv1", "LOLv2_Real", "LOLv2_Synthetic")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_time() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_dataset_paths(dataset: str) -> Tuple[str, str, str, str]:
    if dataset == "LOLv1":
        return (
            "./data/LOLv1/Train/input/*.png",
            "./data/LOLv1/Train/target/*.png",
            "./data/LOLv1/Test/input/*.png",
            "./data/LOLv1/Test/target/*.png",
        )
    if dataset == "LOLv2_Real":
        return (
            "./data/LOLv2/Real_captured/Train/Low/*.png",
            "./data/LOLv2/Real_captured/Train/Normal/*.png",
            "./data/LOLv2/Real_captured/Test/Low/*.png",
            "./data/LOLv2/Real_captured/Test/Normal/*.png",
        )
    if dataset == "LOLv2_Synthetic":
        return (
            "./data/LOLv2/Synthetic/Train/Low/*.png",
            "./data/LOLv2/Synthetic/Train/Normal/*.png",
            "./data/LOLv2/Synthetic/Test/Low/*.png",
            "./data/LOLv2/Synthetic/Test/Normal/*.png",
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


@tf.function
def train_step(raw_images, corrected_images, model, loss_model, optimizer):
    with tf.GradientTape() as tape:
        model_outputs = model(raw_images, training=True)
        y_pred = model_outputs[0] if isinstance(model_outputs, (tuple, list)) else model_outputs
        loss_value = loss(corrected_images, y_pred, loss_model, model)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    gradients_and_variables = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    optimizer.apply_gradients(gradients_and_variables)
    return loss_value


def evaluate(model, test_dataset):
    psnr_metric = tf.keras.metrics.Mean()
    ssim_metric = tf.keras.metrics.Mean()

    for raw_images, corrected_images in test_dataset:
        generated_images = model(raw_images, training=False)
        generated_images = (generated_images + 1.0) / 2.0
        corrected_images = (corrected_images + 1.0) / 2.0

        psnr_metric.update_state(tf.reduce_mean(tf.image.psnr(corrected_images, generated_images, max_val=1.0)))
        ssim_metric.update_state(tf.reduce_mean(tf.image.ssim(corrected_images, generated_images, max_val=1.0)))

    return (
        float(psnr_metric.result().numpy()),
        float(ssim_metric.result().numpy()),
    )


def train(model, dataset, loss_model, optimizer, train_dataset, test_dataset, epochs, start_epoch=1):
    best_psnr = float("-inf")
    output_dir = Path("./experiments") / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        loss_metric = tf.keras.metrics.Mean()
        for raw_images, corrected_images in train_dataset:
            loss_metric.update_state(train_step(raw_images, corrected_images, model, loss_model, optimizer))

        avg_psnr, avg_ssim = evaluate(model, test_dataset)
        avg_loss = float(loss_metric.result().numpy())

        print(
            f"({get_time()}) Epoch {epoch} | PSNR: {avg_psnr:.2f} | "
            f"SSIM: {avg_ssim:.3f} | loss: {avg_loss:.6f}",
            flush=True,
        )

        if avg_psnr > best_psnr:
            print(f"({get_time()}) New best PSNR: {avg_psnr:.2f} (previously {best_psnr:.2f}).", flush=True)
            best_psnr = avg_psnr
            model_name = f"RCDNet_psnr_{avg_psnr:.2f}_ssim_{avg_ssim:.3f}_epoch_{epoch}_dataset_{dataset}.weights.h5"
            model.save_weights(str(output_dir / model_name))
            print(f"({get_time()}) Saved model: {output_dir / model_name}", flush=True)

def start_train(dataset: str, epochs: int, batch_size: int, seed: int):
    set_seed(seed)
    print(f"RCDNet training | dataset: {dataset}")

    raw_image_path, corrected_image_path, raw_test_path, corrected_test_path = get_dataset_paths(dataset)
    print(f"({get_time()}) Loading dataset.")
    train_dataset = dl.get_datasets(
        raw_image_path,
        corrected_image_path,
        batch_size=batch_size,
        seed=seed,
        augment=True,
    )
    test_dataset = dl.get_datasets_metrics(raw_test_path, corrected_test_path, batch_size=1)
    print(f"({get_time()}) Dataset loaded.")

    model = RCDNet(filters=32, num_rab_blocks=4)
    model(tf.zeros((1, 256, 256, 3), dtype=tf.float32), training=False)
    loss_model = load_vgg()

    initial_lr = 2e-4
    min_lr = 1e-6
    steps_per_epoch = int(tf.data.experimental.cardinality(train_dataset).numpy())
    if steps_per_epoch <= 0:
        raise ValueError("The training dataset is empty or has unknown cardinality.")

    total_steps = epochs * steps_per_epoch
    first_decay_steps = 150 * steps_per_epoch
    learning_rate = CosineDecayWithRestartsLearningRateSchedule(
        initial_lr=initial_lr,
        min_lr=min_lr,
        total_steps=total_steps,
        first_decay_steps=first_decay_steps,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print(f"({get_time()}) Starting training.")
    train(
        model=model,
        dataset=dataset,
        loss_model=loss_model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=epochs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RCDNet.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    start_train(args.dataset, args.epochs, args.batch_size, args.seed)
