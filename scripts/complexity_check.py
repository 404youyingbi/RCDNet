import argparse
import ast
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

from model.arch import RCDNet

tf.get_logger().setLevel(logging.ERROR)


def get_time() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_shape(shape_text: str) -> Tuple[int, int, int]:
    try:
        value = ast.literal_eval(shape_text)
    except (SyntaxError, ValueError):
        value = tuple(int(part.strip()) for part in shape_text.split(","))
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise argparse.ArgumentTypeError("shape must have the form '(H,W,C)'")
    shape = tuple(int(dim) for dim in value)
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("all shape dimensions must be positive")
    return shape


def get_total_params(model: tf.keras.Model) -> Tuple[int, int]:
    trainable_params = sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights)
    return trainable_params, non_trainable_params


def compute_flops(model: tf.keras.Model, input_shape: Tuple[int, int, int]) -> int:
    input_layer = tf.keras.Input(shape=input_shape)
    outputs = model(input_layer, training=False)
    profiled_model = tf.keras.Model(inputs=input_layer, outputs=outputs)

    concrete = tf.function(lambda inputs: profiled_model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *tensor.shape[1:]]) for tensor in profiled_model.inputs]
    )
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(frozen_func.graph.as_graph_def(), name="")
        run_meta = tf.compat.v1.RunMetadata()
        options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=options)
        return int(flops.total_float_ops) if flops is not None else 0


def compute_complexity(shape: Tuple[int, int, int] = (256, 256, 3)) -> None:
    print("RCDNet complexity check")
    print(f"({get_time()}) Input shape: {shape}")
    model = RCDNet(filters=32, num_rab_blocks=4)
    model.build(input_shape=(None, *shape))
    flops = compute_flops(model, shape)
    trainable_params, non_trainable_params = get_total_params(model)
    print(f"FLOPs: {flops / (1024 ** 3):.2f} G")
    print(f"Trainable params: {trainable_params}")
    print(f"Non-trainable params: {non_trainable_params}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute RCDNet FLOPs and parameter count.")
    parser.add_argument("--shape", type=parse_shape, default=(256, 256, 3), help="Input shape, e.g. '(256,256,3)'.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    compute_complexity(args.shape)
