import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


def rgb_to_y_scotopic_tf(rgb: tf.Tensor) -> tf.Tensor:
    """Approximate scotopic luminance from RGB values in [0, 1]."""
    r, g, b = tf.split(rgb, 3, axis=-1)
    return 0.06 * r + 0.63 * g + 0.31 * b


def rgb_to_ycbcr_tf(rgb: tf.Tensor) -> tf.Tensor:
    r, g, b = tf.split(rgb, 3, axis=-1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return tf.concat([y, cb, cr], axis=-1)


def chrominance_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_ycbcr = rgb_to_ycbcr_tf(y_true)
    y_pred_ycbcr = rgb_to_ycbcr_tf(y_pred)
    return tf.reduce_mean(tf.abs(y_true_ycbcr[..., 1:] - y_pred_ycbcr[..., 1:]))


def luminance_consistency_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_gray = tf.image.rgb_to_grayscale(y_true)
    y_pred_gray = tf.image.rgb_to_grayscale(y_pred)
    return tf.reduce_mean(tf.abs(y_true_gray - y_pred_gray))


def load_vgg(layer_name: str = "block3_conv3") -> Model:
    """Load the VGG19 feature extractor used for perceptual loss."""
    weights_path = "/home/tpuuser08/Czh/weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    vgg = VGG19(include_top=False, weights=None, input_shape=(None, None, 3))
    vgg.load_weights(weights_path)
    vgg.trainable = False
    return Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output, name="vgg19_perceptual_loss")


def perceptual_loss(y_true: tf.Tensor, y_pred: tf.Tensor, loss_model: Model) -> tf.Tensor:
    y_true_vgg = preprocess_input(tf.clip_by_value(y_true, 0.0, 1.0) * 255.0)
    y_pred_vgg = preprocess_input(tf.clip_by_value(y_pred, 0.0, 1.0) * 255.0)
    return tf.reduce_mean(tf.square(loss_model(y_true_vgg) - loss_model(y_pred_vgg)))


def smooth_l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    diff = tf.abs(y_true - y_pred)
    quadratic = tf.minimum(diff, 1.0)
    linear = diff - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + linear)


def loss(y_true: tf.Tensor, y_pred: tf.Tensor, loss_model: Model, model: tf.keras.Model) -> tf.Tensor:
    """Compute the uncertainty-weighted training objective."""
    y_true_normalized = (y_true + 1.0) / 2.0
    y_pred_normalized = (y_pred + 1.0) / 2.0

    base_losses = {
        "l1": smooth_l1_loss(y_true_normalized, y_pred_normalized),
        "perc": perceptual_loss(y_true_normalized, y_pred_normalized, loss_model),
        "lum": luminance_consistency_loss(y_true_normalized, y_pred_normalized),
        "chroma": chrominance_loss(y_true_normalized, y_pred_normalized),
    }

    log_vars = {
        "l1": model.log_var_l1,
        "perc": model.log_var_perc,
        "lum": model.log_var_lum,
        "chroma": model.log_var_chroma,
    }

    total_loss = 0.0
    for name, base_loss in base_losses.items():
        log_var = log_vars[name]
        total_loss += tf.exp(-log_var) * base_loss + log_var

    return tf.reduce_mean(total_loss)
