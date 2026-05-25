import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

from model.losses import rgb_to_y_scotopic_tf


class DWT(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        h00 = np.array([0.5, 0.5])
        h01 = np.array([-0.5, 0.5])
        h10 = np.array([0.5, -0.5])
        f_ll = np.outer(h00, h00).reshape(2, 2, 1, 1)
        f_lh = np.outer(h01, h00).reshape(2, 2, 1, 1)
        f_hl = np.outer(h00, h10).reshape(2, 2, 1, 1)
        f_hh = np.outer(h01, h10).reshape(2, 2, 1, 1)
        self.filters = tf.constant(np.concatenate([f_ll, f_lh, f_hl, f_hh], axis=3), dtype=tf.float32)

    def call(self, inputs):
        output = tf.nn.conv2d(inputs, self.filters, strides=[1, 2, 2, 1], padding="SAME")
        return tf.split(output, 4, axis=-1)


class IDWT(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        h00 = np.array([0.5, 0.5])
        h01 = np.array([0.5, -0.5])
        h10 = np.array([-0.5, 0.5])
        h11 = np.array([0.5, 0.5])
        f_ll = np.outer(h00, h00).reshape(2, 2, 1, 1)
        f_lh = np.outer(h10, h00).reshape(2, 2, 1, 1)
        f_hl = np.outer(h00, h10).reshape(2, 2, 1, 1)
        f_hh = np.outer(h11, h00).reshape(2, 2, 1, 1)
        filters = tf.constant(np.concatenate([f_ll, f_lh, f_hl, f_hh], axis=2), dtype=tf.float32)
        self.filters = tf.transpose(filters, (1, 0, 2, 3))

    def call(self, inputs):
        ll, lh, hl, hh = inputs
        x = tf.concat([ll, lh, hl, hh], axis=-1)
        input_shape = tf.shape(x)
        batch_size, height, width = input_shape[0], input_shape[1], input_shape[2]
        output_shape = [batch_size, height * 2, width * 2, 1]
        output_list = []
        for i in range(4):
            kernel = self.filters[:, :, i : i + 1, :]
            output_list.append(
                tf.nn.conv2d_transpose(
                    x[..., i : i + 1],
                    kernel,
                    output_shape=output_shape,
                    strides=[1, 2, 2, 1],
                    padding="SAME",
                )
            )
        return tf.add_n(output_list)


class MacularFilter(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strength = self.add_weight(
            shape=(), initializer=tf.constant_initializer(0.1), trainable=True, name="filter_strength"
        )
        self.sigma_ratio = self.add_weight(
            shape=(), initializer=tf.constant_initializer(0.25), trainable=True, name="filter_sigma_ratio"
        )

    def call(self, inputs):
        shape = tf.shape(inputs)
        height, width = shape[1], shape[2]
        x = tf.cast(tf.linspace(-width // 2, width // 2, width), tf.float32)
        y = tf.cast(tf.linspace(-height // 2, height // 2, height), tf.float32)
        xx, yy = tf.meshgrid(x, y)
        sigma = tf.cast(width, tf.float32) * (tf.nn.relu(self.sigma_ratio) + 1e-6)
        gaussian_mask = tf.exp(-(tf.square(xx) + tf.square(yy)) / (2.0 * tf.square(sigma)))
        gaussian_mask = tf.expand_dims(tf.expand_dims(gaussian_mask, axis=0), axis=-1)
        active_strength = tf.nn.sigmoid(self.strength)
        blue_channel_multiplier = 1.0 - active_strength * gaussian_mask
        ones_mask = tf.ones_like(blue_channel_multiplier)
        multiplier = tf.concat([ones_mask, ones_mask, blue_channel_multiplier], axis=-1)
        return inputs * multiplier


class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        hidden_units = max(1, in_planes // ratio)
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc = keras.Sequential(
            [
                layers.Dense(hidden_units, activation="relu", use_bias=False),
                layers.Dense(in_planes, use_bias=False),
            ]
        )
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        weights = self.sigmoid(avg_out + max_out)
        return tf.reshape(weights, [-1, 1, 1, tf.shape(weights)[-1]])


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid")

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=3, keepdims=True)
        max_out = tf.reduce_max(x, axis=3, keepdims=True)
        x_cat = tf.concat([avg_out, max_out], axis=3)
        return self.conv1(x_cat)


class ResidualAttentionBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)
        self.act1 = layers.Activation("relu")
        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)
        self.ca = ChannelAttention(filters)
        self.sa = SpatialAttention()

    def call(self, x):
        residual = x
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return residual + x


class GatedFusionBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.gate_conv = layers.Conv2D(filters, 3, padding="same", activation="sigmoid")

    def call(self, high_level_features, low_level_features):
        gate_signal = self.gate_conv(tf.concat([high_level_features, low_level_features], axis=-1))
        return high_level_features + low_level_features * gate_signal


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.query_dense = layers.Dense(embed_size)
        self.key_dense = layers.Dense(embed_size)
        self.value_dense = layers.Dense(embed_size)
        self.combine_heads = layers.Dense(embed_size)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        return tf.matmul(attention_weights, value)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        query = self.split_heads(self.query_dense(inputs), batch_size)
        key = self.split_heads(self.key_dense(inputs), batch_size)
        value = self.split_heads(self.value_dense(inputs), batch_size)
        attention = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_size))
        output = self.combine_heads(concat_attention)
        return tf.reshape(output, [batch_size, height, width, self.embed_size])


class TransformerBlock(layers.Layer):
    def __init__(self, embed_size, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_size, num_heads)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_size),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class RCDNet(Model):
    def __init__(self, filters=32, c=5.0, num_rab_blocks=4):
        super().__init__(name="RCDNet")
        self.c = c
        self.filters = filters
        self.macular_filter = MacularFilter()
        self.max_purkinje_adjustment = 0.2
        self.purkinje_mlp = keras.Sequential(
            [
                layers.Input(shape=(1,)),
                layers.Dense(8, activation="relu"),
                layers.Dense(3, activation="tanh"),
            ],
            name="Purkinje_Controller",
        )
        self.photopic_path = self._build_luminance_path("photopic", filters, num_rab_blocks)
        self.scotopic_path = self._build_luminance_path("scotopic", filters, num_rab_blocks)
        self.fusion_fcn = keras.Sequential(
            [
                layers.Conv2D(max(1, filters // 4), 3, padding="same", activation="relu"),
                layers.Conv2D(max(1, filters // 8), 3, padding="same", activation="relu"),
                layers.Conv2D(1, 3, padding="same", activation="sigmoid"),
            ],
            name="Luminance_Fusion_FCN",
        )
        self.color_processor_entry = layers.Conv2D(filters, 3, padding="same")
        self.color_processor_body = keras.Sequential([ResidualAttentionBlock(filters) for _ in range(num_rab_blocks)])
        self.color_processor_exit = layers.Conv2D(filters, 3, padding="same")
        self.affine_scale_conv = layers.Conv2D(filters, 1, padding="same", activation="sigmoid", name="affine_scale")
        self.affine_bias_conv = layers.Conv2D(filters, 1, padding="same", activation=None, name="affine_bias")
        self.fusion_module = ResidualAttentionBlock(filters)
        self.recombine = layers.Conv2D(filters, 3, activation="relu", padding="same")
        self.final_adjustments = layers.Conv2D(3, 3, activation="tanh", padding="same")
        self.log_var_l1 = self.add_weight(shape=(), initializer="zeros", trainable=True, name="log_var_l1")
        self.log_var_perc = self.add_weight(shape=(), initializer="zeros", trainable=True, name="log_var_perc")
        self.log_var_lum = self.add_weight(shape=(), initializer="zeros", trainable=True, name="log_var_lum")
        self.log_var_chroma = self.add_weight(shape=(), initializer="zeros", trainable=True, name="log_var_chroma")

    @staticmethod
    def _build_luminance_path(name_prefix, filters, num_rab_blocks):
        return {
            "dwt": DWT(),
            "idwt": IDWT(),
            "hfe_net": keras.Sequential(
                [
                    layers.Conv2D(filters, 3, padding="same", activation="relu"),
                    ResidualAttentionBlock(filters),
                    ResidualAttentionBlock(filters),
                    layers.Conv2D(3, 3, padding="same"),
                ],
                name=f"{name_prefix}_hfe_net",
            ),
            "conv1": layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name_prefix}_lum_conv1"),
            "conv2": layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu", name=f"{name_prefix}_lum_conv2"),
            "conv3": layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu", name=f"{name_prefix}_lum_conv3"),
            "bottleneck": keras.Sequential(
                [TransformerBlock(embed_size=filters, num_heads=4, ff_dim=filters * 2) for _ in range(num_rab_blocks)],
                name=f"{name_prefix}_lum_bottleneck",
            ),
            "up2": layers.Conv2DTranspose(filters, 3, strides=2, padding="same", name=f"{name_prefix}_lum_up2"),
            "gate2": GatedFusionBlock(filters),
            "up3": layers.Conv2DTranspose(filters, 3, strides=2, padding="same", name=f"{name_prefix}_lum_up3"),
            "gate3": GatedFusionBlock(filters),
            "output_conv": layers.Conv2D(1, 3, padding="same", name=f"{name_prefix}_lum_output_conv"),
        }

    def _run_luminance_path(self, log_luminance_input, path, training=False):
        ll, lh, hl, hh = path["dwt"](log_luminance_input)
        high_freq_components = tf.concat([lh, hl, hh], axis=-1)
        l1 = path["conv1"](ll)
        l2 = path["conv2"](l1)
        l3 = path["conv3"](l2)
        b = path["bottleneck"](l3, training=training)
        d2 = path["up2"](b)
        d2_fused = path["gate2"](d2, l2)
        d3 = path["up3"](d2_fused)
        d3_fused = path["gate3"](d3, l1)
        enhanced_ll = path["output_conv"](d3_fused)
        hf_residual = path["hfe_net"](high_freq_components, training=training)
        enhanced_hf = high_freq_components + hf_residual
        enhanced_lh, enhanced_hl, enhanced_hh = tf.split(enhanced_hf, 3, axis=-1)
        return path["idwt"]([enhanced_ll, enhanced_lh, enhanced_hl, enhanced_hh])

    def call(self, inputs, training=False):
        inputs_normalized = (inputs + 1.0) / 2.0
        inputs_filtered = self.macular_filter(inputs_normalized)
        luminance_weights = tf.constant([0.299, 0.587, 0.114], dtype=inputs_filtered.dtype)
        luminance_weights = tf.reshape(luminance_weights, [1, 1, 1, 3])
        global_luminance = tf.reduce_mean(
            tf.reduce_sum(inputs_filtered * luminance_weights, axis=-1, keepdims=True),
            axis=[1, 2, 3],
            keepdims=True,
        )
        darkness_factor = 1.0 - global_luminance
        darkness_input = tf.reshape(darkness_factor, (-1, 1))
        channel_adjustments = self.purkinje_mlp(darkness_input, training=training)
        channel_weights = 1.0 + self.max_purkinje_adjustment * channel_adjustments
        purkinje_weights = tf.reshape(channel_weights, (-1, 1, 1, 3))

        photopic_luminance_input = tf.image.rgb_to_grayscale(inputs_filtered)
        scotopic_luminance_input = rgb_to_y_scotopic_tf(inputs_filtered)
        log_photopic_lum_input = tf.math.log(1.0 + self.c * tf.clip_by_value(photopic_luminance_input, 1e-7, 1.0))
        log_scotopic_lum_input = tf.math.log(1.0 + self.c * tf.clip_by_value(scotopic_luminance_input, 1e-7, 1.0))
        log_color_input = tf.math.log(1.0 + self.c * tf.clip_by_value(inputs_filtered, 1e-7, 1.0))
        log_color_input_adjusted = log_color_input * purkinje_weights

        enhanced_log_lum_photopic = self._run_luminance_path(log_photopic_lum_input, self.photopic_path, training=training)
        enhanced_log_lum_scotopic = self._run_luminance_path(log_scotopic_lum_input, self.scotopic_path, training=training)
        darkness_map = 1.0 - photopic_luminance_input
        fusion_weights_scotopic = self.fusion_fcn(darkness_map, training=training)
        enhanced_log_lum_feat = (
            fusion_weights_scotopic * enhanced_log_lum_scotopic
            + (1.0 - fusion_weights_scotopic) * enhanced_log_lum_photopic
        )

        color_feat = self.color_processor_entry(log_color_input_adjusted)
        color_feat = self.color_processor_body(color_feat, training=training)
        log_color_feat = self.color_processor_exit(color_feat)
        scale = self.affine_scale_conv(enhanced_log_lum_feat) * 2.0
        bias = self.affine_bias_conv(enhanced_log_lum_feat)
        affine_transformed_color_feat = log_color_feat * scale + bias
        fused_feat = self.fusion_module(affine_transformed_color_feat)
        recombined_feat = self.recombine(tf.concat([fused_feat, enhanced_log_lum_feat], axis=-1))
        tanh_output = self.final_adjustments(recombined_feat)
        log_c = tf.math.log(1.0 + self.c)
        log_output = (tanh_output + 1.0) / 2.0 * log_c
        linear_output = (tf.math.exp(log_output) - 1.0) / self.c
        final_output = tf.clip_by_value(linear_output, 0.0, 1.0) * 2.0 - 1.0

        if training:
            return final_output, enhanced_log_lum_feat, log_color_feat
        return final_output
