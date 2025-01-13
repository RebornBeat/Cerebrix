import tensorflow as tf
from .base import BaseDecisionLayer
import numpy as np

class TransformerDecoder(BaseDecisionLayer):
    def __init__(self,
                 num_classes: int,
                 d_model: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dff: int = 1024,
                 dropout_rate: float = 0.1):
        super().__init__(num_classes)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def build(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(None, self.d_model))

        # Positional encoding
        pos_encoding = self.positional_encoding(inputs.shape[1], self.d_model)
        x = inputs + pos_encoding

        # Decoder layers
        for _ in range(self.num_layers):
            # Self attention
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model
            )(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attention)

            # Feed forward
            ffn = self.point_wise_feed_forward_network()
            x = ffn(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # Global pooling and classification
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def point_wise_feed_forward_network(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='relu'),
            tf.keras.layers.Dense(self.d_model)
        ])

    def positional_encoding(self, position, d_model):
        # Implementation of transformer positional encoding
        angles = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
