import tensorflow as tf
from .base import BaseArchitecture
from ..decision_layers.transformer_decoder import TransformerDecoder
from typing import Tuple

class RawFeaturesTransformer(BaseArchitecture):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, ...],
                 d_model: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1):
        super().__init__(num_classes, input_shape)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self) -> tf.keras.Model:
        # Raw data input
        raw_input = tf.keras.Input(shape=self.input_shape, name='raw_input')

        # Feature input (extracted features)
        feature_input = tf.keras.Input(shape=(None,), name='feature_input')

        # Process raw data
        x_raw = self.process_raw_data(raw_input)

        # Process features
        x_features = self.process_features(feature_input)

        # Combine raw and feature representations
        combined = self.combine_representations(x_raw, x_features)

        # Transformer decoder
        transformer = TransformerDecoder(
            num_classes=self.num_classes,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )

        outputs = transformer(combined)

        return tf.keras.Model(
            inputs=[raw_input, feature_input],
            outputs=outputs,
            name='raw_features_transformer'
        )

    def process_raw_data(self, raw_input):
        # Process raw EEG data
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(raw_input)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        return x

    def process_features(self, feature_input):
        # Process extracted features
        x = tf.keras.layers.Dense(self.d_model, activation='relu')(feature_input)
        x = tf.keras.layers.LayerNormalization()(x)
        return x

    def combine_representations(self, x_raw, x_features):
        # Flatten raw data representation
        x_raw_flat = tf.keras.layers.Flatten()(x_raw)
        x_raw_proj = tf.keras.layers.Dense(self.d_model)(x_raw_flat)

        # Cross-attention between raw and feature representations
        cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model
        )(x_features, x_raw_proj)

        # Combine with residual connection
        combined = tf.keras.layers.Add()([x_features, cross_attention])
        combined = tf.keras.layers.LayerNormalization()(combined)

        return combined
