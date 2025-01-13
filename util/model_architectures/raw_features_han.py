import tensorflow as tf
from .base import BaseArchitecture
from ..decision_layers.hierarchical_attention import HierarchicalAttention
from typing import Tuple

class RawFeaturesHAN(BaseArchitecture):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, ...],
                 num_heads: int = 8,
                 ff_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__(num_classes, input_shape)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self) -> tf.keras.Model:
        # Raw data input
        raw_input = tf.keras.Input(shape=self.input_shape, name='raw_input')

        # Feature input
        feature_input = tf.keras.Input(shape=(None,), name='feature_input')

        # Process raw data with hierarchical attention
        x_raw = self.process_raw_data_hierarchical(raw_input)

        # Process features with hierarchical attention
        x_features = self.process_features_hierarchical(feature_input)

        # Combine with attention
        combined = self.hierarchical_fusion(x_raw, x_features)

        # Final classification
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(combined)

        return tf.keras.Model(
            inputs=[raw_input, feature_input],
            outputs=outputs,
            name='raw_features_han'
        )

    def process_raw_data_hierarchical(self, raw_input):
        # Channel attention
        channel_attention = self.attention_block(raw_input, 'channel')

        # Temporal attention
        temporal_attention = self.attention_block(channel_attention, 'temporal')

        return temporal_attention

    def process_features_hierarchical(self, feature_input):
        # Feature group attention
        feature_attention = self.attention_block(feature_input, 'feature_group')

        # Feature interaction attention
        interaction_attention = self.attention_block(feature_attention, 'interaction')

        return interaction_attention

    def attention_block(self, x, name):
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.ff_dim,
            name=f'{name}_attention'
        )(x, x)

        x = tf.keras.layers.LayerNormalization()(x + attention_output)

        ffn_output = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dense(x.shape[-1]),
            tf.keras.layers.Dropout(self.dropout)
        ])(x)

        return tf.keras.layers.LayerNormalization()(x + ffn_output)

    def hierarchical_fusion(self, x_raw, x_features):
        # Cross-modal attention
        fusion_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.ff_dim
        )(x_raw, x_features)

        # Combine with residual connection
        combined = tf.keras.layers.Add()([x_raw, fusion_attention])
        combined = tf.keras.layers.LayerNormalization()(combined)

        return tf.keras.layers.GlobalAveragePooling1D()(combined)
