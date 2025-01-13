import tensorflow as tf
from .base import BaseDecisionLayer

class HierarchicalAttention(BaseDecisionLayer):
    def __init__(self,
                 num_classes: int,
                 num_features: int,
                 num_heads: int = 8,
                 ff_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__(num_classes)
        self.num_features = num_features
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self) -> tf.keras.Model:
        # Feature-level attention
        feature_input = tf.keras.Input(shape=(None, self.num_features))

        # Feature attention
        feature_attention = self.attention_block(
            feature_input,
            name='feature_attention'
        )

        # Temporal attention
        temporal_attention = self.attention_block(
            feature_attention,
            name='temporal_attention'
        )

        # Final classification
        x = tf.keras.layers.GlobalAveragePooling1D()(temporal_attention)
        x = tf.keras.layers.Dense(self.ff_dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        return tf.keras.Model(inputs=feature_input, outputs=outputs)

    def attention_block(self, x, name=None):
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.ff_dim,
            name=f'{name}_multi_head'
        )(x, x)

        # Add & Norm
        x = tf.keras.layers.LayerNormalization(
            name=f'{name}_layernorm_1'
        )(x + attention_output)

        # Feed forward
        ffn_output = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dense(x.shape[-1]),
            tf.keras.layers.Dropout(self.dropout)
        ], name=f'{name}_ffn')(x)

        # Add & Norm
        return tf.keras.layers.LayerNormalization(
            name=f'{name}_layernorm_2'
        )(x + ffn_output)
