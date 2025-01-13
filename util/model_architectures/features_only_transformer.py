import tensorflow as tf
from .base import BaseArchitecture
from ..decision_layers.transformer_decoder import TransformerDecoder

class FeaturesTransformer(BaseArchitecture):
    def __init__(self,
                 num_classes: int,
                 d_model: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self) -> tf.keras.Model:
        # Feature input
        feature_input = tf.keras.Input(shape=(None,), name='feature_input')

        # Feature embedding
        x = tf.keras.layers.Dense(self.d_model)(feature_input)
        x = tf.keras.layers.LayerNormalization()(x)

        # Transformer decoder
        transformer = TransformerDecoder(
            num_classes=self.num_classes,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )

        outputs = transformer(x)

        return tf.keras.Model(
            inputs=feature_input,
            outputs=outputs,
            name='features_transformer'
        )
