import tensorflow as tf
from .base import BaseArchitecture
from ..decision_layers.hierarchical_attention import HierarchicalAttention

class FeaturesHAN(BaseArchitecture):
    def __init__(self,
                 num_classes: int,
                 num_heads: int = 8,
                 ff_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__(num_classes)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self) -> tf.keras.Model:
        # Feature input
        feature_input = tf.keras.Input(shape=(None,), name='feature_input')

        # Process features with hierarchical attention
        han = HierarchicalAttention(
            num_classes=self.num_classes,
            num_features=feature_input.shape[-1],
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout
        )

        outputs = han(feature_input)

        return tf.keras.Model(
            inputs=feature_input,
            outputs=outputs,
            name='features_han'
        )
