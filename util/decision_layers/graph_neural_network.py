import tensorflow as tf
import spektral
from .base import BaseDecisionLayer

class GraphNeuralNetwork(BaseDecisionLayer):
    def __init__(self,
                 num_classes: int,
                 num_features: int,
                 num_gcn_layers: int = 3,
                 gcn_units: int = 64,
                 dropout_rate: float = 0.5):
        super().__init__(num_classes)
        self.num_features = num_features
        self.num_gcn_layers = num_gcn_layers
        self.gcn_units = gcn_units
        self.dropout_rate = dropout_rate

    def build(self) -> tf.keras.Model:
        # Node features input
        node_features = tf.keras.Input(shape=(None, self.num_features))
        # Adjacency matrix input
        adjacency = tf.keras.Input(shape=(None, None))

        # Graph convolutional layers
        x = node_features
        for i in range(self.num_gcn_layers):
            x = spektral.layers.GCNConv(
                channels=self.gcn_units,
                activation='relu'
            )([x, adjacency])
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # Global pooling
        x = spektral.layers.GlobalAttentionPool(self.gcn_units)(x)

        # Classification
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        return tf.keras.Model(inputs=[node_features, adjacency], outputs=outputs)
