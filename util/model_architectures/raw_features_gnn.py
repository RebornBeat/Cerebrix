import tensorflow as tf
from .base import BaseArchitecture
from ..decision_layers.graph_neural_network import GraphNeuralNetwork
from typing import Tuple

class RawFeaturesGNN(BaseArchitecture):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, ...],
                 num_gcn_layers: int = 3,
                 gcn_units: int = 64,
                 dropout_rate: float = 0.5):
        super().__init__(num_classes, input_shape)
        self.num_gcn_layers = num_gcn_layers
        self.gcn_units = gcn_units
        self.dropout_rate = dropout_rate

    def build(self) -> tf.keras.Model:
        # Raw data input
        raw_input = tf.keras.Input(shape=self.input_shape, name='raw_input')

        # Feature input
        feature_input = tf.keras.Input(shape=(None,), name='feature_input')

        # Create graph structure from raw data
        adj_matrix = self.create_adjacency_matrix(raw_input)

        # Process raw data as graph nodes
        x_raw = self.process_raw_data_as_nodes(raw_input)

        # Process features as additional node attributes
        x_features = self.process_features_as_nodes(feature_input)

        # Combine node features
        combined_nodes = self.combine_node_features(x_raw, x_features)

        # Apply GNN
        gnn = GraphNeuralNetwork(
            num_classes=self.num_classes,
            num_features=combined_nodes.shape[-1],
            num_gcn_layers=self.num_gcn_layers,
            gcn_units=self.gcn_units,
            dropout_rate=self.dropout_rate
        )

        outputs = gnn([combined_nodes, adj_matrix])

        return tf.keras.Model(
            inputs=[raw_input, feature_input],
            outputs=outputs,
            name='raw_features_gnn'
        )

    def create_adjacency_matrix(self, raw_input):
        # Create adjacency matrix based on channel connectivity
        # This could be based on physical channel locations or signal correlations
        channels = raw_input.shape[1]
        adj_matrix = tf.keras.layers.Dense(channels)(raw_input)
        adj_matrix = tf.keras.layers.Activation('sigmoid')(adj_matrix)
        return adj_matrix

    def process_raw_data_as_nodes(self, raw_input):
        # Process each channel as a node
        x = tf.keras.layers.Conv1D(32, 3, activation='relu')(raw_input)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
        return x

    def process_features_as_nodes(self, feature_input):
        # Transform features into node attributes
        x = tf.keras.layers.Dense(64, activation='relu')(feature_input)
        x = tf.keras.layers.LayerNormalization()(x)
        return x

    def combine_node_features(self, x_raw, x_features):
        # Combine raw and feature representations for each node
        x_raw_shaped = tf.keras.layers.Dense(64)(x_raw)
        x_features_shaped = tf.keras.layers.Dense(64)(x_features)
        combined = tf.keras.layers.Concatenate()([x_raw_shaped, x_features_shaped])
        return combined
