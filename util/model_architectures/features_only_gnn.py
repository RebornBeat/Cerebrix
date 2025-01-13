import tensorflow as tf
from .base import BaseArchitecture
from ..decision_layers.graph_neural_network import GraphNeuralNetwork

class FeaturesGNN(BaseArchitecture):
    def __init__(self,
                 num_classes: int,
                 num_gcn_layers: int = 3,
                 gcn_units: int = 64,
                 dropout_rate: float = 0.5):
        super().__init__(num_classes)
        self.num_gcn_layers = num_gcn_layers
        self.gcn_units = gcn_units
        self.dropout_rate = dropout_rate

    def build(self) -> tf.keras.Model:
        # Feature input
        feature_input = tf.keras.Input(shape=(None,), name='feature_input')

        # Create feature relationship graph
        adj_matrix = self.create_feature_graph(feature_input)

        # Process features as graph nodes
        x = tf.keras.layers.Dense(self.gcn_units)(feature_input)
        x = tf.keras.layers.LayerNormalization()(x)

        # Apply GNN
        gnn = GraphNeuralNetwork(
            num_classes=self.num_classes,
            num_features=self.gcn_units,
            num_gcn_layers=self.num_gcn_layers,
            gcn_units=self.gcn_units,
            dropout_rate=self.dropout_rate
        )

        outputs = gnn([x, adj_matrix])

        return tf.keras.Model(
            inputs=feature_input,
            outputs=outputs,
            name='features_gnn'
        )

    def create_feature_graph(self, features):
        # Create adjacency matrix based on feature relationships
        similarity = tf.matmul(features, features, transpose_b=True)
        adj_matrix = tf.keras.layers.Activation('sigmoid')(similarity)
        return adj_matrix

    def save_model(self, models_dir: str, identifier: str):
        """Save FeaturesGNN model"""
        model_path = os.path.join(models_dir, f'features_gnn_{identifier}.h5')
        self.model.save(model_path)
        mlflow.keras.log_model(self.model, f"features_gnn_{identifier}")
