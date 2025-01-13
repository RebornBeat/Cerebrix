import tensorflow as tf
from .base import BaseArchitecture
import kerastuner as kt
from typing import Tuple

class CNNLSTMRaw(BaseArchitecture):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, ...],
                 use_attention: bool = True,
                 use_transformer: bool = True,
                 tunable: bool = True):
        super().__init__(num_classes, input_shape=input_shape)
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        self.tunable = tunable

    def build(self) -> tf.keras.Model:
        if self.tunable:
            return self.build_tunable_model()
        return self.build_static_model()

    def build_static_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self.input_shape)

        # CNN branch
        x_cnn = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x_cnn = tf.keras.layers.MaxPooling2D((2, 2))(x_cnn)
        x_cnn = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x_cnn)
        x_cnn = tf.keras.layers.MaxPooling2D((2, 2))(x_cnn)

        # Add attention if enabled
        if self.use_attention:
            temp_attention = tf.keras.layers.Attention()([x_cnn, x_cnn])
            x_cnn = tf.keras.layers.Concatenate()([x_cnn, temp_attention])

        # Reshape for LSTM
        x_cnn = tf.keras.layers.Reshape((-1, x_cnn.shape[-1]))(x_cnn)

        # LSTM layers
        x_cnn = tf.keras.layers.LSTM(128, return_sequences=True)(x_cnn)
        x_cnn = tf.keras.layers.Dropout(0.5)(x_cnn)
        x_cnn = tf.keras.layers.LSTM(128)(x_cnn)
        x_cnn = tf.keras.layers.Dropout(0.5)(x_cnn)

        if self.use_transformer:
            # Transformer branch
            x_transformer = inputs
            for _ in range(2):  # Default 2 transformer blocks
                x_transformer = tf.keras.layers.MultiHeadAttention(
                    num_heads=4,
                    key_dim=32
                )(x_transformer, x_transformer)
                x_transformer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_transformer)
                x_transformer = tf.keras.layers.Conv1D(
                    filters=32,
                    kernel_size=1,
                    activation="relu"
                )(x_transformer)
                x_transformer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_transformer)
            x_transformer = tf.keras.layers.GlobalAveragePooling1D()(x_transformer)

            # Combine branches
            x = tf.keras.layers.Concatenate()([x_cnn, x_transformer])
        else:
            x = x_cnn

        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_lstm_raw')

    def build_tunable_model(self) -> kt.HyperModel:
        def create_model(hp):
            inputs = tf.keras.Input(shape=self.input_shape)

            # CNN branch with tunable parameters
            x_cnn = tf.keras.layers.Conv2D(
                hp.Int('cnn_filters_1', 32, 128, step=32),
                hp.Int('cnn_kernel_1', 3, 9, step=2),
                activation='relu'
            )(inputs)
            x_cnn = tf.keras.layers.MaxPooling2D((2, 2))(x_cnn)
            x_cnn = tf.keras.layers.Conv2D(
                hp.Int('cnn_filters_2', 64, 256, step=64),
                hp.Int('cnn_kernel_2', 3, 9, step=2),
                activation='relu'
            )(x_cnn)
            x_cnn = tf.keras.layers.MaxPooling2D((2, 2))(x_cnn)

            if self.use_attention:
                temp_attention = tf.keras.layers.Attention()([x_cnn, x_cnn])
                x_cnn = tf.keras.layers.Concatenate()([x_cnn, temp_attention])

            x_cnn = tf.keras.layers.Reshape((-1, x_cnn.shape[-1]))(x_cnn)

            # LSTM layers with tunable units
            x_cnn = tf.keras.layers.LSTM(
                hp.Int('lstm_units_1', 32, 128, step=32),
                return_sequences=True
            )(x_cnn)
            x_cnn = tf.keras.layers.LSTM(
                hp.Int('lstm_units_2', 16, 64, step=16)
            )(x_cnn)

            if self.use_transformer:
                # Transformer branch with tunable parameters
                x_transformer = inputs
                for _ in range(hp.Int('num_transformer_blocks', 1, 3)):
                    x_transformer = tf.keras.layers.MultiHeadAttention(
                        num_heads=hp.Int('num_heads', 2, 8),
                        key_dim=hp.Int('key_dim', 16, 64, step=16)
                    )(x_transformer, x_transformer)
                    x_transformer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_transformer)
                    x_transformer = tf.keras.layers.Conv1D(
                        filters=hp.Int('transformer_conv_filters', 16, 64, step=16),
                        kernel_size=1,
                        activation="relu"
                    )(x_transformer)
                    x_transformer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_transformer)
                x_transformer = tf.keras.layers.GlobalAveragePooling1D()(x_transformer)

                # Combine branches
                x = tf.keras.layers.Concatenate()([x_cnn, x_transformer])
            else:
                x = x_cnn

            x = tf.keras.layers.Dense(
                hp.Int('dense_units', 32, 128, step=32),
                activation='relu'
            )(x)
            outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
                ),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model

        return create_model
