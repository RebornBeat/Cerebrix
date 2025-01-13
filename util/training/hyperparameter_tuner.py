import tensorflow as tf
import kerastuner as kt

class ModelTuner:
    def __init__(self,
                 model_architecture,
                 tuning_frequency: int = 5,
                 max_trials: int = 10):
        self.model_architecture = model_architecture
        self.tuning_frequency = tuning_frequency
        self.max_trials = max_trials

    def tune_model(self, data, validation_data, stack_size=None):
        # Perform hyperparameter tuning every `tuning_frequency` epochs or at the start
        """Tune model hyperparameters"""
        tuner = kt.Hyperband(
            self.model_architecture,
            objective='val_accuracy',
            max_epochs=50,
            factor=3,
            directory=f'model_tuning_{stack_size if stack_size else "single"}',
            project_name=f'model_tuning_{stack_size if stack_size else "single"}'
        )

        tuner.search(
            data,
            validation_data=validation_data,
            epochs=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3
                )
            ]
        )

        return tuner.get_best_hyperparameters(num_trials=1)[0]
