import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class VisualizationTools:
    @staticmethod
    def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                   xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def plot_training_comparison(results, approaches):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = list(results[approaches[0]].keys())

        for i, metric in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            for approach in approaches:
                values = results[approach][metric]
                mean = np.mean(values, axis=0)
                std = np.std(values, axis=0)

                ax.plot(range(len(mean)), mean, label=approach)
                ax.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.3)

            ax.set_title(metric)
            ax.legend()

        plt.tight_layout()
        return fig

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(self.models_dir, 'training_history.png')
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        plt.show()

    def visualize_intent_and_commitment(self, predictions, commitments, timestamps, intents):
        plt.figure(figsize=(12, 8))

        # Plot predicted actions
        plt.subplot(3, 1, 1)
        action_indices = [self.actions.index(action) for action in predictions]
        plt.plot(timestamps, action_indices, marker='o')
        plt.yticks(range(len(self.actions)), self.actions)
        plt.title('Predicted Actions Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Predicted Action')

        # Plot commitment levels
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, commitments, marker='o')
        plt.title('Commitment Levels Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Commitment Level')

        # Plot detected intents
        plt.subplot(3, 1, 3)
        for intent in intents:
            action, start_time, end_time = intent
            plt.axvspan(start_time, end_time, alpha=0.3, label=action)
        plt.title('Detected Intents')
        plt.xlabel('Time (s)')
        plt.ylabel('Intent')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_clusters(self, action):
        if action not in self.ica_components or action not in self.cluster_labels:
            self.log_and_emit(f"No data available for action: {action}")
            return

        ica_data = np.vstack([batch.reshape(-1, batch.shape[-1]) for batch in self.ica_components[action]])
        cluster_labels = np.concatenate([labels.flatten() for labels in self.cluster_labels[action]])

        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(ica_data)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f"t-SNE Visualization of EEG Clusters for {action}")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.show()
