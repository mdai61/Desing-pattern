import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .base_trainer import BaseVisualizer

class XGBoostVisualizer(BaseVisualizer):
    def plot_true_vs_predicted(self, y_true, y_pred, title="True vs Predicted Values"):
        """Plot true vs. predicted values."""
        plt.figure(figsize=(10, 6))
        plt.plot(y_true.index, y_true, label="True Values", alpha=0.8, marker='o')
        plt.plot(y_true.index, y_pred, label="Predicted Values", alpha=0.8, linestyle='--', marker='x')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Target")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """Plot confusion matrix."""
        y_pred_rounded = np.round(y_pred).astype(int)
        cm = confusion_matrix(y_true, y_pred_rounded)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap='viridis', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.show()
