import unittest
import pandas as pd
import numpy as np
from scripts.visualizer import XGBoostVisualizer

class TestXGBoostVisualizer(unittest.TestCase):
    def setUp(self):
        """Set up sample data and initialize visualizer."""
        # Create sample data
        self.y_true = pd.Series([0, 1, 1, 2, 0], index=pd.date_range(start="2024-01-01", periods=5, freq="H"))
        self.y_pred = np.array([0.1, 1.2, 0.9, 2.0, 0.2])
        self.classes = ["A", "B", "C"]

        # Initialize visualizer
        self.visualizer = XGBoostVisualizer()

    def test_plot_true_vs_predicted(self):
        """Test the plot_true_vs_predicted method."""
        try:
            self.visualizer.plot_true_vs_predicted(self.y_true, self.y_pred, title="Test True vs Predicted")
        except Exception as e:
            self.fail(f"plot_true_vs_predicted raised an exception: {e}")

    def test_plot_confusion_matrix(self):
        """Test the plot_confusion_matrix method."""
        try:
            self.visualizer.plot_confusion_matrix(self.y_true, self.y_pred, self.classes)
        except Exception as e:
            self.fail(f"plot_confusion_matrix raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
