import unittest
import pandas as pd
import numpy as np
from scripts.xgboost_trainer import XGBoostTrainer
from scripts.visualizer import XGBoostVisualizer

class TestXGBoostTrainer(unittest.TestCase):
    def setUp(self):
        """Set up test data and initialize XGBoostTrainer."""
        # Create sample dataset
        self.data = pd.DataFrame({
            "TimeStamp": pd.date_range(start="2024-01-01", periods=100, freq="H"),
        })
        self.data.set_index("TimeStamp", inplace=True)
        self.data["Feature1"] = np.random.rand(100)
        self.data["Feature2"] = np.random.rand(100)
        self.data["Target"] = np.random.choice(["A", "B", "C", "D", "E"], size=100)

        # Initialize visualizer and trainer
        self.visualizer = XGBoostVisualizer()
        self.trainer = XGBoostTrainer(self.visualizer)

    def test_preprocess_data(self):
        """Test the preprocess_data method."""
        X_train, X_test, y_train, y_test, label_encoder = self.trainer.preprocess_data(
            self.data, target_column="Target", split_date="2024-01-03"
        )
        self.assertEqual(len(X_train), 48)  # Check train size
        self.assertEqual(len(X_test), 52)  # Check test size
        self.assertTrue("Feature1" in X_train.columns)  # Check if features exist
        self.assertTrue("Feature2" in X_train.columns)

    def test_train_model(self):
        """Test the train_model method."""
        X_train, _, y_train, _, _ = self.trainer.preprocess_data(
            self.data, target_column="Target", split_date="2024-01-03"
        )
        model = self.trainer.train_model(X_train, y_train)
        self.assertIsNotNone(model)  # Ensure the model is trained

    def test_evaluate_model(self):
        """Test the evaluate_model method."""
        X_train, X_test, y_train, y_test, label_encoder = self.trainer.preprocess_data(
            self.data, target_column="Target", split_date="2024-01-03"
        )
        model = self.trainer.train_model(X_train, y_train)
        y_pred = self.trainer.evaluate_model(model, X_test, y_test, label_encoder)
        self.assertEqual(len(y_pred), len(y_test))  # Ensure predictions match test set size

    def test_log_metrics(self):
        """Test the log_metrics method."""
        y_test = pd.Series([0, 1, 1, 2, 0], index=pd.date_range(start="2024-01-01", periods=5, freq="H"))
        y_pred = np.array([0.1, 1.2, 0.9, 2.0, 0.2])  # Predictions close to true labels
        metrics = self.trainer.log_metrics(y_test, y_pred)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertGreater(metrics["r2"], 0)  # R² should be > 0 for a reasonable fit


if __name__ == "__main__":
    unittest.main()
