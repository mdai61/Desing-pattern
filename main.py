import pandas as pd
import numpy as np
from scripts.visualizer import XGBoostVisualizer
from scripts.xgboost_trainer import XGBoostTrainer

if __name__ == "__main__":
    # Simplified dataset
    data = pd.DataFrame({
        "TimeStamp": pd.date_range(start="2024-01-01", periods=500, freq="H"),
    })
    data.set_index("TimeStamp", inplace=True)

    # Generate random target classes
    data["Target"] = np.random.choice(["A", "B", "C", "D", "E"], size=500)

    # Create a visualizer and trainer
    visualizer = XGBoostVisualizer()
    trainer = XGBoostTrainer(visualizer)

    # Train the model
    results = trainer.train(data, target_column="Target", split_date="2024-11-01")

    # Output the results
    print(results)  # Logs RMSE and R² metrics
