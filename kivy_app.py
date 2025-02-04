import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.datepicker import DatePicker
from kivy.uix.textinput import TextInput
from scripts.visualizer import XGBoostVisualizer
from scripts.xgboost_trainer import XGBoostTrainer

class ModelApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        # Date selection inputs
        self.add_widget(Label(text="Select Start Date:"))
        self.start_date_picker = DatePicker()
        self.add_widget(self.start_date_picker)

        self.add_widget(Label(text="Select End Date:"))
        self.end_date_picker = DatePicker()
        self.add_widget(self.end_date_picker)

        # Test period input
        self.add_widget(Label(text="Enter Test Period (hours):"))
        self.period_input = TextInput(text="100", multiline=False)
        self.add_widget(self.period_input)

        # Button to trigger model evaluation
        self.run_button = Button(text="Run Model")
        self.run_button.bind(on_press=self.run_model)
        self.add_widget(self.run_button)

        # Label to display results
        self.result_label = Label(text="Results will be displayed here.")
        self.add_widget(self.result_label)

    def run_model(self, instance):
        """Process user inputs and run the model on selected dataset."""
        # Get user input values
        start_date = self.start_date_picker.text
        end_date = self.end_date_picker.text
        test_period = int(self.period_input.text)

        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Generate dataset
        data = pd.DataFrame({
            "TimeStamp": pd.date_range(start=start_date, end=end_date, freq="H"),
        })
        data.set_index("TimeStamp", inplace=True)

        # Generate random target classes
        data["Target"] = np.random.choice(["A", "B", "C", "D", "E"], size=len(data))

        # Determine split date for test dataset
        split_date = data.index[-test_period]

        # Run the model
        visualizer = XGBoostVisualizer()
        trainer = XGBoostTrainer(visualizer)
        results = trainer.train(data, target_column="Target", split_date=split_date)

        # Display results
        self.result_label.text = f"RMSE: {results['rmse']:.4f}, R²: {results['r2']:.4f}"

        # Show confusion matrix
        y_test = data.loc[split_date:, "Target"]
        y_pred = np.random.choice(["A", "B", "C", "D", "E"], size=len(y_test))  # Mock predictions
        cm = confusion_matrix(y_test, y_pred, labels=["A", "B", "C", "D", "E"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A", "B", "C", "D", "E"])
        disp.plot(cmap='viridis')
        plt.title("Confusion Matrix")
        plt.show()

class ModelAppLauncher(App):
    def build(self):
        return ModelApp()

if __name__ == "__main__":
    ModelAppLauncher().run()
