from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, visualizer):
        self.visualizer = visualizer

    def train(self, data, target_column="Target", split_date="2024-11-01"):
        """Template method for training."""
        X_train, X_test, y_train, y_test, label_encoder = self.preprocess_data(data, target_column, split_date)
        model = self.train_model(X_train, y_train)
        y_pred = self.evaluate_model(model, X_test, y_test, label_encoder)
        return self.log_metrics(y_test, y_pred)

    @abstractmethod
    def preprocess_data(self, data, target_column, split_date):
        pass

    @abstractmethod
    def train_model(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate_model(self, model, X_test, y_test, label_encoder):
        pass

    @abstractmethod
    def log_metrics(self, y_test, y_pred):
        pass


class BaseVisualizer(ABC):
    @abstractmethod
    def plot_true_vs_predicted(self, y_true, y_pred, title):
        pass

    @abstractmethod
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        pass
