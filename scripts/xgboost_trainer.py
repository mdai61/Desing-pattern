import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from .base_trainer import BaseTrainer
from .decorators import log_results

class XGBoostTrainer(BaseTrainer):
    def preprocess_data(self, data, target_column, split_date):
        """Split and preprocess data."""
        label_encoder = LabelEncoder()
        data[target_column] = label_encoder.fit_transform(data[target_column])

        train_data = data[data.index < split_date]
        test_data = data[data.index >= split_date]

        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

        return X_train, X_test, y_train, y_test, label_encoder

    def train_model(self, X_train, y_train):
        """Train an XGBoost model."""
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }
        model = xgb.train(params, dtrain, num_boost_round=100)
        return model

    @log_results
    def evaluate_model(self, model, X_test, y_test, label_encoder):
        """Evaluate the trained model."""
        scaler = MinMaxScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        dtest = xgb.DMatrix(X_test_scaled)
        y_pred = model.predict(dtest)

        self.visualizer.plot_true_vs_predicted(y_test, y_pred, title="True vs Predicted Values (XGBoost)")
        self.visualizer.plot_confusion_matrix(y_test, y_pred, classes=label_encoder.classes_)
        return y_pred

    def log_metrics(self, y_test, y_pred):
        """Log RMSE and R² metrics."""
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {"rmse": rmse, "r2": r2}
