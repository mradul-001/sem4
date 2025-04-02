from deep_river.regression import RollingRegressorInitialized
from river import metrics
from torch import nn
import pandas as pd
import numpy as np


class LstmModule(nn.Module):

    def __init__(self, n_features, hidden_size=1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=False,
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)
        return self.fc(output[-1, :])


class StockPredictor:
    
    def __init__(self, n_features):
        self.model = RollingRegressorInitialized(
            module=LstmModule(n_features, 16),
            loss_fn="mse",
            optimizer_fn="adam",
            window_size=20,
            lr=0.001,
            append_predict=True,
        )
        self.metric = metrics.MAE()

    def prepareData(self, dataframe: pd.DataFrame, targetVar: str, lag: int):
        df = dataframe.copy()
        for i in range(1, lag + 1):
            df[f"{targetVar}_{i}"] = df[targetVar].shift(i)
        df.dropna(inplace=True)

        x = df.drop(columns=[targetVar])
        y = df[targetVar]
        return x.to_dict(orient="records"), y.to_list()

    def trainModel(self, x, y, epochs=10):
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(y)):
                y_pred = self.model.predict_one(x[i])
                loss = (y_pred - y[i]) ** 2
                epoch_loss += loss
                self.metric.update(y_true=y[i], y_pred=y_pred)
                self.model.learn_one(x[i], y[i])
            
    
            avg_loss = epoch_loss / len(y)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MAE: {self.metric.get():.4f}")

    def evaluate(self, X, y):
        y_pred = [self.model.predict_one(xi) for xi in X]
        mse = np.mean((np.array(y) - np.array(y_pred)) ** 2)
        return mse