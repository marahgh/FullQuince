from quince.library.datasets import Synthetic
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import QuantileRegressor



if __name__ == "__main__":
    gamma_star = 2.7
    synthetic_dataset = Synthetic(num_examples=1000, gamma_star=gamma_star, mode="mu")
    x = synthetic_dataset.x.reshape(1, -1)
    t = synthetic_dataset.t.reshape(1, -1)
    features = np.vstack((x, t)).T
    y = synthetic_dataset.y
    qr = QuantileRegressor(quantile=0.5, alpha=0)
    y_pred = qr.fit(features, y).predict(features)
    print("")