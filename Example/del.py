# %%
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np

X, y = make_regression(n_targets=1)

model = RandomForestRegressor()
model.fit(X, y)

rrmse = np.sqrt(((y - model.predict(X))**2) /
                ((y - np.mean(y))**2))
np.mean(rrmse)

mean_squared_error(y, model.predict(X), squared=False)
