# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

from sklearn import datasets as dts

X, y = dts.make_regression(n_samples=500,
                           n_features=5,
                           n_targets=5)


title = 'atp7d'
cv_out = 2

random_state = 123
np.random.seed(random_state)

df = pd.DataFrame(y)
(df.corr(method='pearson', min_periods=1)).to_csv(title + "_Correlations.csv")


def input(X, y, i):
    X = np.append(X, y[:, i][:, np.newaxis], axis=1)
    return X


kfold = KFold(n_splits=cv_out, shuffle=True, random_state=random_state)
n_targest = y.shape[1] * 2
scores = np.zeros((cv_out, n_targest))
scores = pd.DataFrame(scores, columns=pd.RangeIndex(0, scores.shape[1], 1))

for _, (train_index, test_index) in enumerate(kfold.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train m models for m outputs
    # for i in range(0, y_train.shape[1], 1):

    # model = RandomForestRegressor()
    # model.fit(x_train, y_train[:, i])
    # score = model.score(x_test, y_test[:, i])
    # scores.iloc[_, i] = score
    # mapping = {scores.columns[i]: 'target_'+str(i)}
    # scores = scores.rename(columns=mapping)

    # 1st experiment: we can see how much
    #   informations there are in outputs to predict other outputs

    # j = 0

    # while j < y_train.shape[1]:

    #   if j+1 < y_train.shape[1]:
    #     X_train =

# %%


# %%
i = 0
while i < y_train.shape[1]:
    if i+1 < y_train.shape[1]:
        X = np.append(y_train[:, i][:, np.newaxis],
                      y_train[:, i+1][:, np.newaxis], axis=1)
        i += 1
    else:
        break


# %%
X.shape
