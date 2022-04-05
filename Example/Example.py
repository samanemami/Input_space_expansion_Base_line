from sklearn import datasets as dts
from Multivariate import multivariate
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor


X, y = dts.make_regression(n_samples=500,
                           n_features=5,
                           n_targets=5)
random_state = 1
cv = 10

if __name__ == "__main__":
    multivariate(model=MLPRegressor(random_state=random_state),
                 X=X,
                 y=y,
                 cv=cv,
                 random_state=random_state,
                 title="NN")

    multivariate(model=BaggingRegressor(random_state=random_state),
                 X=X,
                 y=y,
                 cv=cv,
                 random_state=random_state,
                 title="Bagging")

    multivariate(model=RandomForestRegressor(random_state=random_state),
                 X=X,
                 y=y,
                 cv=cv,
                 random_state=random_state,
                 title="RF")
