from sklearn import datasets as dts
from Multivariate import multivariate


X, y = dts.make_regression(n_samples=500,
                           n_features=5,
                           n_targets=5)


if __name__ == "__main__":
    multivariate(model=MLPRegressor,
                 X=X,
                 y=y,
                 cv=10,
                 random_state=1,
                 title="NN")

    multivariate(model=BaggingRegressor,
                 X=X,
                 y=y,
                 cv=10,
                 random_state=1,
                 title="Bagging")
