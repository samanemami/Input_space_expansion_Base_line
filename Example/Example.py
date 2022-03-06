from sklearn import datasets as dts
from Multivariate import multivariate


X, y = dts.make_regression(n_samples=500,
                           n_features=5,
                           n_targets=5)


if __name__ == "__main__":
    multivariate(X=X,
                 y=y,
                 cv=5,
                 random_state=123,
                 title="regression")
