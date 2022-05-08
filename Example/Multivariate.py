from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")


def multivariate(model, X, y, cv, random_state, title):

    # Returns a data frame includes the
    # score regarding the different approaches

    np.random.seed(random_state)

    df = pd.DataFrame(y)
    corr = df.corr(method='pearson', min_periods=1)
    corr_linkage = hierarchy.ward(corr)
    hierarchy.dendrogram(corr_linkage, leaf_rotation=90)
    plt.title("dendrogram plot")

    plt.savefig(title + " dendrogram_y.jpg", dpi=500)
    plt.close('all')

    # (corr).to_csv(title + "_Correlations.csv")
    plot = plt.imshow(corr, cmap='coolwarm')
    plt.xlabel('Targets')
    plt.ylabel('targets')
    plt.title(title + " Correlation")
    plt.colorbar(plot, shrink=0.7)
    plt.savefig(title + "corr.jpg", dpi=500)
    plt.close('all')

    def augment(X, y):
        X = np.append(X, y, axis=1)
        return X

    def df(cv, y):
        n = y.shape[1] * 3
        df = np.zeros((cv, n))
        return pd.DataFrame(df, columns=pd.RangeIndex(0, df.shape[1], 1))

    def to_csv(df, title, score):
        df.to_csv(title + score, index=False)

    r2_score = df(cv, y)
    rmse = df(cv, y)
    rrmse = df(cv, y)

    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    for _, (train_index, test_index) in enumerate(kfold.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 1st Experiment:
        # Train m models for m outputs
        for i in range(0, y_train.shape[1], 1):

            model = model
            model.fit(x_train, y_train[:, i])

            # Calculate the R2_score of each target
            score = model.score(x_test, y_test[:, i])
            r2_score.iloc[_, i] = score
            mapping = {r2_score.columns[i]: 'target_'+str(i)}
            r2_score = r2_score.rename(columns=mapping)

            # Calculate the RMSE of each target
            rmse.iloc[_, i] = mse(
                y_test[:, i], model.predict(x_test), squared=False)
            mapping = {rmse.columns[i]: 'target_'+str(i)}
            rmse = rmse.rename(columns=mapping)

            # Claculate the RRMSE for each target
            rrmse.iloc[_, i] = np.sqrt(np.abs(1-score))
            mapping = {rrmse.columns[i]: 'target_' + str(i)}
            rrmse = rrmse.rename(columns=mapping)

        # 2nd Experiment:
        # Check the information in outputs to predict another outputs.

        i += 1
        j = 0
        while j < y_train.shape[1]:
            if j+1 <= y_train.shape[1]:
                X_train = np.delete(y_train, j, axis=1)
                Y_train = y_train[:, j]

                X_test = np.delete(y_test, j, axis=1)
                Y_test = y_test[:, j]

                model.fit(X_train, Y_train)
                score = model.score(X_test, Y_test)

                # Calculate the R2_score of each target
                r2_score.iloc[_, i] = score
                mapping = {r2_score.columns[i]: 'Y|target'+str(j)}
                r2_score = r2_score.rename(columns=mapping)

                # Calculate the RMSE of each target
                rmse.iloc[_, i] = mse(
                    Y_test, model.predict(X_test), squared=False)
                mapping = {rmse.columns[i]: 'Y|target'+str(i)}
                rmse = rmse.rename(columns=mapping)

                i += 1
                j += 1
            else:
                break

        # 3rd Experiment:
        # Inspect how much informations there are in outputs
        # to predict other outputs over the baseline.

        j = 0
        while j < y_train.shape[1]:
            if j+1 <= y_train.shape[1]:
                X_train = np.delete(y_train, j, axis=1)
                X_train = augment(x_train, X_train)
                Y_train = y_train[:, j]

                X_test = np.delete(y_test, j, axis=1)
                X_test = augment(x_test, X_test)
                Y_test = y_test[:, j]

                model.fit(X_train, Y_train)
                score = model.score(X_test, Y_test)

                # Calculate the R2_score of each target
                r2_score.iloc[_, i] = score
                mapping = {r2_score.columns[i]: 'D\'_target'+str(j)}
                r2_score = r2_score.rename(columns=mapping)

                # Calculate the RMSE of each target
                rmse.iloc[_, i] = mse(
                    Y_test, model.predict(X_test), squared=False)
                mapping = {rmse.columns[i]: 'D\'_target'+str(j)}
                rmse = rmse.rename(columns=mapping)

                i += 1
                j += 1
            else:
                break
    to_csv(df=r2_score, title=title, score='_R2_score.csv')
    to_csv(df=rmse, title=title, score='_RMSE.csv')
    to_csv(df=rrmse, title=title, score='_RRMSE.csv')
