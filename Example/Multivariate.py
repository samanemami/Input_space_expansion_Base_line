from sklearn.model_selection import KFold
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    n_targest = y.shape[1] * 3
    scores = np.zeros((cv, n_targest))
    scores = pd.DataFrame(scores, columns=pd.RangeIndex(0, scores.shape[1], 1))

    for _, (train_index, test_index) in enumerate(kfold.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 1st Experiment:
        # Train m models for m outputs
        for i in range(0, y_train.shape[1], 1):

            model = model
            model.fit(x_train, y_train[:, i])
            score = model.score(x_test, y_test[:, i])
            scores.iloc[_, i] = score
            mapping = {scores.columns[i]: 'target_'+str(i)}
            scores = scores.rename(columns=mapping)

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

                scores.iloc[_, i] = score

                mapping = {scores.columns[i]: 'Y|target'+str(j)}
                scores = scores.rename(columns=mapping)

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

                scores.iloc[_, i] = score

                mapping = {scores.columns[i]: 'D\'_target'+str(j)}
                scores = scores.rename(columns=mapping)

                i += 1
                j += 1
            else:
                break
    scores.to_csv(title + "_score.csv", index=False)
