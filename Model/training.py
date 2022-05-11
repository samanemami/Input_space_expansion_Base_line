# %%
import warnings
import numpy as np
import pandas as pd
from Dataset import dataset
from _multivariate import multivariate
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor


class model():
    """Train different ML models of MTR datasets

    Parameters
    ----------
    dt_names : list, Refers the list including mtr dataset names.

    dt_info : txt, Referes to the information file.

    model : Sklearn ML models

    cv : int, The number of cross validation folds.

    random_state : int, Determines random seed generation.

    title : string, Uses to save the outputs with the label of the title.

    """

    def __init__(self,
                 dt_names,
                 dt_info,
                 model,
                 cv,
                 random_state,
                 title):

        self.dt_names = dt_names
        self.info = dt_info
        self.model = model
        self.cv = cv
        self.random_state = random_state
        self.title = title

    def _input(self, i):
        names = pd.read_csv(self.info)
        index = names[names['Dataset'].str.contains(self.dt_names[i])]
        X, y = dataset(name=index.iloc[0, 0],
                       d=int(index.iloc[0, 1]))

        return X, y

    def fit(self):

        np.random.seed(self.random_state)
        warnings.simplefilter('ignore')

        for i, j in enumerate(self.dt_names):
            X, y = self._input(i)

            multivariate(model=self.model,
                         X=X,
                         y=y,
                         cv=self.cv,
                         random_state=self.random_state,
                         title=self.title + str(j[:-5]))


if __name__ == "__main__":

    info = 'Dataset\info.txt'
    dt_names = list(pd.read_csv(info).values[:, 0])

    cv = 10
    random_state = 123

    rf = model(dt_names=dt_names,
               dt_info=info,
               model=RandomForestRegressor(
                   random_state=random_state),
               cv=cv,
               random_state=random_state,
               title='RF_')
    rf.fit()

    nn = model(dt_names=dt_names,
               dt_info=info,
               model=MLPRegressor(
                   random_state=random_state),
               cv=cv,
               random_state=random_state,
               title='nn')
    nn.fit()

    bagging = model(dt_names=dt_names,
                    dt_info=info,
                    model=BaggingRegressor(
                        random_state=random_state),
                    cv=cv,
                    random_state=random_state,
                    title='bagging')
    bagging.fit()
