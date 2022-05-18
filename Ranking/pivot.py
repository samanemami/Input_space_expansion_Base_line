from topsis import topsis
import os
import fnmatch
import warnings
import pandas as pd
import numpy as np

warnings.simplefilter('ignore')


def df(n, score):
    index_ = {'name': [], 'shape': []}
    for file in os.listdir('.'):
        if fnmatch.fnmatch(file, '*' + score):
            index_['shape'].append((pd.read_csv(file).shape[1]))
            index_['name'].append(file)

    max_cl = max(index_['shape'])
    file = index_['name'][index_['shape'].index(max_cl)]
    dic = {'key': pd.read_csv(file).columns}
    dic = list(pd.read_csv(file).columns)
    row_names = [index_['name'][i][2:-n] for i in range(len(index_['name']))]
    df = pd.DataFrame(
        np.zeros((len(index_['name']), max_cl)), columns=dic, index=row_names)

    for _, file in enumerate(index_['name']):
        ave = (pd.read_csv(file)).mean()
        for i, j in enumerate(ave.index):
            df.iloc[_, dic.index(j)] = ave.iloc[i]

    return df


rmse = df(n=9, score='_RMSE.csv').T
r2 = df(n=13, score='_R2_score.csv').T



# Building the decision matrix
rmse_ = rmse.iloc[32:, :]
decision_matrix = pd.DataFrame(
    np.zeros((rmse_.shape[0], 1)), index=rmse_.index)
for j in range(rmse_.shape[0]):
    decision_matrix.iloc[j, 0] = (rmse_.iloc[j, 0])

decision_matrix = decision_matrix.loc[(decision_matrix != 0).any(1)]
decision_matrix = decision_matrix.rename(columns={0: 'RMSE'})


topsis_ = topsis(decision_matrix=decision_matrix,
                 weight=[1], impact=['-'])
topsis_.rank()
