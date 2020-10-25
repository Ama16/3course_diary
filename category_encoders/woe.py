import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class WoEEncoder(BaseEstimator, TransformerMixin):
     """Weight of Evidence Encoding для категориальных данных.
         Только для бинарного таргета.
    
    Параметры
    ----------
    
    col: list
        название колонки, которую необходимо преобразовать
    unknown: float
        если значение категориального признака впервые встретилось в преобразовываемой колонкe, помещается unknown. При None выбрасывается исключение
        
    """
    def __init__(self, col=None, unknown=None):
        self.col = col
        self.unknown = unknown
        
    def fit(self, X, y): 
        col = X[self.col].reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        
        data = pd.DataFrame(pd.concat([col, y], axis=1))
        name = data.columns[1]
        tmp = pd.DataFrame(data.groupby(col)[name].count())[name]
        data = pd.DataFrame(data.groupby(col)[name].sum())
        data['not_target'] = tmp - data[name]
        data['answer'] = np.log((data[name] + 0.5) / (data['not_target'] + 0.5))
        self.d = dict(data['answer'])
        
        return self
    
    def transform(self, X):
        col = X[self.col]
        answer = np.empty(len(col))
        for k, i in enumerate(X.index):
            if col[i] in self.d.keys():
                answer[k] = self.d[col[i]]
            else:
                if self.unknown == None:
                    raise Exception("Unexpected value")
                else:
                    answer[k] = self.unknown
        return pd.Series(answer, index=col.index)
