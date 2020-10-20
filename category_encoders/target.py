import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding для категориальных признаков. 
    Значение признака заменяется на некоторую статистику от таргета от элементов из обучающей выборки с таким же значением категориального признака.
    
    Параметры
    ----------
    
    col: str
        название преобразуемой колонки
    method: callable
        функция статистики (к примеру, среднее, медиана, среднеквадратичное отклонение). При None берется среднее арифметическое
    unknown: float
        если значение категориального признака впервые встретилось в преобразовываемой колонкe, помещается unknown. При None выбрасывается исключение
    """
    
    def __init__(self, col, method=None, unknown=None):
        self.col = col
        self.method = method
        if self.method == None:
            self.method = lambda x: np.mean(x)
        self.unknown = unknown
            
    
        
    def fit(self, X, y): 
        col = X[self.col].reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        self.d = dict(zip(np.unique(col), np.zeros(len(np.unique(col)))))
        for i in np.unique(col):
            self.d[i] = self.method(y[col == i].values)
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
