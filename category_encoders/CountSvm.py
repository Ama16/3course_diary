import pandas as pd
from numpy.linalg import svd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CountSvmEncoder(BaseEstimator, TransformerMixin):
    """CountSvm encoding для категориальных признаков. 
    Кодируемый категориальный признак кодируется k столбцами матрицы U в сингулярном разложении матрицы подсчета P (P_i_j равно 
    количеству элементов в обучающей выборке, у который кодируемый признак равен d1[i], а кодирующий - d2[j], где d1 и d2 - некоторые
    биективные отображения множеств чисел [0; кол-во уникальных значений кодируемого (кодирующего) признака] в значения признака соответственно)
    
    Параметры
    ----------
    
    cols: tuple (str, str)
        названия колонок. Первое название кодируемого признака, второе - кодирующего
    k: int
        количество столбцов матрицы U в сингулярном разложении. Если None, то берется вся матрица U.
        
    """
    
    def __init__(self, col, k=None):
        self.name1 = col[0]
        self.name2 = col[1]
        self.k = k
        
    def fit(self, X):
        col1 = X[self.name1].reset_index(drop=True)
        col2 = X[self.name2].reset_index(drop=True)
        if self.k == None:
            self.k = len(np.unique(col2))
        self.k = min(self.k, len(np.unique(col2)))
        self.d1 = dict(zip(np.unique(col1), np.arange(len(np.unique(col1)))))
        self.d2 = dict(zip(np.unique(col2), np.arange(len(np.unique(col2)))))
        p = np.zeros((len(np.unique(col1)), len(np.unique(col2))))
        for i in range(len(col1)):
            p[self.d1[col1[i]]][self.d2[col2[i]]] += 1
        u, _, _ = svd(p)
        self.u = u[:, :self.k]
        return self
    
    def transform(self, X):
        col = X[self.name1].reset_index(drop=True)
        p = np.empty((len(col), self.k))
        for i in range(len(col)):
            assert col[i] in self.d1.keys(), "Unexpected value"
            now = self.d1[col[i]]
            p[i] = self.u[now]
        return pd.DataFrame(p, columns=[self.name1 +' _' + str(i) for i in range(p.shape[1])])
