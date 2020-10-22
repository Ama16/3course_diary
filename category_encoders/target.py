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
    min_samples: int
        минимальное количество объектов с одинаковым значением признака для учета их категории
    smoothing: float
        smoothing effect для балланса между средним по категории и общим средним. бОльшие значения ведут к более сильной регуляризации.
        Значение должно быть больше 0. При значениях <= 0 или же при отсутствии значения выполняется обычный target encoder.
    """
    
    def __init__(self, col, method=None, unknown=None, min_samples = 0, smoothing = 0):
        self.col = col
        self.method = method
        if self.method == None:
            self.method = lambda x: np.mean(x)
        self.unknown = unknown
        self.smoothing = float(smoothing)
        self.min_samples = min_samples
          
        
    def fit(self, X, y): 
        """
        Параметры
        ----------
        X : Series
            Колонка с обучающими значениями
        y : Series
            Колонка таргета
        """
        col = X[self.col].reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        prior = y.mean()
        name = self.method.__name__
        stats = y.groupby(col).agg(['count', self.method])
        
        if self.smoothing > 0:
            smoove = 1 / (1 + np.exp(-(stats['count'] - self.min_samples) / self.smoothing))
            smoothing = prior * (1 - smoove) + stats[name] * smoove
            smoothing[stats['count'] < self.min_samples] = prior #если меньше min_samples, присваиваем общее среднее
        
        if self.smoothing > 0:
            self.d = smoothing
            return self
        
        self.d = dict(zip(np.unique(col), np.zeros(len(np.unique(col)))))
        for i in np.unique(col):
            if stats['count'][i] < self.min_samples:
                self.d[i] = prior
            else:
                self.d[i] = stats[name][i]
        return self
        return self
    
    def transform(self, X):
        """
        Параметры
        ----------
        X : Series
            Колонка с преобразуемыми значениями
        """
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
