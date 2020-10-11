import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoding для категориальных данных, меняет значение категории на частоту встречаемости данного 
    значения среди обучающей выборки.
    
    Параметры
    ----------
    
    cols: list
        список названий колонок, которые необходимо преобразовать
    drop_invariant: bool
        логическое значение, отбрасывать ли столбцы с нулевой дисперсией
    handle_unknown: str
        возможные значения: 'error', 'value'. При 'value' при неизвестном значении будет подставляться 0.
    handle_missing: str
        возможные значения: 'error', 'value'. При 'value' при null будет подставляться 0.
        
    """
    def __init__(self, cols=None, drop_invariant=False, handle_unknown='value', handle_missing='value'):
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        
    def fit(self, X, y=None):
        if self.drop_invariant:
            self.drop = []
            for i in self.cols:
                if len(X[i].unique()) == 1:
                    self.drop.append(i)
            for i in self.drop:
                self.cols.remove(i)
        self.maps = []
        for i in self.cols:
            self.maps.append(X.groupby(i).size() / len(X))
        return self   
        
    def transform(self, X):
        if self.drop_invariant:
            X = X.drop(self.drop, axis=1)
        
        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')
        
        for i, j in zip(self.cols, self.maps):
            X.loc[:, i+'_freq'] = X[i].map(j)
        X = X.drop(self.cols, axis=1)
        
        if self.handle_unknown == 'error':
            if X[[(i+'_freq') for i in self.cols]].isnull().any().any():
                raise ValueError('Columns contain unexpected value')
        
        X[[(i+'_freq') for i in self.cols]] = X[[(i+'_freq') for i in self.cols]].fillna(0)
        
        return X
    
    def get_feature_names(self):
        return [(i+'_freq') for i in self.cols]
