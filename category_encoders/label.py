import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing

class LabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoding для категориальных данных, каждой категории присваивается значение от 0 до N-1
    (здесь N - количество категорий для признака)
    
    Параметры
    ----------
    
    cols: list
        список названий колонок, которые необходимо преобразовать
    drop_invariant: bool
        логическое значение, отбрасывать ли столбцы с нулевой дисперсией
        
    """
    def __init__(self, cols=None, drop_invariant=False):
        self.cols = cols
        self.drop_invariant = drop_invariant
        
    def fit(self, X, y=None):
        if self.drop_invariant:
            self.drop = []
            for i in self.cols:
                if len(X[i].unique()) == 1:
                    self.drop.append(i)
            for i in self.drop:
                self.cols.remove(i)
        self.le = []
        for i in self.cols:
            self.le.append(preprocessing.LabelEncoder().fit(X[i]))
        return self   
        
    def transform(self, X):
        if self.drop_invariant:
            X = X.drop(self.drop, axis=1)
        
        for i, j in zip(self.cols, self.le):
            X.loc[:, i+'_le'] = j.transform(X[i])
        X = X.drop(self.cols, axis=1)
        
        return X
    
    def get_feature_names(self):
        return [(i+'_le') for i in self.cols]
