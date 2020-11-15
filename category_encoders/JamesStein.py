from sklearn.utils import check_random_state
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformerWithTargetMixin:
    """
    Необходим для добавления шума в train данные
    """
    def fit_transform(self, X, y=None):
        if y is None:
            raise TypeError('fit_transform() missing argument: ''y''')
        return self.fit(X, y).transform(X, y)

class JamesSteinEncoder(BaseEstimator, MyTransformerWithTargetMixin):
    """James Stein Encoding для категориальных данных.
    
    Параметры
    ----------
    
    cols: list
        названия колонок, которые необходимо преобразовать
    random_state: int
    randomized: boolean
        при True добавляет нормальный шум с средним значением 1 в train данные (умножает на него). Test данные не трогаются
    sigma: float
        среднее квадратическое отклонение нормального распределения 
    """
    def __init__(self, cols=None, random_state=None, randomized=False, sigma=0.05):
        self.cols = cols
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.mapping = {}
        
    def fit(self, X, y): 
    """
    Обязательно len(X) == len(y)
    
    Параметры
    ----------
    
    X: pd.DataFrame
    y: pd.Series
    """
        col = X[self.cols].reset_index(drop=True)
        y = y.reset_index(drop=True).astype('float')
        prior = y.mean()
        global_count = len(y)
        global_var = y.var()
        
        for name_col in self.cols:
            stats = y.groupby(col[name_col]).agg(['mean', 'var'])

            i_var = stats['var'].fillna(0) 

            smoothing = i_var / (global_var + i_var) 
            self.mapping[name_col] = (1 - smoothing)*(stats['mean']) + smoothing*prior
        
        return self
    
    
    def transform(self, X, y=None):
    """
    y подается только при fit_transform, при transform подавать y нельзя!
    """
        X_now = X.copy()
        for col in self.cols:
            X_now[col] = X_now[col].map(self.mapping[col])

            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X_now[col] = (X_now[col] * random_state_generator.normal(1., self.sigma, X_now[col].shape[0]))

        return X_now
