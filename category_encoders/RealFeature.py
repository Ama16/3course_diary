import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RealFeatureEncoder(BaseEstimator, TransformerMixin):
    """RealFeature encoding для категориальных признаков.
        
    Для данного категориального признака заменяет его значение X на результат работы 
    симметричной функции к значениям некоторого другого вещественного признака у элементов 
    из обучающей выборки, у которых значение исходного признака X.
        
        Параметры
        ----------
        col: tuple (str, str)
            названия колонок. Первое название кодируемого признака, второе - кодирующего
        function: callable
            симметричная функция от переменного числа аргументов 
        inplace: bool
            логическое значение, изменять ли данную в transform таблицу или нет
        
    """
    def __init__(self, col, function, inplace=False):
        self.name1 = col[0]
        self.name2 = col[1]
        self.fun = function
        self.inplace = inplace
        
    def fit(self, X):
        self.col1 = X[self.name1]
        self.col2 = X[self.name2]
        return self
    
    def transform(self, X):
        d = {}
        def trans(value):
            if value not in d.keys():
                d[value] = self.fun(*self.col2[self.col1 == value].values)
                return d[value]
            return d[value]
        if self.inplace:
            X[self.name1] = X[self.name1].apply(trans)
            return X[self.name1]
        return X[self.name1].apply(trans)
