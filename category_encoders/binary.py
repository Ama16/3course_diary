import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce

class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Binary encoding для категориальных признаков. Похоже на One-Hot, но хванит категории как двоичные битовые строки.
    
    Параметры
    ----------
    
    cols: list
        список названий колонок, которые необходимо преобразовать
    drop_invariant: bool
        логическое значение, отбрасывать ли столбцы с нулевой дисперсией
    return_df: bool
        логическое значение, указывающее, следует ли возвращать pandas DataFrame 
        из transform (в противном случае это будет массив numpy)
    handle_unknown: str
        возможные значения: 'error', 'return_nan', 'value', 'indicator'. Если используется индикатор, для
        трансформируемой матрицы будет добавлена дополнительная колонка.
    handle_missing: str
        возможные значения: 'error', 'return_nan', 'value', 'indicator'. Если используется индикатор, для
        трансформируемой матрицы будет добавлена дополнительная колонка.
        
    """
    def __init__(self, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value'):
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.base_n_encoder = ce.BaseNEncoder(base=2, cols=self.cols, drop_invariant=self.drop_invariant, 
                                              return_df=self.return_df, handle_unknown=self.handle_unknown, 
                                              handle_missing=self.handle_missing)
        
        
    def fit(self, X, y=None):
        self.base_n_encoder.fit(X, y)
        return self   
        
    def transform(self, X, override_return_df=False):
        return self.base_n_encoder.transform(X)
    
    def get_feature_names(self):
        return self.base_n_encoder.get_feature_names()
