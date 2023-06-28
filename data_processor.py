import json
import numpy as np
import pandas as pd
from problem_config import ProblemConfig


class DataProcessor:
        
    @staticmethod
    def get_input(request):
        data = json.loads(request.data.decode('utf-8'))
        id = data['id']
        rows = data['rows']
        columns = data['columns']
        df = pd.DataFrame(rows, columns=columns)
        return id, df
    
    @staticmethod
    def process(df, numerical_cols, categorical_cols, encoder, scaler, poly):
        df = df[numerical_cols + categorical_cols]

        OH_X = encoder.transform(df[categorical_cols])
        scaled_X = scaler.transform(df[numerical_cols])
        poly_X = poly.transform(scaled_X)

        X = np.concatenate((OH_X, poly_X), axis=1)
        return X
