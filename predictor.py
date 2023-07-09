import logging
import os
import pandas as pd
from pandas.util import hash_pandas_object
import time


class Predictor:

    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        start_time = time.time()
        y_pred = self.model.predict(X)
        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f'prediction takes {run_time} ms')
        return y_pred
    
    def detect_drift(self, X):
        return 0

    @staticmethod
    def save_request_data(df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id:
            file_name = data_id.strip()
        else:
            file_name = hash_pandas_object(df).sum()
        output_file_path = os.path.join(captured_data_dir, f'{file_name}.parquet')
        df.to_parquet(output_file_path, index=False)
        return output_file_path
    