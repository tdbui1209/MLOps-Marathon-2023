import argparse

from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np
import pandas as pd
from utils import AppPath, Data
from predictor import Predictor
from problem_config import ProblemConfig


class PredictorAPI:

    def __init__(self):
        self.app = FastAPI()

        @self.app.post('/phase-2/prob-1/predict')
        async def predict(data: Data):
            df = pd.DataFrame(data.rows, columns=data.columns)
            if args.capture_data:
                Predictor.save_request_data(df, AppPath.CAPTURED_DATA_DIR, str(data.id))

            X_num = phase2_prob1_config.scaler.transform(df[phase2_prob1_config.get_numerical_cols()])
            X_cat = phase2_prob1_config.encoder.transform(df[phase2_prob1_config.get_categorical_cols()])
            X = np.concatenate([X_num, X_cat], axis=1)

            y_pred = Predictor(phase2_prob1_config.model).predict(X)
            return {
                'id': data.id,
                'predictions': y_pred.tolist(),
                'drift': 0
            }

        @self.app.post('/phase-2/prob-2/predict')
        async def predict(data: Data):
            df = pd.DataFrame(data.rows, columns=data.columns)
            if args.capture_data:
                Predictor.save_request_data(df, AppPath.CAPTURED_DATA_DIR, str(data.id))

            X_num = phase2_prob2_config.scaler.transform(df[phase2_prob2_config.get_numerical_cols()])
            X_cat = phase2_prob2_config.encoder.transform(df[phase2_prob2_config.get_categorical_cols()])
            X = np.concatenate([X_num, X_cat], axis=1)

            y_pred = Predictor(phase2_prob2_config.predict_model).predict(X)
            return {
                'id': data.id,
                'predictions': label_encod2.inverse_transform(y_pred).tolist(),
                'drift': 0
            }
        
    def run(self, host, port):
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=int)
    parser.add_argument('--capture-data', action='store_true')
    args = parser.parse_args()

    phase2_prob1_config = ProblemConfig('phase2', 'prob1')
    phase2_prob2_config = ProblemConfig('phase2', 'prob2')

    label_encod2 = pickle.load(open('models/phase2/prob2/' + 'label_encoder2.sav', 'rb'))

    api = PredictorAPI()
    api.run(host=args.host, port=args.port)