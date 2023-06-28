from flask import Flask, request
import pickle
import numpy as np
from data_processor import DataProcessor
from problem_config import ProblemConfig
from predictor import Predictor


class PredictorApi:

    def __init__(self):
        self.app = Flask(__name__)
        self.config_prob1 = ProblemConfig('phase1', 'prob1')
        self.config_prob2 = ProblemConfig('phase1', 'prob2')

        @self.app.route('/')
        def root():
            return {'message': 'Hello World!'}

        @self.app.route('/phase-1/prob-1/predict', methods=['POST'])
        def predict_prob1():
            return self.predict(request, self.config_prob1)

        @self.app.route('/phase-1/prob-2/predict', methods=['POST'])
        def predict_prob2():
            return self.predict(request, self.config_prob2)
        
    def predict(self, request, config):
        id, df = DataProcessor.get_input(request)
        category_columns = config.get_categorical_cols()
        numeric_columns = config.get_numerical_cols()

        encoder = config.encoder
        scaler = config.scaler
        poly = config.poly
        model = config.model

        X = DataProcessor.process(df, numeric_columns, category_columns, encoder, scaler, poly)
        predictor = Predictor(model)
        y_pred = predictor.predict(X)
        is_drift = predictor.detect_drift(df)

        output = {
            'id': id,
            'predictions': y_pred,
            'drift': is_drift
        }
        return output
        
    def run(self, host, port, debug=False):
        self.app.run(host, port, debug)


if __name__ == '__main__':
    host = '192.168.1.11'
    port = 5000
    
    api = PredictorApi()
    api.run(host, port, debug=True)
