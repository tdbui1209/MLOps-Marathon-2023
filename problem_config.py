import logging
import json
from utils import AppPath
import pickle


class ProblemConfig:

    def __init__(self, phase_id: str, prob_id: str):
        self.config = self.get_config(phase_id, prob_id)
        self.phase_id = phase_id
        self.prob_id = prob_id
        self.encoder = self.get_encoder()
        self.scaler = self.get_scaler()
        self.poly = self.get_poly()
        self.predict_model = self.get_predict_model()
        logging.info(f'ProblemConfig {phase_id} - {prob_id} is initialized')

    def get_config(self, phase_id: str, prob_id: str):
        with open(AppPath.MODEL_CONFIG_DIR / phase_id / f'{prob_id}.json') as f:
            config = json.load(f)
        return config

    def get_numerical_cols(self):
        return self.config['numerical_columns']
    
    def get_categorical_cols(self):
        return self.config['categorical_columns']
    
    def get_encoder(self):
        return self.load_model('encoder')
    
    def get_scaler(self):
        return self.load_model('scaler')
    
    def get_poly(self):
        return self.load_model('poly')
    
    def get_predict_model(self):
        return self.load_model('predict_model')

    def load_model(self, config_key):
        try:
            with open(AppPath.MODEL_DIR / self.phase_id / self.prob_id / self.config[config_key], 'rb') as f:
                return pickle.load(f)
        except KeyError:
            logging.info(f'{self.phase_id} {self.prob_id} does not have {config_key}')
            return None