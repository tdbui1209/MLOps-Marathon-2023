import json
from utils import AppPath
import pickle


class ProblemConfig:

    def __init__(self, phase_id: str, prob_id: str):
        self.config = self.get_config(phase_id, prob_id)
        with open(AppPath.MODEL_DIR / phase_id / prob_id / self.get_encoder_path(), 'rb') as f:
            self.encoder = pickle.load(f)
        with open(AppPath.MODEL_DIR / phase_id / prob_id / self.get_scaler_path(), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(AppPath.MODEL_DIR / phase_id / prob_id / self.get_poly_path(), 'rb') as f:
            self.poly = pickle.load(f)
        with open(AppPath.MODEL_DIR / phase_id / prob_id / self.get_model_path(), 'rb') as f:
            self.model = pickle.load(f)
        print('ProblemConfig is initialized')

    def get_config(self, phase_id: str, prob_id: str):
        with open(AppPath.MODEL_CONFIG_DIR / phase_id / f'{prob_id}.json') as f:
            config = json.load(f)
        return config

    def get_numerical_cols(self):
        return self.config['numerical_columns']
    
    def get_categorical_cols(self):
        return self.config['categorical_columns']
    
    def get_encoder_path(self):
        return self.config['encoder_path']
    
    def get_scaler_path(self):
        return self.config['scaler_path']
    
    def get_poly_path(self):
        return self.config['poly_path']
    
    def get_model_path(self):
        return self.config['model_path']
