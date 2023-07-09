from fastapi import FastAPI, Request
from pydantic import BaseModel

import uvicorn
import pickle
import numpy as np
import pandas as pd


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class PredictorAPI:

    def __init__(self):
        self.app = FastAPI()

        @self.app.post('/phase-2/prob-1/predict')
        async def predict(data: Data):
            df = pd.DataFrame(data.rows, columns=data.columns)

            X_num = scaler1.transform(df[numerical_features1])
            X_cat = encod1.transform(df[categorical_features1])

            X = np.concatenate([X_num, X_cat], axis=1)
            y_pred = model1.predict(X)
            return {
                'id': data.id,
                'predictions': y_pred.tolist(),
                'drift': 0
            }

        @self.app.post('/phase-2/prob-2/predict')
        async def predict(data: Data):
            df = pd.DataFrame(data.rows, columns=data.columns)
            X_num = scaler2.transform(df[numerical_features2])
            X_cat = encod2.transform(df[categorical_features2])

            X = np.concatenate([X_num, X_cat], axis=1)
            y_pred = model2.predict(X)
            return {
                'id': data.id,
                'predictions': label_encod2.inverse_transform(y_pred).tolist(),
                'drift': 0
            }
        
    def run(self, host, port):
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    models_path = 'models/'
    global model1, model2
    model1 = pickle.load(open(models_path + 'model1.sav', 'rb'))
    model2 = pickle.load(open(models_path + 'model2.sav', 'rb'))

    global encod1, encod2
    encod1 = pickle.load(open(models_path + 'encoder1.sav', 'rb'))
    encod2 = pickle.load(open(models_path + 'encoder2.sav', 'rb'))

    global scaler1, scaler2
    scaler1 = pickle.load(open(models_path + 'scaler1.sav', 'rb'))
    scaler2 = pickle.load(open(models_path + 'scaler2.sav', 'rb'))

    global label_encod2
    label_encod2 = pickle.load(open(models_path + 'label_encoder2.sav', 'rb'))

    global categorical_features1, categorical_features2
    global numerical_features1, numerical_features2

    numerical_features1 = ["feature1", "feature5", "feature6", "feature7", "feature8", "feature9",
                          "feature10", "feature11", "feature12", "feature13", "feature14", "feature15",
                          "feature16", "feature17", "feature18", "feature19", "feature20", "feature21",
                          "feature22", "feature23", "feature24", "feature25", "feature26", "feature27",
                          "feature28", "feature29", "feature30", "feature31", "feature32", "feature33",
                          "feature34", "feature35", "feature36", "feature37", "feature38", "feature39",
                          "feature40", "feature41"]
    categorical_features1 = ["feature2", "feature3", "feature4"]

    numerical_features2 = ["feature1", "feature5", "feature6", "feature7", "feature8", "feature9",
                          "feature10", "feature11", "feature12", "feature13", "feature14", "feature15",
                          "feature16", "feature17", "feature18", "feature19", "feature20", "feature21",
                          "feature22", "feature23", "feature24", "feature25", "feature26", "feature27",
                          "feature28", "feature29", "feature30", "feature31", "feature32", "feature33",
                          "feature34", "feature35", "feature36", "feature37", "feature38", "feature39",
                          "feature40", "feature41"]
    categorical_features2 = ["feature2", "feature3", "feature4"]

    api = PredictorAPI()
    api.run(host="192.168.1.11", port=5000)