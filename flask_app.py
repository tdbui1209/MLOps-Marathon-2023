from flask import Flask, jsonify, request
import pickle
import numpy as np
from utils import get_input


app = Flask(__name__)

@app.route('/phase-1/prob-1/predict', methods=['POST'])
def predict():
    id, df = get_input(request)

    df = df[category_columns1 + numeric_columns1]
    OH_X = encoder1.transform(df[category_columns1])
    scaled_X = scaler1.transform(df[numeric_columns1])
    poly_X = poly1.transform(scaled_X)

    X = np.concatenate((OH_X, poly_X), axis=1)

    pred = model1.predict(X)
    output = {
        'id': id,
        'predictions': [int(i) for i in pred],
        'drift': 0
    }
    return jsonify(output)

@app.route('/phase-1/prob-2/predict', methods=['POST'])
def predict2():
    id, df = get_input(request)

    df = df[category_columns2 + numeric_columns2]
    OH_X = encoder2.transform(df[category_columns2])

    scaled_X = scaler2.transform(df[numeric_columns2])
    poly_X = poly2.transform(scaled_X)

    X = np.concatenate((OH_X, poly_X), axis=1)

    pred = model2.predict(X)
    output = {
        'id': id,
        'predictions': [int(i) for i in pred],
        'drift': 0
    }
    return jsonify(output)


if __name__ == '__main__':
    host = '192.168.1.11'
    port = 5000

    global model1, encoder1, scaler1, poly1
    global model2, encoder2, scaler2, poly2

    model_paths = 'models/phase1/prob1/'
    model_file = model_paths + 'model.sav'
    with open(model_file, 'rb') as f:
        model1 = pickle.load(f)
    encoder_file = model_paths + 'encoder.sav'
    with open(encoder_file, 'rb') as f:
        encoder1 = pickle.load(f)
    scaler_file = model_paths + 'scaler.sav'
    with open(scaler_file, 'rb') as f:
        scaler1 = pickle.load(f)
    poly_file = model_paths + 'poly.sav'
    with open(poly_file, 'rb') as f:
        poly1 = pickle.load(f)


    model_paths = 'models/phase1/prob2/'
    model_file = model_paths + 'model.sav'
    with open(model_file, 'rb') as f:
        model2 = pickle.load(f)
    encoder_file = model_paths + 'encoder.sav'
    with open(encoder_file, 'rb') as f:
        encoder2 = pickle.load(f)
    scaler_file = model_paths + 'scaler.sav'
    with open(scaler_file, 'rb') as f:
        scaler2 = pickle.load(f)
    poly_file = model_paths + 'poly.sav'
    with open(poly_file, 'rb') as f:
        poly2 = pickle.load(f)

    global category_columns1, numeric_columns1
    category_columns1 = ['feature2', 'feature1']
    numeric_columns1 = ['feature3', 'feature4', 'feature5', 'feature6', 'feature7',
                       'feature8', 'feature9', 'feature10', 'feature11', 'feature12',
                       'feature13', 'feature14', 'feature15', 'feature16']
    
    global category_columns2, numeric_columns2
    category_columns2 = ['feature1', 'feature3', 'feature4', 'feature6', 'feature7', 'feature8',
                        'feature9', 'feature10', 'feature11', 'feature12', 'feature14', 'feature15',
                        'feature16', 'feature17', 'feature19', 'feature20']
    numeric_columns2 = ['feature2', 'feature5', 'feature13', 'feature18']
    
    app.run(host=host, port=port, debug=True)
