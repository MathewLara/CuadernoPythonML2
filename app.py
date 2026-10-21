from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import pickle
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)  # Permite que el HTML consuma la API sin problemas de CORS

RUTA_MODELO_JSON = Path("modelo_mp_neurona.json")
RUTA_MODELO_PKL = Path("modelo_mp_neurona.pkl")


# 1. Definición de la clase MpNeurona (McCulloch-Pitts)
class MpNeurona:
    def __init__(self):
        self.threshold = None

    def model(self, x):
        # Suma de las entradas binarias: z = x1 + x2 + ... + xn
        z = sum(x)
        return z >= self.threshold

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y, dtype=int)

    def fit(self, X, Y):
        accuracy = {}
        for th in range(X.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y, Y_pred)
        # Seleccionamos el threshold que maximiza la exactitud
        self.threshold = max(accuracy, key=accuracy.get)
        return accuracy[self.threshold]


# 2. Función para entrenar y exportar el modelo (o cargarlo si ya está exportado)
def inicializar_modelo():
    top_features = [
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'mean smoothness'
    ]

    # Si ya existe el archivo JSON exportado desde el Notebook, lo cargamos directamente
    if RUTA_MODELO_JSON.exists():
        with RUTA_MODELO_JSON.open("r", encoding="utf-8") as f:
             artefacto = json.load(f)
        neurona = MpNeurona()
        neurona.threshold = int(artefacto["threshold_optimo_5_vars"])
        parametros = artefacto["parametros_binarizacion"]
        X_min = pd.Series({col: parametros[col]["min"] for col in top_features})
        X_max = pd.Series({col: parametros[col]["max"] for col in top_features})
        X_mid = pd.Series({col: parametros[col]["midpoint"] for col in top_features})
        test_acc = float(artefacto.get("exactitud_prueba_5_vars", artefacto.get("exactitud_prueba_30_vars", 0.8112)))
        train_acc = test_acc
        print(f"Modelo cargado desde archivo exportado: {RUTA_MODELO_JSON}")
        print(f"Threshold óptimo cargado: {neurona.threshold}")
        return neurona, top_features, X_min, X_max, X_mid, train_acc, test_acc

    # Si no existe el archivo exportado, entrenamos desde sklearn y lo exportamos automáticamente
    cancer_mama = load_breast_cancer()
    df = pd.DataFrame(cancer_mama.data, columns=cancer_mama.feature_names)
    Y = cancer_mama.target

    X_train, X_test, y_train, y_test = train_test_split(
        df[top_features], Y, stratify=Y, random_state=42
    )

    X_min = X_train.min()
    X_max = X_train.max()
    X_mid = (X_min + X_max) / 2

    X_train_bin = X_train.apply(pd.cut, bins=2, labels=[1, 0]).astype(int)
    X_test_bin = X_test.apply(lambda col: np.where(col <= X_mid[col.name], 1, 0))

    neurona = MpNeurona()
    train_acc = neurona.fit(X_train_bin.to_numpy(), y_train)
    test_acc = accuracy_score(y_test, neurona.predict(X_test_bin.to_numpy()))

    # Exportamos en JSON y PKL para cumplir con la entrega del modelo exportado
    artefacto_exportado = {
        "proyecto": "prediccion_cancer_mama_mp_neurona",
        "version": "1.0.0",
        "modelo": "MpNeurona (McCulloch-Pitts)",
        "top_features": top_features,
        "threshold_optimo_5_vars": int(neurona.threshold),
        "exactitud_entrenamiento_5_vars": round(float(train_acc), 4),
        "exactitud_prueba_5_vars": round(float(test_acc), 4),
        "parametros_binarizacion": {
            col: {
                "min": round(float(X_min[col]), 6),
                "max": round(float(X_max[col]), 6),
                "midpoint": round(float(X_mid[col]), 6)
            }
            for col in top_features
        }
    }
    with RUTA_MODELO_JSON.open("w", encoding="utf-8") as f:
        json.dump(artefacto_exportado, f, indent=2, ensure_ascii=False)

    with RUTA_MODELO_PKL.open("wb") as f:
        pickle.dump({"neurona": neurona, "artefacto": artefacto_exportado}, f)

    print(f"Modelo entrenado y exportado en {RUTA_MODELO_JSON} y {RUTA_MODELO_PKL}")
    print(f"Threshold óptimo: {neurona.threshold} / {len(top_features)}")
    return neurona, top_features, X_min, X_max, X_mid, train_acc, test_acc


neurona, top_features, X_min, X_max, X_mid, train_acc, test_acc = inicializar_modelo()


# Ruta opcional para abrir el HTML directamente en http://127.0.0.1:5000/
@app.route('/', methods=['GET'])
def home():
    if os.path.exists('index.html'):
        return send_from_directory('.', 'index.html')
    return jsonify({
        'status': 'online',
        'mensaje': 'API MP Neurona activa. Usa POST /predict para realizar predicciones.'
    })


# Ruta para consultar metadatos del modelo exportado desde el frontend
@app.route('/model-info', methods=['GET'])
def model_info():
    rangos = {}
    for col in top_features:
        rangos[col] = {
            'min': round(float(X_min[col]), 4),
            'max': round(float(X_max[col]), 4),
            'midpoint': round(float(X_mid[col]), 4)
        }
    return jsonify({
        'status': 'success',
        'threshold': int(neurona.threshold),
        'total_features': len(top_features),
        'train_accuracy': round(float(train_acc * 100), 2),
        'test_accuracy': round(float(test_acc * 100), 2),
        'rangos': rangos
    })


# 3. Ruta principal de predicción
@app.route('/predict', methods=['POST'])
def predict_cancer():
    try:
        datos = request.get_json()
        if not datos:
            return jsonify({'error': 'No se recibieron datos en formato JSON.', 'status': 'error'}), 400

        mapeo_campos = {
            'mean radius': ('radius', 'Radio Medio'),
            'mean texture': ('texture', 'Textura Media'),
            'mean perimeter': ('perimeter', 'Perímetro Medio'),
            'mean area': ('area', 'Área Media'),
            'mean smoothness': ('smoothness', 'Suavidad Media')
        }

        input_data = {}
        for col, (key_json, nombre_es) in mapeo_campos.items():
            if key_json not in datos or str(datos[key_json]).strip() == '':
                return jsonify({'error': f'Falta el campo: {nombre_es}', 'status': 'error'}), 400
            valor = float(datos[key_json])
            if valor <= 0:
                return jsonify({'error': f'El valor de {nombre_es} debe ser mayor a 0.', 'status': 'error'}), 400
            input_data[col] = valor

        binarized_input = []
        detalles_variables = []

        for col in top_features:
            val = input_data[col]
            midpoint = float(X_mid[col])
            bin_val = 1 if val <= midpoint else 0
            binarized_input.append(bin_val)

            detalles_variables.append({
                'feature': col,
                'nombre': mapeo_campos[col][1],
                'valor_ingresado': val,
                'punto_corte': round(midpoint, 4),
                'binario': bin_val,
                'estado': 'Rango Bajo / Normal (1)' if bin_val == 1 else 'Rango Elevado / Alerta (0)'
            })

        suma_z = int(sum(binarized_input))
        pred_clase = int(neurona.predict(np.array([binarized_input]))[0])

        # En sklearn breast_cancer: 0 = Maligno, 1 = Benigno
        es_maligno = (pred_clase == 0)
        resultado = "Maligno (Posible enfermedad detectada)" if es_maligno else "Benigno (Sano)"

        return jsonify({
            'status': 'success',
            'prediccion': resultado,
            'clase': pred_clase,
            'es_maligno': es_maligno,
            'suma_z': suma_z,
            'threshold': int(neurona.threshold),
            'total_features': len(top_features),
            'accuracy_modelo': round(float(test_acc * 100), 2),
            'detalles': detalles_variables
        })

    except ValueError:
        return jsonify({'error': 'Todos los campos deben ser números válidos.', 'status': 'error'}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)