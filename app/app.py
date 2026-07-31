import sys
import os
import numpy as np
from flask import Flask, request, jsonify

# Добавляем корень проекта в путь, чтобы видеть src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_model, load_scaler

app = Flask(__name__)

# Загружаем модель и скейлер один раз при старте
model = load_model()
scaler = load_scaler()


@app.route('/health', methods=['GET'])
def health():
    # Проверка работоспособности сервиса
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    # Принимаем JSON с признаками, возвращаем прогноз
    try:
        data = request.get_json(force=True)
        
        if 'features' not in data:
            return jsonify({'error': 'Поле features обязательно'}), 400
        
        # 23 признака (без ID и без таргета)
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0].tolist()
        
        result = {
            'prediction': int(prediction),
            'probability': float(probabilities[int(prediction)]),
            'class': 'default' if prediction == 1 else 'no default'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)