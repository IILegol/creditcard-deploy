import pickle
import os


def load_model(path='models/model.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Файл модели не найден: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_scaler(path='models/scaler.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Файл скейлера не найден: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)