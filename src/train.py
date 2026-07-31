import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_model(data_path='data/raw/UCI_Credit_Card.csv'):
    # Загружаем данные
    df = pd.read_csv(data_path)
    
    # Убираем ID и разделяем на X и y
    X = df.drop(['ID', 'default.payment.next.month'], axis=1)
    y = df['default.payment.next.month']
    
    # Делим на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Масштабируем (важно: тот же скейлер потом используем в API)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучаем RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Проверяем качество
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print('Classification report:')
    print(classification_report(y_test, y_pred))
    print(f'ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}')
    
    # Сохраняем модель и скейлер через pickle
    os.makedirs('models', exist_ok=True)
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print('Модель и скейлер сохранены в папку models/')


if __name__ == '__main__':
    train_model()