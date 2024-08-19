import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import joblib  # Для сохранения и загрузки масштабатора

def load_and_prepare_data(file_path, time_step=60):
    df = pd.read_csv(file_path)
    prices = df['price'].values.reshape(-1, 1)

    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Подготовка данных для LSTM
    def create_dataset(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_prices, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

# Загрузка данных для нескольких токенов
data_folder = r'C:\Users\prosh\Desktop\а ну-ка\token predictor\data'
tokens = ['ETH', 'BTC', 'BNB', 'SOL', 'ARB']

X_all, y_all = [], []
scalers = []

for token in tokens:
    X, y, scaler = load_and_prepare_data(os.path.join(data_folder, f'{token}_historical_prices.csv'))
    X_all.append(X)
    y_all.append(y)
    scalers.append(scaler)  # Добавляем масштабатор в список

# Объединение данных
X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

# Построение модели LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_all.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Увеличенное количество эпох
model.fit(X_all, y_all, epochs=500, batch_size=128)  # Увеличен размер батча

# Сохранение модели и масштабатора
model_folder = r'C:\Users\prosh\Desktop\а ну-ка\token predictor\model'
os.makedirs(model_folder, exist_ok=True)
model.save(os.path.join(model_folder, 'lstm_model.keras'))

# Сохраняем первый масштабатор для использования в будущем (можно сохранить все, если они разные)
joblib.dump(scalers[0], os.path.join(model_folder, 'scaler.save'))

print("Модель и масштабатор сохранены в папке 'model'")
