import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
results_dir = 'results_task2'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

N_SAMPLES = 200
np.random.seed(42)
traffic = np.random.uniform(50, 500, N_SAMPLES)
temperature = np.random.uniform(-10, 30, N_SAMPLES)
X = np.stack([traffic, temperature], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def calculate_pm25(traffic, temperature, noise_type='gaussian', noise_level=1.0):
    base_pm25 = (traffic * 0.05) + ((temperature - 10) ** 2 * 0.02) + 10

    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, traffic.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, traffic.shape)
    else:
        noise = 0

    y = base_pm25 + noise
    return np.maximum(y, 0)

def create_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

experiments = [
    {'split': 0.2, 'noise_type': 'gaussian', 'noise_level': 2.0, 'epochs': 100},
    {'split': 0.2, 'noise_type': 'gaussian', 'noise_level': 5.0, 'epochs': 100},
    {'split': 0.2, 'noise_type': 'uniform', 'noise_level': 3.0, 'epochs': 100},
    {'split': 0.3, 'noise_type': 'gaussian', 'noise_level': 2.0, 'epochs': 100},
    {'split': 0.1, 'noise_type': 'gaussian', 'noise_level': 2.0, 'epochs': 150}
]

results_data = []
print("Початок проведення експериментів...")

for i, exp in enumerate(experiments):
    print(f"\n--- Експеримент {i + 1}/{len(experiments)} ---")
    print(f"Параметри: {exp}")

    y = calculate_pm25(X[:, 0], X[:, 1], exp['noise_type'], exp['noise_level'])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=exp['split'], random_state=42
    )

    model = create_model(input_dim=2)

    history = model.fit(
        X_train, y_train,
        epochs=exp['epochs'],
        validation_split=0.1,
        verbose=0
    )

    print("Навчання завершено.")
    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    final_val_loss = history.history['val_loss'][-1]
    print(f"Результат R^2 на тестових даних: {r2:.4f}")

    results_data.append([
        i + 1,
        exp['noise_type'],
        exp['noise_level'],
        f"{int((1 - exp['split']) * 100)}% / {int(exp['split'] * 100)}%",
        exp['epochs'],
        f"{r2:.4f}"
    ])

    if i == 0:
        print("Збереження графіків для експерименту 1...")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Результати моделювання: Реальні vs Прогнозовані (Експеримент 1)')
        plt.xlabel('Реальні значення (y_test)')
        plt.ylabel('Прогнозовані значення (y_pred)')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'actual_vs_predicted.png'))

        plt.figure(figsize=(12, 5))
        plt.plot(history.history['loss'], label='Втрати на навчанні (Loss)')
        plt.plot(history.history['val_loss'], label='Втрати на валідації (Validation Loss)')
        plt.title('Історія навчання (Експеримент 1)')
        plt.xlabel('Епохи')
        plt.ylabel('Втрати (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'training_history_task2.png'))

results_df = pd.DataFrame(
    results_data,
    columns=["№ Експерименту", "Тип шуму", "Рівень шуму", "Розподіл (Train/Test)", "Епохи", "R^2 (на тесті)"]
)

print("\n\n--- Узагальнена таблиця результатів ---")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(results_dir, 'results_table.csv'), index=False)
print(f"\nВсі результати та графіки збережено у папку: {results_dir}")