import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
results_dir = 'results'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def target_function(x):
    return 2 * x**2 - x / (np.exp(x) + 1)

x_range = np.linspace(-5, 5, 100)
y_true = target_function(x_range)
noise = np.random.normal(0, 0.5, y_true.shape)
y_noisy = y_true + noise
X = x_range
y = y_noisy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(12, 7))
plt.plot(x_range, y_true, label='Оригінальна функція', color='blue', linewidth=2)
plt.scatter(X_train, y_train, label='Навчальні дані', color='blue', alpha=0.6)
plt.scatter(X_test, y_test, label='Тестові дані', color='green', alpha=0.8)
plt.title('Цільова функція та згенеровані дані для навчання')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'initial_data.png'))
plt.show()

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
print("Архітектура моделі:")
model.summary()

print("\nПочаток навчання моделі...")
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=1)
print("Навчання завершено.")

y_pred = model.predict(X_test).flatten()
r2 = r2_score(y_test, y_pred)
print(f"\nКоефіцієнт детермінації R^2: {r2:.4f}")

plt.figure(figsize=(12, 7))
plt.plot(x_range, y_true, label='Оригінальна функція', color='blue', linewidth=2)
plt.scatter(X_test, y_test, label='Тестові дані (реальні)', color='green', alpha=0.8)
plt.scatter(X_test, y_pred, label='Прогноз моделі', color='red', marker='x', s=100)
plt.title('Результати роботи нейромережі на тестових даних')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'model_results.png'))
plt.show()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Втрати на навчанні (Loss)')
plt.plot(history.history['val_loss'], label='Втрати на валідації (Validation Loss)')
plt.title('Історія навчання: Втрати')
plt.xlabel('Епохи')
plt.ylabel('Втрати (MSE)')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='MAE на навчанні')
plt.plot(history.history['val_mae'], label='MAE на валідації')
plt.title('Історія навчання: Середня абсолютна помилка')
plt.xlabel('Епохи')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_history.png'))
plt.show()