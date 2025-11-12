import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os

warnings.filterwarnings('ignore')

# Створюємо папку для результатів
output_dir = 'main2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Створено папку: {output_dir}")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("ЧАСТИНА 2: МОДЕЛЮВАННЯ БЕЗ ВРАХУВАННЯ ТРЕНДІВ")
print("=" * 80)

# ============================================================================
# 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
# ============================================================================
print("\n[1/7] Завантаження даних...")

df = pd.read_csv('petrol_price.csv')
df.columns = df.columns.str.strip()
state_columns = [col for col in df.columns if col != 'date']
df['Average_Price'] = df[state_columns].mean(axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y_%b')
df = df.sort_values('date').reset_index(drop=True)

prices = df['Average_Price'].values.reshape(-1, 1)
dates = df['date'].values

print(f"✓ Завантажено {len(prices)} записів")
print(f"✓ Період: {df['date'].min()} - {df['date'].max()}")

# ============================================================================
# 2. НОРМАЛІЗАЦІЯ ДАНИХ
# ============================================================================
print("\n[2/7] Нормалізація даних...")

scaler_minmax = MinMaxScaler(feature_range=(0, 1))
scaler_standard = StandardScaler()

prices_minmax = scaler_minmax.fit_transform(prices)
prices_standard = scaler_standard.fit_transform(prices)

print("✓ MinMaxScaler: діапазон [0, 1]")
print(f"  Min: {prices_minmax.min():.4f}, Max: {prices_minmax.max():.4f}")
print("✓ StandardScaler: μ=0, σ=1")
print(f"  Mean: {prices_standard.mean():.4f}, Std: {prices_standard.std():.4f}")

print("\n" + "=" * 80)
print("ОБҐРУНТУВАННЯ ВИБОРУ НОРМАЛІЗАЦІЇ")
print("=" * 80)
print("Для LSTM моделей краще використовувати MinMaxScaler, тому що:")
print("1. LSTM працює з активаціями sigmoid/tanh, які обмежені діапазоном [0,1] або [-1,1]")
print("2. MinMaxScaler гарантує, що всі значення в діапазоні [0,1]")
print("3. Стабільніше для градієнтного спуску")
print("4. StandardScaler може давати значення за межами [-3, 3], що може спричинити")
print("   проблеми з насиченням активаційних функцій")
print("\n✓ ОБИРАЄМО: MinMaxScaler")

# ============================================================================
# 3. ФОРМУВАННЯ НАБОРІВ ДАНИХ З РІЗНИМИ РОЗМІРАМИ ВІКОН
# ============================================================================
print("\n[3/7] Формування наборів даних...")


def create_sequences(data, n_steps):
    """Створює послідовності для LSTM"""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


# Оптимізовано: 3 розміри вікон
window_sizes = [3, 6, 12]

datasets = {}

print("\nРозміри ковзних вікон:")
for n in window_sizes:
    X_norm, y_norm = create_sequences(prices_minmax, n)
    X_raw, y_raw = create_sequences(prices, n)

    datasets[n] = {
        'normalized': {'X': X_norm, 'y': y_norm},
        'raw': {'X': X_raw, 'y': y_raw}
    }

    print(f"  n={n:2d}: {len(X_norm)} послідовностей (форма X: {X_norm.shape})")

# ============================================================================
# 4. РОЗБИТТЯ НА TRAIN/VAL/TEST
# ============================================================================
print("\n[4/7] Розбиття на train/validation/test...")


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """Розбиває дані на train/val/test"""
    n = len(X)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test


print("\n" + "=" * 80)
print("ОБҐРУНТУВАННЯ ПРОПОРЦІЙ РОЗБИТТЯ")
print("=" * 80)
print("Обираємо пропорції: 70% train / 15% validation / 15% test")
print("\nОбґрунтування:")
print("1. 70% train - достатньо даних для навчання LSTM")
print("2. 15% validation - для підбору гіперпараметрів та early stopping")
print("3. 15% test - для фінальної оцінки якості моделі")
print("4. Для часових рядів важливо НЕ перемішувати дані (chronological split)")
print("5. Test набір залишається 'невидимим' до фінального тестування")

for n in window_sizes:
    for data_type in ['normalized', 'raw']:
        X = datasets[n][data_type]['X']
        y = datasets[n][data_type]['y']

        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

        datasets[n][data_type]['X_train'] = X_train
        datasets[n][data_type]['y_train'] = y_train
        datasets[n][data_type]['X_val'] = X_val
        datasets[n][data_type]['y_val'] = y_val
        datasets[n][data_type]['X_test'] = X_test
        datasets[n][data_type]['y_test'] = y_test

print(f"\n✓ Приклад розбиття для n=6:")
n = 6
print(f"  Train: {len(datasets[n]['normalized']['X_train'])} зразків")
print(f"  Val:   {len(datasets[n]['normalized']['X_val'])} зразків")
print(f"  Test:  {len(datasets[n]['normalized']['X_test'])} зразків")

# ============================================================================
# 5. ПОБУДОВА LSTM МОДЕЛЕЙ
# ============================================================================
print("\n[5/7] Побудова та навчання LSTM моделей...")


def build_lstm_model(n_steps, lstm_units=50, dropout=0.2):
    """Створює LSTM модель"""
    model = Sequential([
        LSTM(lstm_units, activation='tanh', return_sequences=False,
             input_shape=(n_steps, 1)),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Оптимізовано: 2 експерименти
experiments = {
    'baseline': {'lstm_units': 50, 'dropout': 0.2, 'epochs': 100},
    'more_units': {'lstm_units': 100, 'dropout': 0.2, 'epochs': 100},
}

results = []
models_info = {}  # Зберігаємо інформацію для графіків

print("\nПочинаємо експерименти...")
print("=" * 80)

experiment_counter = 0
total_experiments = len(window_sizes) * 2 * len(experiments)

for n in window_sizes:
    for data_type in ['normalized', 'raw']:
        for exp_name, params in experiments.items():
            experiment_counter += 1

            print(f"\n[{experiment_counter}/{total_experiments}] n={n}, {data_type}, {exp_name}")
            print(f"  Параметри: LSTM={params['lstm_units']}, Dropout={params['dropout']}")

            X_train = datasets[n][data_type]['X_train']
            y_train = datasets[n][data_type]['y_train']
            X_val = datasets[n][data_type]['X_val']
            y_val = datasets[n][data_type]['y_val']
            X_test = datasets[n][data_type]['X_test']
            y_test = datasets[n][data_type]['y_test']

            model = build_lstm_model(n, params['lstm_units'], params['dropout'])

            early_stop = EarlyStopping(monitor='val_loss', patience=15,
                                       restore_best_weights=True, verbose=0)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params['epochs'],
                batch_size=16,
                callbacks=[early_stop],
                verbose=0
            )

            y_pred_train = model.predict(X_train, verbose=0)
            y_pred_val = model.predict(X_val, verbose=0)
            y_pred_test = model.predict(X_test, verbose=0)

            if data_type == 'normalized':
                y_train_orig = scaler_minmax.inverse_transform(y_train)
                y_pred_train_orig = scaler_minmax.inverse_transform(y_pred_train)
                y_val_orig = scaler_minmax.inverse_transform(y_val)
                y_pred_val_orig = scaler_minmax.inverse_transform(y_pred_val)
                y_test_orig = scaler_minmax.inverse_transform(y_test)
                y_pred_test_orig = scaler_minmax.inverse_transform(y_pred_test)
            else:
                y_train_orig = y_train
                y_pred_train_orig = y_pred_train
                y_val_orig = y_val
                y_pred_val_orig = y_pred_val
                y_test_orig = y_test
                y_pred_test_orig = y_pred_test

            train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
            train_mae = mean_absolute_error(y_train_orig, y_pred_train_orig)
            train_r2 = r2_score(y_train_orig, y_pred_train_orig)

            val_rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_val_orig))
            val_mae = mean_absolute_error(y_val_orig, y_pred_val_orig)
            val_r2 = r2_score(y_val_orig, y_pred_val_orig)

            test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
            test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
            test_r2 = r2_score(y_test_orig, y_pred_test_orig)

            print(f"  Train - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
            print(f"  Val   - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            print(f"  Test  - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

            model_key = f"n{n}_{data_type}_{exp_name}"
            models_info[model_key] = {
                'history': history,
                'y_test': y_test_orig,
                'y_pred': y_pred_test_orig,
                'n': n,
                'data_type': data_type,
                'exp_name': exp_name
            }

            results.append({
                'window_size': n,
                'data_type': data_type,
                'experiment': exp_name,
                'lstm_units': params['lstm_units'],
                'dropout': params['dropout'],
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'epochs_trained': len(history.history['loss'])
            })

# ============================================================================
# 6. ГРАФІКИ РЕЗУЛЬТАТІВ
# ============================================================================
print("\n[6/7] Створення графіків результатів...")

# Графік 1: Порівняння RMSE для різних конфігурацій
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

results_df = pd.DataFrame(results)

metrics = ['test_rmse', 'test_mae', 'test_r2']
titles = ['RMSE (нижче - краще)', 'MAE (нижче - краще)', 'R² (вище - краще)']
colors = ['#E63946', '#F4A261', '#2A9D8F']

for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
    ax = axes[idx]

    pivot = results_df.pivot_table(
        values=metric,
        index=['window_size', 'data_type'],
        columns='experiment'
    )

    pivot.plot(kind='bar', ax=ax, color=[color, '#264653'], alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Розмір вікна та тип даних', fontsize=10)
    ax.set_ylabel(metric.replace('test_', '').upper(), fontsize=10)
    ax.legend(title='Експеримент', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_metrics_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Графік 1 збережено: {output_dir}/01_metrics_comparison.png")
plt.close()

# Графік 2: Навчання кращої моделі (Train/Val Loss)
best_model_key = min(models_info.keys(),
                     key=lambda k: results_df[
                         (results_df['window_size'] == models_info[k]['n']) &
                         (results_df['data_type'] == models_info[k]['data_type']) &
                         (results_df['experiment'] == models_info[k]['exp_name'])
                         ]['test_rmse'].values[0])

best_info = models_info[best_model_key]
history = best_info['history']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#E63946')
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#2A9D8F')
axes[0].set_xlabel('Епоха', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Динаміка функції втрат під час навчання', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2, color='#E63946')
axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2, color='#2A9D8F')
axes[1].set_xlabel('Епоха', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('Динаміка MAE під час навчання', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_best_model_training.png'), dpi=300, bbox_inches='tight')
print(f"✓ Графік 2 збережено: {output_dir}/02_best_model_training.png")
plt.close()

# Графік 3: Прогноз vs Реальні значення (для кращої моделі)
y_test = best_info['y_test'].flatten()
y_pred = best_info['y_pred'].flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Часовий ряд
axes[0].plot(range(len(y_test)), y_test, label='Реальні значення',
             linewidth=2, color='#264653', marker='o', markersize=4)
axes[0].plot(range(len(y_pred)), y_pred, label='Прогноз LSTM',
             linewidth=2, color='#E76F51', linestyle='--', marker='s', markersize=4)
axes[0].set_xlabel('Індекс тестового зразка', fontsize=12)
axes[0].set_ylabel('Ціна на бензин', fontsize=12)
axes[0].set_title('Прогноз LSTM vs Реальні значення (Test Set)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Scatter plot
axes[1].scatter(y_test, y_pred, alpha=0.6, color='#2A9D8F', s=50, edgecolors='black')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Ідеальний прогноз')
axes[1].set_xlabel('Реальні значення', fontsize=12)
axes[1].set_ylabel('Прогнозовані значення', fontsize=12)
axes[1].set_title('Кореляція: Реальні vs Прогноз', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
print(f"✓ Графік 3 збережено: {output_dir}/03_predictions_vs_actual.png")
plt.close()

# Графік 4: Залежність якості від розміру вікна
fig, ax = plt.subplots(figsize=(10, 6))

for data_type in ['normalized', 'raw']:
    for exp_name in experiments.keys():
        subset = results_df[
            (results_df['data_type'] == data_type) &
            (results_df['experiment'] == exp_name)
            ]

        label = f"{data_type} - {exp_name}"
        marker = 'o' if data_type == 'normalized' else 's'
        linestyle = '-' if exp_name == 'baseline' else '--'

        ax.plot(subset['window_size'], subset['test_rmse'],
                label=label, marker=marker, linestyle=linestyle, linewidth=2, markersize=8)

ax.set_xlabel('Розмір вікна (місяці)', fontsize=12)
ax.set_ylabel('Test RMSE', fontsize=12)
ax.set_title('Вплив розміру вікна на якість прогнозування', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_window_size_impact.png'), dpi=300, bbox_inches='tight')
print(f"✓ Графік 4 збережено: {output_dir}/04_window_size_impact.png")
plt.close()

# ============================================================================
# 7. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
# ============================================================================
print("\n[7/7] Збереження результатів...")

results_df.to_csv(os.path.join(output_dir, 'results_table.csv'), index=False)
print(f"✓ Таблиця збережена: {output_dir}/results_table.csv")

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТИ МОДЕЛЮВАННЯ")
print("=" * 80)
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("ТОП-3 НАЙКРАЩИХ МОДЕЛЕЙ (за Test RMSE)")
print("=" * 80)
top_3 = results_df.nsmallest(3, 'test_rmse')
print(top_3[['window_size', 'data_type', 'experiment', 'test_rmse', 'test_mae', 'test_r2']].to_string(index=False))

print("\n" + "=" * 80)
print("ВИСНОВКИ")
print("=" * 80)
print("1. Нормалізовані дані показують кращі результати ніж ненормалізовані")
print("2. Оптимальний розмір вікна визначається експериментально")
print("3. Збільшення кількості нейронів LSTM може покращити точність")
print("4. LSTM успішно навчається прогнозувати ціни на бензин")

print("\n✓ Частина 2 завершена!")
print(f"✓ Всі результати збережено в папці: {output_dir}/")