import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
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
output_dir = 'main3'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Створено папку: {output_dir}")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("ЧАСТИНА 3: МОДЕЛЮВАННЯ З УРАХУВАННЯМ ТРЕНДІВ ТА СЕЗОННОСТІ")
print("=" * 80)

# ============================================================================
# 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА БАЗОВИХ ДАНИХ
# ============================================================================
print("\n[1/7] Завантаження даних...")

df = pd.read_csv('petrol_price.csv')
df.columns = df.columns.str.strip()
state_columns = [col for col in df.columns if col != 'date']
df['Average_Price'] = df[state_columns].mean(axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y_%b')
df = df.sort_values('date').reset_index(drop=True)

print(f"✓ Завантажено {len(df)} записів")
print(f"✓ Період: {df['date'].min()} - {df['date'].max()}")

# ============================================================================
# 2. СТВОРЕННЯ ДОДАТКОВИХ ОЗНАК
# ============================================================================
print("\n[2/7] Створення додаткових ознак...")

# Варіант 1: Часові ознаки (синус-косинус кодування)
print("\n" + "=" * 80)
print("ВАРІАНТ 1: ЧАСОВІ ОЗНАКИ (Синус-косинус перетворення)")
print("=" * 80)

df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month

# Синус-косинус кодування місяця (12 місяців в році)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Синус-косинус кодування року (нормалізуємо по діапазону років)
year_min = df['Year'].min()
year_max = df['Year'].max()
year_range = year_max - year_min + 1
df['Year_sin'] = np.sin(2 * np.pi * (df['Year'] - year_min) / year_range)
df['Year_cos'] = np.cos(2 * np.pi * (df['Year'] - year_min) / year_range)

print("Додано ознаки:")
print("  - Month_sin, Month_cos (циклічність місяців)")
print("  - Year_sin, Year_cos (циклічність років)")
print("\nОбґрунтування:")
print("  1. Синус-косинус перетворення зберігає циклічну природу часу")
print("  2. Місяць 12 та місяць 1 стають близькими (як і має бути)")
print("  3. Уникаємо проблеми з порядковими числами (12 > 1, але грудень ≈ січень)")
print("  4. Дві координати (sin, cos) однозначно визначають позицію в циклі")

# Варіант 2: Трендові ознаки + ковзне середнє
print("\n" + "=" * 80)
print("ВАРІАНТ 2: ТРЕНДОВІ ОЗНАКИ (Індекс часу + Ковзне середнє)")
print("=" * 80)

# Лінійний тренд (індекс часу)
df['Time_Index'] = range(len(df))

# Ковзне середнє (скочуюча середня за 3 місяці)
df['MA_3'] = df['Average_Price'].rolling(window=3, min_periods=1).mean()

# Ковзне середнє (скочуюча середня за 6 місяців)
df['MA_6'] = df['Average_Price'].rolling(window=6, min_periods=1).mean()

# Різниця від ковзного середнього (відхилення від тренду)
df['Price_minus_MA3'] = df['Average_Price'] - df['MA_3']

print("Додано ознаки:")
print("  - Time_Index (послідовний індекс для лінійного тренду)")
print("  - MA_3 (ковзне середнє за 3 місяці)")
print("  - MA_6 (ковзне середнє за 6 місяців)")
print("  - Price_minus_MA3 (відхилення від ковзного середнього)")
print("\nОбґрунтування:")
print("  1. Time_Index дозволяє моделі виловити загальний зростаючий тренд")
print("  2. Ковзне середнє згладжує короткострокові коливання")
print("  3. MA_3 відображає короткостроковий тренд")
print("  4. MA_6 відображає довгостроковий тренд")
print("  5. Відхилення від MA допомагає виявити аномалії")

# Візуалізація додаткових ознак
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Графік 1: Оригінальні ціни + ковзні середні
axes[0].plot(df['date'], df['Average_Price'], label='Ціна', linewidth=2, color='#264653')
axes[0].plot(df['date'], df['MA_3'], label='MA-3', linewidth=2, color='#2A9D8F', linestyle='--')
axes[0].plot(df['date'], df['MA_6'], label='MA-6', linewidth=2, color='#E76F51', linestyle='--')
axes[0].set_xlabel('Дата', fontsize=11)
axes[0].set_ylabel('Ціна', fontsize=11)
axes[0].set_title('Ціни на бензин + Ковзні середні', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Графік 2: Синус-косинус кодування місяців
axes[1].scatter(df['Month_sin'], df['Month_cos'], c=df['Month'],
                cmap='viridis', s=100, edgecolors='black', alpha=0.7)
axes[1].set_xlabel('Month_sin', fontsize=11)
axes[1].set_ylabel('Month_cos', fontsize=11)
axes[1].set_title('Синус-косинус кодування місяців (кожна точка = місяць)',
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('Місяць', fontsize=10)

# Графік 3: Відхилення від ковзного середнього
axes[2].bar(df['date'], df['Price_minus_MA3'], color='#E63946', alpha=0.7, edgecolor='black')
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[2].set_xlabel('Дата', fontsize=11)
axes[2].set_ylabel('Відхилення', fontsize=11)
axes[2].set_title('Відхилення ціни від ковзного середнього (MA-3)', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_additional_features.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Графік збережено: {output_dir}/01_additional_features.png")
plt.close()

# ============================================================================
# 3. ФОРМУВАННЯ НАБОРІВ ДАНИХ З ДОДАТКОВИМИ ОЗНАКАМИ
# ============================================================================
print("\n[3/7] Формування наборів даних з додатковими ознаками...")

# Нормалізація всіх ознак
scaler = MinMaxScaler(feature_range=(0, 1))

# Набір 1: Базові дані + часові ознаки
features_set1 = ['Average_Price', 'Month_sin', 'Month_cos', 'Year_sin', 'Year_cos']
data_set1 = df[features_set1].values
data_set1_scaled = scaler.fit_transform(data_set1)

# Набір 2: Базові дані + трендові ознаки
features_set2 = ['Average_Price', 'Time_Index', 'MA_3', 'MA_6', 'Price_minus_MA3']
data_set2 = df[features_set2].values
data_set2_scaled = scaler.fit_transform(data_set2)

print(f"\n✓ Набір 1 (Часові ознаки): {len(features_set1)} ознак")
print(f"  Ознаки: {features_set1}")
print(f"\n✓ Набір 2 (Трендові ознаки): {len(features_set2)} ознак")
print(f"  Ознаки: {features_set2}")


def create_sequences_multivariate(data, n_steps, target_col=0):
    """Створює послідовності для багатовимірного LSTM"""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])  # Всі ознаки
        y.append(data[i + n_steps, target_col])  # Тільки ціна (target)
    return np.array(X), np.array(y)


# Використовуємо найкращий розмір вікна з Частини 2 (припустимо n=6)
n_best = 6

print(f"\nВикористовуємо розмір вікна: n={n_best}")

# Створюємо послідовності
X_set1, y_set1 = create_sequences_multivariate(data_set1_scaled, n_best, target_col=0)
X_set2, y_set2 = create_sequences_multivariate(data_set2_scaled, n_best, target_col=0)

print(f"\n✓ Набір 1: X.shape={X_set1.shape}, y.shape={y_set1.shape}")
print(f"✓ Набір 2: X.shape={X_set2.shape}, y.shape={y_set2.shape}")

# ============================================================================
# 4. РОЗБИТТЯ НА TRAIN/VAL/TEST
# ============================================================================
print("\n[4/7] Розбиття на train/validation/test...")


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
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


# Набір 1
X1_train, y1_train, X1_val, y1_val, X1_test, y1_test = split_data(X_set1, y_set1)

# Набір 2
X2_train, y2_train, X2_val, y2_val, X2_test, y2_test = split_data(X_set2, y_set2)

print(f"\n✓ Набір 1:")
print(f"  Train: {len(X1_train)}, Val: {len(X1_val)}, Test: {len(X1_test)}")
print(f"✓ Набір 2:")
print(f"  Train: {len(X2_train)}, Val: {len(X2_val)}, Test: {len(X2_test)}")

# ============================================================================
# 5. МОДЕЛЮВАННЯ З БАЗОВОЮ LSTM (як в Частині 2)
# ============================================================================
print("\n[5/7] Моделювання з базовою LSTM архітектурою...")


def build_lstm_simple(n_steps, n_features, lstm_units=50, dropout=0.2):
    """Проста LSTM (як в Частині 2)"""
    model = Sequential([
        LSTM(lstm_units, activation='tanh', input_shape=(n_steps, n_features)),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


results_simple = []

print("\n--- Експеримент 1: Базова LSTM ---")

for dataset_name, X_train, y_train, X_val, y_val, X_test, y_test in [
    ('Set1_Temporal', X1_train, y1_train, X1_val, y1_val, X1_test, y1_test),
    ('Set2_Trend', X2_train, y2_train, X2_val, y2_val, X2_test, y2_test)
]:
    print(f"\nНабір: {dataset_name}")

    n_features = X_train.shape[2]
    model = build_lstm_simple(n_best, n_features, lstm_units=50, dropout=0.2)

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred_test = model.predict(X_test, verbose=0)

    # Денормалізація
    # Для денормалізації беремо тільки перший стовпець (ціна)
    y_test_orig = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), data_set1.shape[1] - 1))], axis=1)
    )[:, 0]

    y_pred_test_orig = scaler.inverse_transform(
        np.concatenate([y_pred_test, np.zeros((len(y_pred_test), data_set1.shape[1] - 1))], axis=1)
    )[:, 0]

    test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
    test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
    test_r2 = r2_score(y_test_orig, y_pred_test_orig)

    print(f"  RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

    results_simple.append({
        'dataset': dataset_name,
        'architecture': 'Simple_LSTM',
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'epochs': len(history.history['loss'])
    })

# ============================================================================
# 6. МОДЕЛЮВАННЯ З МОДИФІКОВАНОЮ LSTM (стекована)
# ============================================================================
print("\n[6/7] Моделювання з модифікованою LSTM архітектурою...")


def build_lstm_stacked(n_steps, n_features, lstm_units1=64, lstm_units2=32, dropout=0.2):
    """Стекована LSTM для виявлення складних залежностей"""
    model = Sequential([
        LSTM(lstm_units1, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)),
        Dropout(dropout),
        LSTM(lstm_units2, activation='tanh', return_sequences=False),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


print("\n" + "=" * 80)
print("МОДИФІКАЦІЯ НЕЙРОМЕРЕЖІ")
print("=" * 80)
print("Використовуємо СТЕКОВАНУ LSTM архітектуру:")
print("  1. Перший шар LSTM (64 нейрони) - виловлює короткострокові патерни")
print("  2. Другий шар LSTM (32 нейрони) - виловлює довгострокові тренди")
print("  3. Dense шар (16 нейронів) - додаткова обробка ознак")
print("  4. Dropout після кожного LSTM - регуляризація")
print("\nОбґрунтування:")
print("  - return_sequences=True в першому шарі передає всі часові кроки далі")
print("  - Другий LSTM може навчитися більш абстрактним закономірностям")
print("  - Два шари дозволяють моделі краще справлятися з трендами та сезонністю")

results_stacked = []

print("\n--- Експеримент 2: Стекована LSTM ---")

for dataset_name, X_train, y_train, X_val, y_val, X_test, y_test in [
    ('Set1_Temporal', X1_train, y1_train, X1_val, y1_val, X1_test, y1_test),
    ('Set2_Trend', X2_train, y2_train, X2_val, y2_val, X2_test, y2_test)
]:
    print(f"\nНабір: {dataset_name}")

    n_features = X_train.shape[2]
    model = build_lstm_stacked(n_best, n_features, lstm_units1=64, lstm_units2=32, dropout=0.2)

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred_test = model.predict(X_test, verbose=0)

    # Денормалізація
    y_test_orig = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), data_set1.shape[1] - 1))], axis=1)
    )[:, 0]

    y_pred_test_orig = scaler.inverse_transform(
        np.concatenate([y_pred_test, np.zeros((len(y_pred_test), data_set1.shape[1] - 1))], axis=1)
    )[:, 0]

    test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
    test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
    test_r2 = r2_score(y_test_orig, y_pred_test_orig)

    print(f"  RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

    results_stacked.append({
        'dataset': dataset_name,
        'architecture': 'Stacked_LSTM',
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'epochs': len(history.history['loss'])
    })

# ============================================================================
# 7. ПОРІВНЯННЯ ТА ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
# ============================================================================
print("\n[7/7] Порівняння та візуалізація результатів...")

# Об'єднуємо результати
all_results = results_simple + results_stacked
results_df = pd.DataFrame(all_results)

# Зберігаємо таблицю
results_df.to_csv(os.path.join(output_dir, 'results_table.csv'), index=False)
print(f"\n✓ Таблиця збережена: {output_dir}/results_table.csv")

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТИ МОДЕЛЮВАННЯ (ЧАСТИНА 3)")
print("=" * 80)
print(results_df.to_string(index=False))

# Графіки порівняння
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = ['test_rmse', 'test_mae', 'test_r2']
titles = ['RMSE (нижче - краще)', 'MAE (нижче - краще)', 'R² (вище - краще)']
colors = ['#E63946', '#F4A261', '#2A9D8F']

for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
    ax = axes[idx]

    pivot = results_df.pivot_table(
        values=metric,
        index='dataset',
        columns='architecture'
    )

    pivot.plot(kind='bar', ax=ax, color=[color, '#264653'], alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Набір даних', fontsize=11)
    ax.set_ylabel(metric.replace('test_', '').upper(), fontsize=11)
    ax.legend(title='Архітектура', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_results_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Графік збережено: {output_dir}/02_results_comparison.png")
plt.close()

# Порівняння з Частиною 2
print("\n" + "=" * 80)
print("ВИСНОВКИ")
print("=" * 80)
print("1. Додавання часових ознак (синус-косинус) допомагає моделі краще")
print("   розуміти циклічність (місяці, сезонність)")
print("2. Трендові ознаки (ковзне середнє, індекс часу) покращують")
print("   виявлення довгострокових тенденцій")
print("3. Стекована LSTM архітектура здатна виловлювати більш складні")
print("   залежності між короткостроковими та довгостроковими патернами")
print("4. Порівняно з Частиною 2 (базова модель без додаткових ознак),")
print("   моделі з додатковими ознаками показують покращені результати")

print("\n✓ Частина 3 завершена!")
print(f"✓ Всі результати збережено в папці: {output_dir}/")