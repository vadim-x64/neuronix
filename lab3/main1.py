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
import warnings
import os
warnings.filterwarnings('ignore')
output_dir = 'main1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Створено папку: {output_dir}")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('petrol_price.csv')
df.columns = df.columns.str.strip()
state_columns = [col for col in df.columns if col != 'date']
df['Average_Price'] = df[state_columns].mean(axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y_%b')
df = df.sort_values('date')
df = df.reset_index(drop=True)
print("=" * 60)
print("ДАНІ ЗАВАНТАЖЕНО")
print("=" * 60)
print(f"Кількість записів: {len(df)}")
print(f"Період: {df['date'].min()} - {df['date'].max()}")
print(f"\nПерші 5 рядків:")
print(df[['date', 'Average_Price']].head())
print(f"\nОстанні 5 рядків:")
print(df[['date', 'Average_Price']].tail())
prices = df['Average_Price'].values
print("\n" + "=" * 60)
print("СТАТИСТИЧНІ ПОКАЗНИКИ")
print("=" * 60)
stats = {
    'Середнє арифметичне': np.mean(prices),
    'Медіана': np.median(prices),
    'Стандартне відхилення': np.std(prices),
    'Мінімум': np.min(prices),
    'Максимум': np.max(prices),
    '25% квартиль': np.percentile(prices, 25),
    '50% квартиль (медіана)': np.percentile(prices, 50),
    '75% квартиль': np.percentile(prices, 75),
    'Коефіцієнт асиметрії': pd.Series(prices).skew(),
    'Куртозис': pd.Series(prices).kurtosis()
}
for key, value in stats.items():
    print(f"{key:30s}: {value:.4f}")
print("\n" + "=" * 60)
print("АНАЛІЗ СТАТИСТИЧНИХ ПОКАЗНИКІВ")
print("=" * 60)
mean_val = stats['Середнє арифметичне']
median_val = stats['Медіана']
diff = abs(mean_val - median_val)
print(f"Різниця між середнім та медіаною: {diff:.4f}")
if diff > 2:
    print("⚠️  Значна різниця! Можливі викиди або асиметрія розподілу.")
else:
    print("✓ Невелика різниця. Розподіл близький до симетричного.")
std_val = stats['Стандартне відхилення']
mean_percent = (std_val / mean_val) * 100
print(f"\nСтандартне відхилення: {std_val:.4f} ({mean_percent:.2f}% від середнього)")
if mean_percent > 15:
    print("⚠️  Велика варіативність! Потрібна нормалізація.")
else:
    print("✓ Помірна варіативність.")
skew = stats['Коефіцієнт асиметрії']
print(f"\nКоефіцієнт асиметрії: {skew:.4f}")
if abs(skew) > 0.5:
    print("⚠️  Помітна асиметрія розподілу.")
else:
    print("✓ Розподіл близький до симетричного.")
kurt = stats['Куртозис']
print(f"\nКуртозис: {kurt:.4f}")
if kurt > 3:
    print("⚠️  Гостровершинний розподіл (можливі викиди).")
elif kurt < -1:
    print("⚠️  Плосковершинний розподіл.")
else:
    print("✓ Нормальний розподіл.")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].plot(df['date'], df['Average_Price'], linewidth=2, color='#2E86AB')
axes[0].set_xlabel('Дата', fontsize=12)
axes[0].set_ylabel('Середня ціна на бензин', fontsize=12)
axes[0].set_title('Динаміка середніх цін на бензин в Індії (2017-2022)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)
z = np.polyfit(range(len(df)), df['Average_Price'], 1)
p = np.poly1d(z)
axes[0].plot(df['date'], p(range(len(df))), "--", color='red', linewidth=2, label=f'Тренд: {z[0]:.2f}x + {z[1]:.2f}')
axes[0].legend()
axes[1].hist(df['Average_Price'], bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
axes[1].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Середнє: {mean_val:.2f}')
axes[1].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Медіана: {median_val:.2f}')
axes[1].set_xlabel('Ціна на бензин', fontsize=12)
axes[1].set_ylabel('Частота', fontsize=12)
axes[1].set_title('Розподіл цін на бензин', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
output_path = os.path.join(output_dir, '01_data_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Графіки збережено: {output_path}")
plt.show()
print("\n" + "=" * 60)
print("ВИСНОВКИ З ГРАФІКІВ")
print("=" * 60)
print("1. Графік динаміки показує чіткий ЗРОСТАЮЧИЙ ТРЕНД цін з 2017 по 2022.")
print("2. Є коливання цін (сезонність/події на ринку).")
print("3. Гістограма показує розподіл цін - переважно в діапазоні 70-110.")
print("4. Дані підходять для прогнозування за допомогою LSTM.")