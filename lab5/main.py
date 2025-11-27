import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import cv2

# ==========================================
# НАЛАШТУВАННЯ ТА КОНСТАНТИ
# ==========================================
IMG_SIZE = (128, 128)  # Розмір, до якого стискаємо зображення
BATCH_SIZE = 16
EPOCHS = 20  # Для тесту можна менше, для звіту краще 15-20
BASE_DIR = 'dataset'  # Головна папка з даними
RESULTS_DIR = 'results'  # Папка для збереження графіків


# Створюємо заглушки папок та файлів, якщо їх немає
def check_folders():
    paths = [
        f'{BASE_DIR}/part1/class_1', f'{BASE_DIR}/part1/class_0',
        f'{BASE_DIR}/production',
        f'{BASE_DIR}/part2/apple', f'{BASE_DIR}/part2/plum', f'{BASE_DIR}/part2/pear',
        RESULTS_DIR
    ]
    for p in paths:
        # 1. Створюємо папку, якщо немає
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"Створено папку: {p}")

        # 2. Перевіряємо, чи папка порожня (крім папки results).
        if p == RESULTS_DIR:
            continue

        # Ігноруємо .DS_Store та інші системні файли при перевірці
        valid_files = [f for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not valid_files:
            print(f"УВАГА: Папка {p} порожня! Створюю тестове зображення (dummy.jpg), щоб код не впав.")
            # Створюємо пусте зображення-заглушку (чорний квадрат)
            dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "No Data", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            try:
                cv2.imwrite(os.path.join(p, 'dummy.jpg'), dummy_img)
            except Exception as e:
                print(f"Не вдалося створити dummy.jpg через помилку cv2: {e}.")


check_folders()


# ==========================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================

def plot_history(history, title="Model Training", filename=None):
    if not history:
        return
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    if filename:
        save_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(save_path)
        print(f"Графік збережено: {save_path}")

    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix", filename=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

    if filename:
        save_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(save_path)
        print(f"Матрицю збережено: {save_path}")

    plt.show()
    plt.close()


# ==========================================
# ЧАСТИНА 1: БІНАРНА КЛАСИФІКАЦІЯ (Яблуко vs Немає Яблука)
# ==========================================
print("\n--- ПОЧАТОК ЧАСТИНИ 1: Binary Classification ---")

# 1. Підготовка даних
datagen_part1 = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

# Перехоплюємо помилки, якщо дані відсутні
try:
    train_gen_p1 = datagen_part1.flow_from_directory(
        f'{BASE_DIR}/part1',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_gen_p1 = datagen_part1.flow_from_directory(
        f'{BASE_DIR}/part1',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
except Exception as e:
    print(f"Помилка при завантаженні даних Частини 1: {e}")
    train_gen_p1 = None


# 2. Створення моделі
def build_binary_model(kernel_size=(3, 3), dropout=False):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, kernel_size, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
    ])

    if dropout:
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Запускаємо навчання ТІЛЬКИ якщо є дані
model_p1 = build_binary_model()
if train_gen_p1 and train_gen_p1.samples > 0:
    print(f"Знайдено {train_gen_p1.samples} зображень для навчання. Починаємо...")
    history_p1 = model_p1.fit(train_gen_p1, epochs=EPOCHS, validation_data=val_gen_p1, verbose=1)
    plot_history(history_p1, "Part 1: Binary Model", filename="part1_training_history.png")

    # Перевірка на Production Dataset
    print("\nПеревірка на Production Dataset:")
    prod_dir = f'{BASE_DIR}/production'
    if os.path.exists(prod_dir) and os.listdir(prod_dir):
        for img_name in os.listdir(prod_dir):
            img_path = os.path.join(prod_dir, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0) / 255.0

                    prediction = model_p1.predict(img_array, verbose=0)
                    result = "ОБ'ЄКТ Є (Class 1)" if prediction[0][0] > 0.5 else "ОБ'ЄКТУ НЕМАЄ (Class 0)"
                    print(f"File: {img_name} -> {result} (Prob: {prediction[0][0]:.4f})")
                except Exception as e:
                    print(f"Помилка обробки файлу {img_name}: {e}")
    else:
        print("Production dataset порожній або не існує.")
else:
    print("!!! УВАГА: Немає даних для навчання Частини 1. Завантажте зображення у dataset/part1/class_1 та class_0 !!!")

# ==========================================
# ЧАСТИНА 2: БАГАТОКЛАСОВА КЛАСИФІКАЦІЯ (Яблуко, Слива, Груша)
# ==========================================
print("\n--- ПОЧАТОК ЧАСТИНИ 2: Multi-class Classification ---")

# 1. Аугментація
datagen_part2 = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

try:
    train_gen_p2 = datagen_part2.flow_from_directory(
        f'{BASE_DIR}/part2',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen_p2 = datagen_part2.flow_from_directory(
        f'{BASE_DIR}/part2',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
except Exception as e:
    print(f"Помилка при завантаженні даних Частини 2: {e}")
    train_gen_p2 = None

# Якщо даних немає, скрипт завершить роботу тут
if not train_gen_p2 or train_gen_p2.samples == 0:
    print("!!! УВАГА: Немає даних для навчання Частини 2. Завантажте зображення у dataset/part2/apple, plum, pear !!!")
else:
    class_names = list(train_gen_p2.class_indices.keys())
    print(f"Класи: {class_names}")


    # Реалізуємо 3 різні архітектури
    def get_model_v1():
        m = models.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(3, activation='softmax')
        ])
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return m


    def get_model_v2():
        m = models.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return m


    def get_model_v3():
        m = models.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(32, (5, 5), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return m


    models_list = [("Model V1 (Simple)", get_model_v1(), "v1"),
                   ("Model V2 (Deep+Dropout)", get_model_v2(), "v2"),
                   ("Model V3 (GlobalAvgPool)", get_model_v3(), "v3")]

    best_acc = 0
    best_model_name = ""

    for name, model, tag in models_list:
        print(f"\nНавчання {name}...")
        try:
            history = model.fit(train_gen_p2, epochs=EPOCHS, validation_data=val_gen_p2, verbose=0)

            val_loss, val_acc = model.evaluate(val_gen_p2, verbose=0)
            print(f"{name} -> Val Accuracy: {val_acc:.4f}")

            # Зберігаємо графік навчання для кожної моделі
            plot_history(history, f"Training {name}", filename=f"part2_history_{tag}.png")

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_name = name

            Y_pred = model.predict(val_gen_p2)
            y_pred = np.argmax(Y_pred, axis=1)
            y_true = val_gen_p2.classes

            plot_confusion_matrix(y_true, y_pred, class_names, title=f"Confusion Matrix - {name}",
                                  filename=f"part2_confusion_{tag}.png")

            print(f"Metrics for {name}:")
            print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            macro_f1 = np.mean(f1)
            micro_f1 = val_acc
            weights = np.bincount(y_true) / len(y_true)
            weighted_f1 = np.sum(f1 * weights)

            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Micro F1: {micro_f1:.4f}")
            print(f"Weighted F1: {weighted_f1:.4f}")

        except Exception as e:
            print(f"Помилка під час навчання {name}: {e}")

    print(f"\nНайкраща модель: {best_model_name} з точністю {best_acc:.4f}")