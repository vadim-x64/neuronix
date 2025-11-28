import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
BASE_DIR = 'dataset'
RESULTS_DIR = 'results'

def check_folders_part1():
    paths = [
        f'{BASE_DIR}/part1/class_1',
        f'{BASE_DIR}/part1/class_0',
        f'{BASE_DIR}/production',
        RESULTS_DIR
    ]

    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"Створено папку: {p}")
        if p != RESULTS_DIR and not os.listdir(p):
            dummy = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.putText(dummy, "No Data", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(p, 'dummy.jpg'), dummy)
            print(f"-> Додано dummy.jpg у {p}")

def plot_history(history, filename):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path)
    plt.show()
    print(f"Графік збережено у {save_path}")

def build_binary_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout для боротьби з перенавчанням
        layers.Dense(1, activation='sigmoid')  # Sigmoid для бінарного виходу (0 або 1)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    check_folders_part1()
    print("--- Запуск Частини 1 (Binary) ---")
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    try:
        train_gen = datagen.flow_from_directory(
            f'{BASE_DIR}/part1',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training'
        )

        val_gen = datagen.flow_from_directory(
            f'{BASE_DIR}/part1',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation'
        )

    except Exception as e:
        print(f"Помилка завантаження даних: {e}")
        return

    if train_gen.samples == 0:
        print("Немає зображень для навчання! Завантажте фото в dataset/part1.")
        return

    model = build_binary_model()
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    plot_history(history, "part1_results.png")
    prod_path = f'{BASE_DIR}/production'
    print(f"\nПеревірка на Production Dataset ({prod_path}):")

    for fname in os.listdir(prod_path):
        fpath = os.path.join(prod_path, fname)
        if fpath.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = tf.keras.preprocessing.image.load_img(fpath, target_size=IMG_SIZE)
            img_arr = tf.keras.preprocessing.image.img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0) / 255.0
            pred = model.predict(img_arr, verbose=0)[0][0]
            label = "Є ОБ'ЄКТ" if pred > 0.5 else "НЕМАЄ ОБ'ЄКТУ"

            print(f"{fname}: {label} ({pred:.4f})")

if __name__ == "__main__":
    main()