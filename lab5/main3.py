import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import hamming_loss, accuracy_score, jaccard_score
import cv2

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
BASE_DIR = 'dataset'
RESULTS_DIR = 'results'

def check_folders_part3():
    path_img = f'{BASE_DIR}/part3/images'
    path_res = RESULTS_DIR

    for p in [path_img, path_res]:
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"Створено папку: {p}")

    csv_path = f'{BASE_DIR}/part3/labels.csv'
    if not os.path.exists(csv_path):
        print(f"Створюю тестовий labels.csv у {csv_path}")

        filenames = [f'dummy{i}.jpg' for i in range(1, 11)]
        df = pd.DataFrame({
            'filename': filenames,
            'apple': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'pear': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'plum': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        })
        df.to_csv(csv_path, index=False)

        for i, name in enumerate(filenames):
            dummy = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            cv2.putText(dummy, f"Img{i + 1}", (10, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(path_img, name), dummy)

def build_multilabel_model(n_classes):
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
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    check_folders_part3()
    print("--- Запуск Частини 3 (Multi-label) ---")

    csv_path = f'{BASE_DIR}/part3/labels.csv'
    img_dir = f'{BASE_DIR}/part3/images'

    try:
        df = pd.read_csv(csv_path)
        classes = list(df.columns[1:])
        print(f"Знайдені класи: {classes}")
    except Exception as e:
        print(f"Помилка читання CSV: {e}")
        return

    if len(df) < 5:
        print(f"⚠️ Занадто мало даних ({len(df)}), validation вимкнено")
        use_validation = False
    else:
        use_validation = True

    if use_validation:
        datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)

    try:
        if use_validation:
            train_gen = datagen.flow_from_dataframe(
                dataframe=df,
                directory=img_dir,
                x_col="filename",
                y_col=classes,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode="raw",
                subset='training'
            )
            val_gen = datagen.flow_from_dataframe(
                dataframe=df,
                directory=img_dir,
                x_col="filename",
                y_col=classes,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode="raw",
                subset='validation',
                shuffle=False
            )
        else:
            train_gen = datagen.flow_from_dataframe(
                dataframe=df,
                directory=img_dir,
                x_col="filename",
                y_col=classes,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode="raw"
            )
            val_gen = None

    except Exception as e:
        print(f"Помилка генератора даних: {e}")
        return

    if train_gen.samples == 0:
        print("Немає даних для навчання!")
        return

    model = build_multilabel_model(len(classes))

    if use_validation:
        model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    else:
        model.fit(train_gen, epochs=EPOCHS)

    print("\n--- Розрахунок метрик Multi-label ---")
    eval_gen = val_gen if use_validation else train_gen

    Y_pred_prob = model.predict(eval_gen)
    Y_pred = (Y_pred_prob > 0.5).astype(int)
    Y_true = eval_gen.labels

    hl = hamming_loss(Y_true, Y_pred)
    print(f"Hamming Loss: {hl:.4f}")

    em = accuracy_score(Y_true, Y_pred)
    print(f"Exact Match Accuracy: {em:.4f}")

    js = jaccard_score(Y_true, Y_pred, average='samples', zero_division=0)
    print(f"Mean Jaccard Score: {js:.4f}")

if __name__ == "__main__":
    main()