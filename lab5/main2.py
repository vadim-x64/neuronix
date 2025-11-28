import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import cv2

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 15
BASE_DIR = 'dataset'
RESULTS_DIR = 'results'

def check_folders_part2():
    paths = [
        f'{BASE_DIR}/part2/apple',
        f'{BASE_DIR}/part2/pear',
        f'{BASE_DIR}/part2/plum',
        RESULTS_DIR
    ]

    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
            if p != RESULTS_DIR:
                dummy = np.zeros((128, 128, 3), dtype=np.uint8)
                cv2.putText(dummy, "No Data", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(p, 'dummy.jpg'), dummy)

def plot_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def get_model_simple(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_model_deep(num_classes):
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
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_model_avgpool(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),  # Заміна Flatten на GlobalAvg
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    check_folders_part2()
    print("--- Запуск Частини 2 (Multi-class) ---")

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    try:
        train_gen = datagen.flow_from_directory(
            f'{BASE_DIR}/part2',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )

        val_gen = datagen.flow_from_directory(
            f'{BASE_DIR}/part2',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

    except Exception as e:
        print(f"Помилка даних: {e}")
        return

    if train_gen.samples == 0:
        print("Дані відсутні у dataset/part2!")
        return

    classes = list(train_gen.class_indices.keys())
    models_list = [
        ("Simple_Model", get_model_simple(len(classes))),
        ("Deep_Dropout_Model", get_model_deep(len(classes))),
        ("GlobalAvg_Model", get_model_avgpool(len(classes)))
    ]

    best_acc = 0
    best_name = ""

    for name, model in models_list:
        print(f"\nНавчання моделі: {name}...")
        history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, verbose=1)

        Y_pred = model.predict(val_gen)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = val_gen.classes

        plot_confusion_matrix(y_true, y_pred, classes, f"part2_cm_{name}.png")
        print(f"--- Результати для {name} ---")
        print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

        val_acc = history.history['val_accuracy'][-1]
        if val_acc > best_acc:
            best_acc = val_acc
            best_name = name

    print(f"\n=== ПІДСУМОК ===")
    print(f"Найкраща модель: {best_name} з точністю {best_acc:.4f}")

if __name__ == "__main__":
    main()