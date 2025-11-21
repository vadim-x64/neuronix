import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import cv2
import os
import glob
from scipy import ndimage
import random

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
OUTPUT_DIR = 'lab_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Версія TensorFlow: {tf.__version__}")
print("\n[1/4] Завантаження та підготовка даних MNIST...")

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels_cat = utils.to_categorical(train_labels, 10)
test_labels_cat = utils.to_categorical(test_labels, 10)

train_images_cnn = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images_cnn = test_images.reshape((test_images.shape[0], 28, 28, 1))

def visualize_mnist_samples(images, labels, save_name="mnist_samples.png"):
    plt.figure(figsize=(12, 5))
    for i in range(10):
        indices = np.where(labels == i)[0]
        count = min(len(indices), 2)
        random_indices = np.random.choice(indices, count, replace=False)
        for j, idx in enumerate(random_indices):
            ax = plt.subplot(2, 10, i + 1 + j * 10)
            plt.imshow(images[idx], cmap='gray')
            plt.axis('off')
            if j == 0:
                ax.set_title(str(i))
    plt.suptitle("Приклади з MNIST")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name))

visualize_mnist_samples(train_images, train_labels)

print("\n[2/4] Обробка власного датасету...")

def get_best_shift(img):
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty

def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

def process_custom_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(img_thresh)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    pad = 10
    img_thresh = img_thresh[max(0, y - pad):min(img_thresh.shape[0], y + h + pad),
    max(0, x - pad):min(img_thresh.shape[1], x + w + pad)]

    rows, cols = img_thresh.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))

    if cols == 0 or rows == 0: return None

    img_thresh = cv2.resize(img_thresh, (cols, rows), interpolation=cv2.INTER_AREA)

    new_img = np.zeros((28, 28), dtype=np.uint8)
    start_row = (28 - rows) // 2
    start_col = (28 - cols) // 2
    new_img[start_row:start_row + rows, start_col:start_col + cols] = img_thresh
    shiftx, shifty = get_best_shift(new_img)
    shifted = shift(new_img, shiftx, shifty)

    return shifted.astype('float32') / 255.0

def load_custom_dataset(folder_path='my_digits'):
    custom_images = []
    custom_labels = []

    if not os.path.exists(folder_path):
        print(f"УВАГА: Папка '{folder_path}' не знайдена. Створіть папку та додайте фото.")
        return None, None

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    print(f"Знайдено {len(files)} файлів.")
    if not files: return None, None

    for f in files:
        try:
            filename = os.path.basename(f)

            if filename[0].isdigit():
                label = int(filename[0])
                img = process_custom_image(f)
                if img is not None:
                    custom_images.append(img)
                    custom_labels.append(label)
        except Exception as e:
            print(f"Помилка файлу {f}: {e}")

    if not custom_images: return None, None
    return np.array(custom_images), np.array(custom_labels)

custom_x, custom_y = load_custom_dataset('my_digits')

has_custom = False
if custom_x is not None:
    has_custom = True
    print(f"Завантажено власних зображень: {len(custom_x)}")

    plt.figure(figsize=(12, 3))
    count = min(len(custom_x), 10)
    for i in range(count):
        plt.subplot(1, 10, i + 1)
        plt.imshow(custom_x[i], cmap='gray')
        plt.title(f"L: {custom_y[i]}")
        plt.axis('off')
    plt.suptitle("Власні дані (оброблені)")
    plt.savefig(os.path.join(OUTPUT_DIR, "custom_data_viz.png"))

    custom_x_cnn = custom_x.reshape((-1, 28, 28, 1))
    custom_y_cat = utils.to_categorical(custom_y, 10)
else:
    print("!!! Власні дані відсутні. Використовуємо частину MNIST !!!")
    indices = np.random.choice(range(len(test_images)), 20, replace=False)
    custom_x = test_images[indices]
    custom_y = test_labels[indices]
    custom_x_cnn = test_images_cnn[indices]
    custom_y_cat = test_labels_cat[indices]


def analyze_results(model, x, y_true, title, filename_prefix):
    preds = model.predict(x)
    y_pred = np.argmax(preds, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    csv_filename = f"{filename_prefix}.csv"
    df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
    df_cm.to_csv(os.path.join(OUTPUT_DIR, csv_filename))
    print(f"Матрицю збережено у CSV: {csv_filename}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Передбачений клас')
    plt.ylabel('Справжній клас')
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_prefix}.png"))
    plt.show()

    print(f"\nЗвіт класифікації для {title}:")
    labels_present = np.unique(np.concatenate((y_true, y_pred)))
    print(classification_report(y_true, y_pred, labels=labels_present, zero_division=0))

print("\n[3/4] Навчання MLP...")

model_mlp = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_mlp = model_mlp.fit(train_images, train_labels_cat,
                            epochs=5, batch_size=64, validation_split=0.1, verbose=1)

loss_mlp, acc_mlp = model_mlp.evaluate(custom_x, custom_y_cat, verbose=0)
print(f"--> Точність MLP на тестовому наборі: {acc_mlp:.2%}")

analyze_results(model_mlp, custom_x, custom_y, "Confusion Matrix (MLP)", "cm_mlp")

print("\n[4/4] Навчання CNN (LeNet-5)...")

model_lenet = models.Sequential([
    layers.Conv2D(6, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(16, kernel_size=(5, 5), activation='relu', padding='valid'),
    layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_lenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_lenet = model_lenet.fit(train_images_cnn, train_labels_cat,
                                epochs=5, batch_size=64, validation_split=0.1, verbose=1)

loss_lenet, acc_lenet = model_lenet.evaluate(custom_x_cnn, custom_y_cat, verbose=0)
print(f"--> Точність LeNet на тестовому наборі: {acc_lenet:.2%}")

analyze_results(model_lenet, custom_x_cnn, custom_y, "Confusion Matrix (LeNet)", "cm_lenet")

print("\n=== ПІДСУМКИ ===")
print(f"MLP Accuracy:   {acc_mlp:.2%}")
print(f"LeNet Accuracy: {acc_lenet:.2%}")
print(f"Результати збережено в папку '{OUTPUT_DIR}'")