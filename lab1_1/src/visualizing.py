import os
import matplotlib.pyplot as plt
import numpy as np

def plot_classification_results(training_data, training_labels, test_data, test_labels,
                                perceptron, test_predictions,
                                min_vals, max_vals, title_suffix=""):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.figure(figsize=(12, 8))

    plt.scatter(training_data[training_labels == 0][:, 0],
                training_data[training_labels == 0][:, 1],
                c='blue', marker='o', s=100, alpha=0.7, label='Навчальні: Звичайні')
    plt.scatter(training_data[training_labels == 1][:, 0],
                training_data[training_labels == 1][:, 1],
                c='red', marker='s', s=100, alpha=0.7, label='Навчальні: Спортивні')

    test_predictions = np.array(test_predictions)
    correct_preds = test_data[test_labels == test_predictions]
    wrong_preds = test_data[test_labels != test_predictions]
    correct_labels = test_labels[test_labels == test_predictions]

    if len(correct_preds) > 0:
        correct_0 = correct_preds[correct_labels == 0]
        correct_1 = correct_preds[correct_labels == 1]
        if len(correct_0) > 0:
            plt.scatter(correct_0[:, 0], correct_0[:, 1],
                        c='cyan', marker='^', s=150, label='Тест: Звичайні (правильно)')
        if len(correct_1) > 0:
            plt.scatter(correct_1[:, 0], correct_1[:, 1],
                        c='magenta', marker='^', s=150, label='Тест: Спортивні (правильно)')

    if len(wrong_preds) > 0:
        plt.scatter(wrong_preds[:, 0], wrong_preds[:, 1],
                    c='yellow', marker='x', s=200, label='Тест: Помилка класифікації')

    x1_min, x1_max = training_data[:, 0].min() - 1, training_data[:, 0].max() + 1
    x1_range = np.linspace(x1_min, x1_max, 100)

    w = perceptron.weights
    x1_range_norm = (x1_range - min_vals[0]) / (max_vals[0] - min_vals[0])
    x2_line_norm = (-w[0] * x1_range_norm - w[2]) / w[1]
    x2_line = x2_line_norm * (max_vals[1] - min_vals[1]) + min_vals[1]

    plt.plot(x1_range, x2_line, 'g-', linewidth=3, label='Роздільна лінія')
    plt.xlabel('Час розгону до 100 км/год (сек)')
    plt.ylabel('Максимальна швидкість (км/год)')
    plt.title(f'Класифікація автомобілів: Спортивні vs Звичайні{title_suffix}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    filename = f'plots/graph{title_suffix.replace(" ", "_").replace("-", "")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Графік збережено в файл: {filename}")
    plt.show()