from src.generating import create_datasets
from src.visualizing import plot_classification_results
from src.perceptron import Perceptron, PerceptronAlternating, PerceptronRandom
import numpy as np

def normalize_minmax(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals), min_vals, max_vals

def normalize_l2(data):
    norms = np.sqrt(np.sum(data ** 2, axis=0))
    return data / norms

def main():
    training_data_orig, training_labels, test_data_orig, test_labels = create_datasets()

    print()
    print("ВМІСТ НАВЧАЛЬНОЇ ВИБІРКИ")
    print(" № | Час розгону (с) | Макс. швидкість (км/год) | Клас | Тип автомобіля")
    print("-----------------------------------------------------------------------")

    for i, (data, label) in enumerate(zip(training_data_orig, training_labels)):
        class_num = label
        class_name = "Спортивний" if label == 1 else "Звичайний"
        print(f"{i + 1:<2d} | {data[0]:<13.1f} | {data[1]:<23.0f} | {class_num:<4d} | {class_name:<10s}")

    print("-----------------------------------------------------------------------")
    print()
    print("ВМІСТ ТЕСТОВОЇ ВИБІРКИ")
    print(" № | Час розгону (с) | Макс. швидкість (км/год) | Клас | Тип автомобіля")
    print("-----------------------------------------------------------------------")

    for i, (data, label) in enumerate(zip(test_data_orig, test_labels)):
        class_num = label
        class_name = "Спортивний" if label == 1 else "Звичайний"
        print(f"{i + 1:<2d} | {data[0]:<13.1f} | {data[1]:<23.0f} | {class_num:<4d} | {class_name:<10s}")

    print("-----------------------------------------------------------------------")
    print()
    training_data_minmax, min_vals, max_vals = normalize_minmax(training_data_orig)
    test_data_minmax = (test_data_orig - min_vals) / (max_vals - min_vals)
    perceptron_basic = Perceptron(input_size=2)
    perceptron_basic.train(training_data_minmax, training_labels)
    train_accuracy_basic, train_predictions_basic = perceptron_basic.evaluate(training_data_minmax, training_labels)
    test_accuracy_basic, test_predictions_basic = perceptron_basic.evaluate(test_data_minmax, test_labels)

    print()
    print("ДОДАТКОВЕ ЗАВДАННЯ")
    print()
    print("Метод 1. Поперемінна подача об'єктів різних класів.")

    perceptron_alt = PerceptronAlternating(input_size=2)
    perceptron_alt.train_alternating(training_data_minmax, training_labels)
    train_accuracy_alt, train_predictions_alt = perceptron_alt.evaluate(training_data_minmax, training_labels)
    test_accuracy_alt, test_predictions_alt = perceptron_alt.evaluate(test_data_minmax, test_labels)

    print(f"Поперемінна подача - точність на навчальній вибірці: {train_accuracy_alt:.1f}%.")
    print(f"Поперемінна подача - точність на тестовій вибірці: {test_accuracy_alt:.1f}%.")
    print()
    print("Метод 2. Випадкова вибірка з навчальної вибірки.")

    perceptron_random = PerceptronRandom(input_size=2)
    perceptron_random.train_random(training_data_minmax, training_labels, max_epochs=10000)
    train_accuracy_random, train_predictions_random = perceptron_random.evaluate(training_data_minmax, training_labels)
    test_accuracy_random, test_predictions_random = perceptron_random.evaluate(test_data_minmax, test_labels)

    print(f"Випадкова вибірка - точність на навчальній вибірці: {train_accuracy_random:.1f}%")
    print(f"Випадкова вибірка - точність на тестовій вибірці: {test_accuracy_random:.1f}%")
    print()

    training_data_l2 = normalize_l2(training_data_orig)
    test_data_l2 = normalize_l2(test_data_orig)

    print("НАВЧАЛЬНА ВИБІРКА (L2-НОРМАЛІЗОВАНІ ДАНІ).")
    print(" № | Час розгону (норм) | Макс. швидкість (норм) | Клас | Тип автомобіля")
    print("-----------------------------------------------------------------------")

    for i, (data, label) in enumerate(zip(training_data_l2, training_labels)):
        class_num = label
        class_name = "Спортивний" if label == 1 else "Звичайний"
        print(f"{i + 1:<2d} | {data[0]:<16.4f} | {data[1]:<20.4f} | {class_num:<4d} | {class_name:<10s}")

    print("-----------------------------------------------------------------------")
    print()
    print("ТЕСТОВА ВИБІРКА (L2-НОРМАЛІЗОВАНІ ДАНІ).")
    print(" № | Час розгону (норм) | Макс. швидкість (норм) | Клас | Тип автомобіля")
    print("-----------------------------------------------------------------------")

    for i, (data, label) in enumerate(zip(test_data_l2, test_labels)):
        class_num = label
        class_name = "Спортивний" if label == 1 else "Звичайний"
        print(f"{i + 1:<2d} | {data[0]:<16.4f} | {data[1]:<20.4f} | {class_num:<4d} | {class_name:<10s}")

    print("-----------------------------------------------------------------------")
    print()

    perceptron_l2 = Perceptron(input_size=2)
    perceptron_l2.train(training_data_l2, training_labels)
    train_accuracy_l2, train_predictions_l2 = perceptron_l2.evaluate(training_data_l2, training_labels)
    test_accuracy_l2, test_predictions_l2 = perceptron_l2.evaluate(test_data_l2, test_labels)

    print(f"L2-нормалізація - точність на навчальній вибірці: {train_accuracy_l2:.1f}%.")
    print(f"L2-нормалізація - точність на тестовій вибірці: {test_accuracy_l2:.1f}%.")

    print()
    print("ПОРІВНЯЛЬНА ТАБЛИЦЯ З ОТРИМАНИМИ ПОКАЗНИКАМИ ТОЧНОСТІ ДЛЯ ВСІХ РЕЗУЛЬТАТІВ МОДЕЛЮВАННЯ")
    print("+--------------------------+------------+------------------+------------------+")
    print("| Метод навчання           | Нормалізація | Навчальна точність | Тестова точність |")
    print("+--------------------------+------------+------------------+------------------+")
    print(f"| Послідовна подача        | Min-Max    | {train_accuracy_basic:<16.1f}% | {test_accuracy_basic:<14.1f}% |")
    print(f"| Поперемінна подача       | Min-Max    | {train_accuracy_alt:<16.1f}% | {test_accuracy_alt:<14.1f}% |")
    print(f"| Випадкова вибірка        | Min-Max    | {train_accuracy_random:<16.1f}% | {test_accuracy_random:<14.1f}% |")
    print(f"| Базовий                  | L2         | {train_accuracy_l2:<16.1f}% | {test_accuracy_l2:<14.1f}% |")
    print("+--------------------------+------------+------------------+------------------+")
    print()

    # Детальні результати для L2-нормалізації
    print(f"ТОЧНІСТЬ НА НАВЧАЛЬНІЙ ВИБІРЦІ З L2-НОРМАЛІЗАЦІЄЮ - {train_accuracy_l2:.1f}%.")
    print(" Зразок | Очікуване    | Передбачене   | Правильно")
    print("---------------------------------------------------")

    for i, (expected, predicted) in enumerate(zip(training_labels, train_predictions_l2)):
        correct = "Так" if expected == predicted else "Ні"
        exp_name = "Спортивний" if expected == 1 else "Звичайний"
        pred_name = "Спортивний" if predicted == 1 else "Звичайний"
        print(f"{i + 1:<7d} | {exp_name:<12s} | {pred_name:<13s} | {correct:<9s}")

    print()
    print(f"ТОЧНІСТЬ НА ТЕСТОВІЙ ВИБІРЦІ З L2-НОРМАЛІЗАЦІЄЮ - {test_accuracy_l2:.1f}%.")
    print(" Зразок | Очікуване    | Передбачене   | Правильно")
    print("---------------------------------------------------")

    for i, (expected, predicted) in enumerate(zip(test_labels, test_predictions_l2)):
        correct = "Так" if expected == predicted else "Ні"
        exp_name = "Спортивний" if expected == 1 else "Звичайний"
        pred_name = "Спортивний" if predicted == 1 else "Звичайний"
        print(f"{i + 1:<7d} | {exp_name:<12s} | {pred_name:<13s} | {correct:<9s}")

    print()

    if test_accuracy_basic == test_accuracy_alt == test_accuracy_random == test_accuracy_l2 == 100:
        print("Всі методи показали відмінні результати з точністю 100%!")
    else:
        best_method = max(
            ("Базовий", test_accuracy_basic),
            ("Поперемінна подача", test_accuracy_alt),
            ("Випадкова вибірка", test_accuracy_random),
            ("L2-нормалізація", test_accuracy_l2),
            key=lambda x: x[1]
        )
        print(f"Найкращий результат показав метод: {best_method[0]} з точністю {best_method[1]:.1f}%.")

    print()
    print("ПОБУДОВА ГРАФІКІВ...")

    plot_classification_results(
        training_data_orig, training_labels,
        test_data_orig, test_labels,
        perceptron_basic,
        test_predictions_basic,
        min_vals, max_vals,
        title_suffix=" - Базовий метод"
    )

    plot_classification_results(
        training_data_orig, training_labels,
        test_data_orig, test_labels,
        perceptron_alt,
        test_predictions_alt,
        min_vals, max_vals,
        title_suffix=" - Поперемінна подача"
    )

if __name__ == "__main__":
    main()