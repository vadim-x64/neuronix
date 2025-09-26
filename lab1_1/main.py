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
    print()
    print("====== ЛАБОРАТОРНА РОБОТА №1 ======")
    print("Прості нейронні архітектури. Вирішення задачі класифікації.")
    print("Варіант 15")

    training_data_orig, training_labels, test_data_orig, test_labels = create_datasets()

    print()
    print("Об'єктом класифікації буде автомобіль.")
    print()
    print("Ознаки:\n"
          "• ознака 1 - час розгону до 100 км/год (с);\n"
          "• ознака 2 - максимальна швидкість (км/год).")
    print()
    print("Класи:\n"
          "• клас 0 (звичайний автомобіль);\n"
          "• клас 1 (спортивний автомобіль).")

    print()
    print("3. ВМІСТ НАВЧАЛЬНОЇ ВИБІРКИ")
    print(" № | Розгін (сек) | Швидкість (км/год) | Клас")
    print("-" * 50)

    for i, (data, label) in enumerate(zip(training_data_orig, training_labels)):
        class_name = "Спортивний" if label == 1 else "Звичайний"
        print(f"{i + 1:2d} | {data[0]:8.1f}     | {data[1]:11.0f}        | {class_name}")

    print(f"Загальна кількість навчальних зразків: {len(training_data_orig)}")
    print()
    print("4. ВМІСТ ТЕСТОВОЇ ВИБІРКИ")
    print(" № | Розгін (сек) | Швидкість (км/год) | Клас")
    print("-" * 50)

    for i, (data, label) in enumerate(zip(test_data_orig, test_labels)):
        class_name = "Спортивний" if label == 1 else "Звичайний"
        print(f"{i + 1:2d} | {data[0]:8.1f}     | {data[1]:11.0f}        | {class_name}")

    print(f"Загальна кількість тестових зразків: {len(test_data_orig)}")

    print()

    training_data_minmax, min_vals, max_vals = normalize_minmax(training_data_orig)
    test_data_minmax = (test_data_orig - min_vals) / (max_vals - min_vals)
    perceptron_basic = Perceptron(input_size=2)
    perceptron_basic.train(training_data_minmax, training_labels)
    train_accuracy_basic, train_predictions_basic = perceptron_basic.evaluate(training_data_minmax, training_labels)
    test_accuracy_basic, test_predictions_basic = perceptron_basic.evaluate(test_data_minmax, test_labels)

    print()
    print("====== ДОДАТКОВЕ ЗАВДАННЯ ======")

    print()
    print("5. МОДИФІКАЦІЯ АЛГОРИТМУ НАВЧАННЯ")
    print()
    print("5.1 Метод 1: Поперемінна подача об'єктів різних класів")

    perceptron_alt = PerceptronAlternating(input_size=2)
    perceptron_alt.train_alternating(training_data_minmax, training_labels)

    train_accuracy_alt, train_predictions_alt = perceptron_alt.evaluate(training_data_minmax, training_labels)
    test_accuracy_alt, test_predictions_alt = perceptron_alt.evaluate(test_data_minmax, test_labels)

    print(f"Поперемінна подача - Точність на навчальній вибірці: {train_accuracy_alt:.1f}%")
    print(f"Поперемінна подача - Точність на тестовій вибірці: {test_accuracy_alt:.1f}%")

    print()
    print("5.2 Метод 2: Випадкова вибірка (10x ітерацій)")

    perceptron_random = PerceptronRandom(input_size=2)
    perceptron_random.train_random(training_data_minmax, training_labels, max_epochs=10000)

    train_accuracy_random, train_predictions_random = perceptron_random.evaluate(training_data_minmax, training_labels)
    test_accuracy_random, test_predictions_random = perceptron_random.evaluate(test_data_minmax, test_labels)

    print(f"Випадкова вибірка - Точність на навчальній вибірці: {train_accuracy_random:.1f}%")
    print(f"Випадкова вибірка - Точність на тестовій вибірці: {test_accuracy_random:.1f}%")

    print()
    print("6. НОРМУВАННЯ ДАНИХ")
    print()
    print("6.1 L2-нормалізація (варіант 15)")

    training_data_l2 = normalize_l2(training_data_orig)
    test_data_l2 = normalize_l2(test_data_orig)

    print("Нормовані навчальні дані (L2):")
    print(" № | Розгін (норм) | Швидкість (норм) | Клас")
    print("-" * 55)

    for i, (data, label) in enumerate(zip(training_data_l2, training_labels)):
        class_name = "Спортивний" if label == 1 else "Звичайний"
        print(f"{i + 1:2d} | {data[0]:12.6f}  | {data[1]:15.6f}   | {class_name}")

    print()
    print("Нормовані тестові дані (L2):")
    print(" № | Розгін (норм) | Швидкість (норм) | Клас")
    print("-" * 55)

    for i, (data, label) in enumerate(zip(test_data_l2, test_labels)):
        class_name = "Спортивний" if label == 1 else "Звичайний"
        print(f"{i + 1:2d} | {data[0]:12.6f}  | {data[1]:15.6f}   | {class_name}")

    print()
    perceptron_l2 = Perceptron(input_size=2)
    perceptron_l2.train(training_data_l2, training_labels)


    train_accuracy_l2, train_predictions_l2 = perceptron_l2.evaluate(training_data_l2, training_labels)
    test_accuracy_l2, test_predictions_l2 = perceptron_l2.evaluate(test_data_l2, test_labels)

    print()
    print(f"L2-нормалізація - Точність на навчальній вибірці: {train_accuracy_l2:.1f}%")
    print(f"L2-нормалізація - Точність на тестовій вибірці: {test_accuracy_l2:.1f}%")

    print()
    print("7. ПОРІВНЯЛЬНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
    print()
    print("| Метод навчання              | Нормалізація | Навчальна точність | Тестова точність |")
    print("|-----------------------------|--------------|--------------------|------------------|")
    print(
        f"| Базовий (послідовна подача) | Min-Max      | {train_accuracy_basic:18.1f}% | {test_accuracy_basic:16.1f}% |")
    print(f"| Поперемінна подача          | Min-Max      | {train_accuracy_alt:18.1f}% | {test_accuracy_alt:16.1f}% |")
    print(
        f"| Випадкова вибірка (10x)     | Min-Max      | {train_accuracy_random:18.1f}% | {test_accuracy_random:16.1f}% |")
    print(f"| Базовий                     | L2           | {train_accuracy_l2:18.1f}% | {test_accuracy_l2:16.1f}% |")
    print()

    print("АНАЛІЗ РЕЗУЛЬТАТІВ:")
    if test_accuracy_basic == test_accuracy_alt == test_accuracy_random == test_accuracy_l2 == 100:
        print("Всі методи показали відмінні результати з точністю 100%.")
        print("Це пояснюється тим, що класи добре лінійно відокремлюються.")
    else:
        best_method = max(
            ("Базовий", test_accuracy_basic),
            ("Поперемінна подача", test_accuracy_alt),
            ("Випадкова вибірка", test_accuracy_random),
            ("L2-нормалізація", test_accuracy_l2),
            key=lambda x: x[1]
        )
        print(f"Найкращий результат показав метод: {best_method[0]} з точністю {best_method[1]:.1f}%")

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