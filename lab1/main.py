from src.generating import create_datasets
from src.visualizing import plot_classification_results
from src.perceptron import Perceptron

def main():
    print()
    print("====== ЛАБОРАТОРНА РОБОТА №1 ======")
    print("Прості нейронні архітектури. Вирішення задачі класифікації.")
    print("Варіант 15")

    training_data_orig, training_labels, test_data_orig, test_labels = create_datasets()

    min_vals = training_data_orig.min(axis=0)
    max_vals = training_data_orig.max(axis=0)
    training_data = (training_data_orig - min_vals) / (max_vals - min_vals)
    test_data = (test_data_orig - min_vals) / (max_vals - min_vals)

    print()
    print("Об’єктом класифікації буде автомобіль.")
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
        print(f"{i+1:2d} | {data[0]:8.1f}     | {data[1]:11.0f}        | {class_name}")

    print(f"Загальна кількість навчальних зразків: {len(training_data_orig)}")
    print()
    print("4. ВМІСТ ТЕСТОВОЇ ВИБІРКИ")
    print(" № | Розгін (сек) | Швидкість (км/год) | Клас")
    print("-" * 50)

    for i, (data, label) in enumerate(zip(test_data_orig, test_labels)):
        class_name = "Спортивний" if label == 1 else "Звичайний"
        print(f"{i+1:2d} | {data[0]:8.1f}     | {data[1]:11.0f}        | {class_name}")

    print(f"Загальна кількість тестових зразків: {len(test_data_orig)}")
    print()
    print("====== ОСНОВНЕ ЗАВДАННЯ ======")

    perceptron = Perceptron(input_size=2)

    print()
    print(f"Початкові ваги: {perceptron.weights}")
    print()

    perceptron.train(training_data, training_labels)

    print(f"Фінальні ваги: {perceptron.weights}")

    train_accuracy, train_predictions = perceptron.evaluate(training_data, training_labels)

    print()
    print(f"ТОЧНІСТЬ НА НАВЧАЛЬНІЙ ВИБІРЦІ - {train_accuracy:.1f}%")
    print("Зразок | Очікуване | Передбачене | Правильно")
    print("-" * 48)

    for i, (expected, predicted) in enumerate(zip(training_labels, train_predictions)):
        correct = "Так" if expected == predicted else "Ні"
        exp_name = "Спортивний" if expected == 1 else "Звичайний"
        pred_name = "Спортивний" if predicted == 1 else "Звичайний"
        print(f"{i+1:6d} | {exp_name:9s} | {pred_name:11s} | {correct:9s}")

    test_accuracy, test_predictions = perceptron.evaluate(test_data, test_labels)

    print()
    print(f"ТОЧНІСТЬ НА ТЕСТОВІЙ ВИБІРЦІ - {test_accuracy:.1f}%")
    print("Зразок | Очікуване | Передбачене | Правильно")
    print("-" * 48)

    for i, (expected, predicted) in enumerate(zip(test_labels, test_predictions)):
        correct = "Так" if expected == predicted else "Ні"
        exp_name = "Спортивний" if expected == 1 else "Звичайний"
        pred_name = "Спортивний" if predicted == 1 else "Звичайний"
        print(f"{i+1:6d} | {exp_name:9s} | {pred_name:11s} | {correct:9s}")

    print()

    if test_accuracy == 100:
        print("Модель показує відмінні результати!")
    elif test_accuracy >= 80:
        print("Модель показує хороші результати!")
    elif test_accuracy >= 50:
        print("Модель показує непогані результати!")
    else:
        print("Модель потребує доопрацювання")

    print(f"Рівняння роздільної лінії (для нормалізованих даних): "
          f"{perceptron.weights[0]:.3f}*x1 + {perceptron.weights[1]:.3f}*x2 + {perceptron.weights[2]:.3f} = 0")
    print("де x1 - нормалізований час розгону, x2 - нормалізована максимальна швидкість")

    print()
    print("ПОБУДОВА ГРАФІКА...")

    plot_classification_results(
        training_data_orig, training_labels,
        test_data_orig, test_labels,
        perceptron,
        test_predictions,
        min_vals, max_vals
    )

if __name__ == "__main__":
    main()