from src.generating import create_datasets
from src.visualizing import plot_classification_results
from src.perceptron import Perceptron

def main():
    training_data_orig, training_labels, test_data_orig, test_labels = create_datasets()
    min_vals = training_data_orig.min(axis=0)
    max_vals = training_data_orig.max(axis=0)
    training_data = (training_data_orig - min_vals) / (max_vals - min_vals)
    test_data = (test_data_orig - min_vals) / (max_vals - min_vals)

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
    print("ОСНОВНЕ ЗАВДАННЯ")
    print()

    perceptron = Perceptron(input_size=2)

    print(f"Початкові ваги: {perceptron.weights}")
    perceptron.train(training_data, training_labels)
    print(f"Фінальні ваги: {perceptron.weights}")

    train_accuracy, train_predictions = perceptron.evaluate(training_data, training_labels)

    print()
    print(f"ТОЧНІСТЬ НА НАВЧАЛЬНІЙ ВИБІРЦІ - {train_accuracy:.1f}%.")
    print(" Зразок | Очікуване    | Передбачене   | Правильно")
    print("---------------------------------------------------")

    for i, (expected, predicted) in enumerate(zip(training_labels, train_predictions)):
        correct = "Так" if expected == predicted else "Ні"
        exp_name = "Спортивний" if expected == 1 else "Звичайний"
        pred_name = "Спортивний" if predicted == 1 else "Звичайний"
        print(f"{i + 1:<7d} | {exp_name:<12s} | {pred_name:<13s} | {correct:<9s}")

    test_accuracy, test_predictions = perceptron.evaluate(test_data, test_labels)

    print()
    print(f"ТОЧНІСТЬ НА ТЕСТОВІЙ ВИБІРЦІ - {test_accuracy:.1f}%.")
    print(" Зразок | Очікуване    | Передбачене   | Правильно")
    print("---------------------------------------------------")

    for i, (expected, predicted) in enumerate(zip(test_labels, test_predictions)):
        correct = "Так" if expected == predicted else "Ні"
        exp_name = "Спортивний" if expected == 1 else "Звичайний"
        pred_name = "Спортивний" if predicted == 1 else "Звичайний"
        print(f"{i + 1:<7d} | {exp_name:<12s} | {pred_name:<13s} | {correct:<9s}")

    print()

    if test_accuracy == 100:
        print("Модель показує відмінні результати!")
    elif test_accuracy >= 80:
        print("Модель показує хороші результати!")
    elif test_accuracy >= 50:
        print("Модель показує непогані результати!")
    else:
        print("Модель потребує доопрацювання!")

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