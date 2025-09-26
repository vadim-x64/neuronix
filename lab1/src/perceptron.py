import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

class Perceptron:
    def __init__(self, input_size):

        # коефіцієнти рівняння прямої
        self.weights = np.random.uniform(-0.1, 0.1, input_size + 1)

        # коефіцієнт, який визначає, наскільки сильно потрібно "посувати" лінію після кожної помилки
        self.learning_rate = 0.1

        self.training_errors = []

    # тригер або перемикач
    def activation_function(self, x):
        return 1 if x >= 0 else 0

    # прогноз
    def predict(self, inputs):
        inputs_with_bias = np.append(inputs, 1)
        weighted_sum = np.dot(self.weights, inputs_with_bias)

        return self.activation_function(weighted_sum)

    # процес навчання
    def train(self, training_data, training_labels, max_epochs=1000):
        epoch = 0

        while epoch < max_epochs:
            total_error = 0

            for i, inputs in enumerate(training_data):
                prediction = self.predict(inputs)
                expected = training_labels[i]
                error = expected - prediction
                total_error += abs(error)

                if error != 0:
                    inputs_with_bias = np.append(inputs, 1)

                    for j in range(len(self.weights)):
                        self.weights[j] += self.learning_rate * error * inputs_with_bias[j]

            self.training_errors.append(total_error)

            if total_error == 0:
                print(f"Навчання завершено на епосі {epoch + 1}")
                break

            epoch += 1

        if epoch == max_epochs:
            print(f"Навчання завершено після {max_epochs} епох")

    # тестові дані без зміни ваг
    def evaluate(self, test_data, test_labels):
        correct = 0
        predictions = []

        for i, inputs in enumerate(test_data):
            prediction = self.predict(inputs)
            predictions.append(prediction)

            if prediction == test_labels[i]:
                correct += 1

        accuracy = correct / len(test_data) * 100

        return accuracy, predictions