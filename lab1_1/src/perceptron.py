import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(-0.1, 0.1, input_size + 1)
        self.learning_rate = 0.1
        self.training_errors = []

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        inputs_with_bias = np.append(inputs, 1)
        weighted_sum = np.dot(self.weights, inputs_with_bias)
        return self.activation_function(weighted_sum)

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


class PerceptronAlternating(Perceptron):
    def __init__(self, input_size):
        super().__init__(input_size)

    def train_alternating(self, training_data, training_labels, max_epochs=1000):
        class0_indices = np.where(training_labels == 0)[0]
        class1_indices = np.where(training_labels == 1)[0]

        max_class_size = max(len(class0_indices), len(class1_indices))

        if len(class0_indices) < max_class_size:
            class0_indices = np.pad(class0_indices,
                                    (0, max_class_size - len(class0_indices)),
                                    mode='wrap')
        if len(class1_indices) < max_class_size:
            class1_indices = np.pad(class1_indices,
                                    (0, max_class_size - len(class1_indices)),
                                    mode='wrap')

        alternating_indices = []
        for i in range(max_class_size):
            alternating_indices.append(class0_indices[i])
            alternating_indices.append(class1_indices[i])

        epoch = 0
        while epoch < max_epochs:
            total_error = 0
            for idx in alternating_indices:
                inputs = training_data[idx]
                expected = training_labels[idx]
                prediction = self.predict(inputs)
                error = expected - prediction
                total_error += abs(error)
                if error != 0:
                    inputs_with_bias = np.append(inputs, 1)
                    for j in range(len(self.weights)):
                        self.weights[j] += self.learning_rate * error * inputs_with_bias[j]
            self.training_errors.append(total_error)
            if total_error == 0:
                print(f"Поперемінне навчання завершено на епосі {epoch + 1}")
                break
            epoch += 1
        if epoch == max_epochs:
            print(f"Поперемінне навчання завершено після {max_epochs} епох")


class PerceptronRandom(Perceptron):
    def __init__(self, input_size):
        super().__init__(input_size)

    def train_random(self, training_data, training_labels, max_epochs=10000):
        n_samples = len(training_data)
        epoch = 0
        consecutive_zero_errors = 0

        while epoch < max_epochs:
            total_error = 0
            indices = np.random.choice(n_samples, n_samples, replace=True)

            for idx in indices:
                inputs = training_data[idx]
                expected = training_labels[idx]
                prediction = self.predict(inputs)
                error = expected - prediction
                total_error += abs(error)
                if error != 0:
                    inputs_with_bias = np.append(inputs, 1)
                    for j in range(len(self.weights)):
                        self.weights[j] += self.learning_rate * error * inputs_with_bias[j]

            self.training_errors.append(total_error)

            if total_error == 0:
                consecutive_zero_errors += 1
                if consecutive_zero_errors >= 10:
                    print(f"Випадкове навчання завершено на епосі {epoch + 1} (10 епох без помилок)")
                    break
            else:
                consecutive_zero_errors = 0

            epoch += 1

        if epoch == max_epochs:
            print(f"Випадкове навчання завершено після {max_epochs} епох")