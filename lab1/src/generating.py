import numpy as np

def create_datasets():
    training_data = np.array([
        [8.5, 180],
        [9.2, 160],
        [7.8, 200],
        [11.0, 150],
        [6.5, 220],
        [10.1, 170],
        [3.8, 280],
        [4.2, 300],
        [3.5, 265],
        [4.8, 260],
        [3.1, 260],
        [5.8, 200],
        [2.2, 250],
        [4.5, 320],
        [3.8, 270]
    ])

    training_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])

    test_data = np.array([
        [9.5, 190],
        [7.2, 210],
        [12.0, 140],
        [4.5, 278],
        [3.9, 310],
        [2.6, 330],
        [7.4, 190]
    ])

    test_labels = np.array([0, 0, 0, 1, 1, 1, 0])

    return training_data, training_labels, test_data, test_labels