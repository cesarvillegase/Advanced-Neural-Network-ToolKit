# Kohonen Self-Organizing Maps (SOM)

import numpy as np
import matplotlib.pyplot as plt


class SOM:
    @staticmethod
    def norm_data(data):
        min_value = np.min(data)
        max_value = np.max(data)
        normalized_data = (data - min_value) / (max_value - min_value)
        return normalized_data

    # Function to initialize weights
    @staticmethod
    def init_weights(num_neurons, input_dim):
        weights = np.random.rand(num_neurons, input_dim)
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        normalized_weights = (weights - min_weight) / (max_weight - min_weight)
        return normalized_weights

    # =============TRAIN PHASE=============
    def train(self, num_neurons, input_dim, input, alpha, epoch_max):
        normalized_weights = self.init_weights(num_neurons, input_dim)
        normalized_data = self.norm_data(input)
        for epoch in range(epoch_max):
            for data in normalized_data:
                # Compute the Euclidean distance
                distances = np.linalg.norm(normalized_weights - data, axis=1)
                # Winner neuron
                k = np.argmin(distances)

                # Update the weights of the winner neuron
                normalized_weights[k] += alpha * (data - normalized_weights[k])

                # Update the neighboring neurons
                for neighbor in range(num_neurons):
                    # Compute the distance from the winner neuron
                    dist_tok = np.linalg.norm(normalized_weights[neighbor] - normalized_weights[k])

                    if dist_tok < 0.5:
                        normalized_weights[neighbor] += alpha * np.exp(-(dist_tok ** 2) / (2 * (alpha ** 2))) * (
                                data - normalized_weights[neighbor])

        return normalized_weights


def plot(data, weights, title):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        plt.scatter(data[:, 0], data[:, 1], c='r', marker='x', label='Training Data')
        plt.scatter(weights[:, 0], weights[:, 1], c='b', marker='o', label='Neurons')
        plt.title(title)
        plt.legend()
        plt.show()

    # TESTING


def generate_data(num_points_p_class, num_classes):
    np.random.seed(42)
    data = []
    labels = []

    for i in range(num_classes):
        points = np.random.rand(num_points_p_class, 2) * 2

        if i == 1:
            points += np.array([3, 3])
        elif i == 2:
            points += np.array([0, 4])
        elif i == 3:
            points += np.array([3, 0])

        data.append(points)
        labels.append(np.full(num_points_p_class, i))

    data = np.vstack(data)
    y = np.concatenate(labels)

    return data, y


'''
X_train_som, labels_som = generate_data(num_points_p_class=20, num_classes=2)

print(X_train_som)

model = SOM()

norm_X_train_som = model.norm_data(X_train_som)
weights_som = model.init_weights(num_neurons=12, input_dim=2)
plot(norm_X_train_som, weights_som, title='Before the training')

pretrained_weights_som = model.train(num_neurons=8, input_dim=2, input=norm_X_train_som, alpha=0.4, epoch_max=1200)
plot(norm_X_train_som, pretrained_weights_som, title='After training')
'''
