# Kohonen Self-Organizing Maps (SOM)

import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def norm_data(self, data):
        min_value = np.min(data)
        max_value = np.max(data)
        normalized_data = (data - min_value) / (max_value - min_value)
        return normalized_data
    
    # Function to initialize weights
    def init_weights(self, num_neurons, inputs):
        weights = np.random.rand(num_neurons, inputs)
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        normalized_weights = (weights - min_weight) / (max_weight - min_weight)
        return normalized_weights
    
    #=============TRAIN PHASE=============
    def train(self, num_neurons, data, inputs, alpha, epoch_max):
        normalized_data = self.norm_data(data)
        normalized_weights = self.init_weights(num_neurons, inputs)
        for epoch in range(epoch_max):
            for input in normalized_data:
                # Compute the euclidean distance
                distances = np.linalg.norm(normalized_weights - input, axis=1)
                # Winner neuron
                k = np.argmin(distances)
                
                # Update the weights of the winner neuron
                normalized_weights[k] += alpha * (input - normalized_weights[k])
                
            # Update the neighboring neurons
            for neighbor in range(num_neurons):
                # Compute the distance from the winner neuron
                dist_tok = np.linalg.norm(normalized_weights[neighbor] - normalized_weights[k])

                if dist_tok < 0.7:
                    normalized_weights[neighbor] += alpha * np.exp(-(dist_tok**2) / (2 * (alpha**2))) * (input - normalized_weights[neighbor])

        return normalized_weights
    
def plot(data, weights, title):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        plt.scatter(data[:,0], data[:, 1],c='r', marker='x', label='Training Data')
        plt.scatter(weights[:, 0], weights[:, 1], c='b', marker='o', label = 'Neurons')
        plt.title(title)
        plt.legend()
        plt.show()        
        
# TESTING

# @title > Función generadora de datos
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

    input = np.vstack(data)
    y = np.concatenate(labels)

    return input, y

X_train_som, labels_som = generate_data(num_points_p_class=20, num_classes=2)

model = SOM()
norm_X_train_som = model.norm_data(X_train_som)
weights_som = model.init_weights(num_neurons=8, inputs=2)
plot(norm_X_train_som, weights_som, title='Before the training')

pretrained_weights_som = model.train(num_neurons=8, data=norm_X_train_som, inputs=2, alpha=0.4, epoch_max=400)
plot(norm_X_train_som, pretrained_weights_som, title='After training')

