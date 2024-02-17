import numpy as np
import matplotlib.pyplot as plt

class LVQ:
    def __init__(self):
        self.vectors = None

    def norm_data(self, data): # normalize the inputs and vectors to the range [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    def init_vectors(self, data, y): # Init the vectors
        num_classes = len(np.unique(y))
        normalized_vectors = []

        for i in range(num_classes):
            indices = np.where(y == i)[0]
            selected_indices = np.random.choice(indices)
            normalized_vectors.append(data[selected_indices])

        normalized_vectors = np.array(normalized_vectors)
        return normalized_vectors

    def train(self, normalized_data, y, delta, epoch_max):
        self.vectors = self.init_vectors(normalized_data, y)
        for epoch in range(epoch_max):
            for data_point in range(len(normalized_data)):
                sample = normalized_data[data_point]
                output = y[data_point]

                distances = np.linalg.norm(self.vectors - sample, axis = 1)
                winner = np.argmin(distances)

                # salida ganadora es igual actual
                if y[winner] == output:
                  self.vectors[winner] += delta * (sample - self.vectors[winner])
                else:
                  # if the label of the winner is not equal to the output the winner vector get move away
                  self.vectors[winner] -= delta * (sample - self.vectors[winner])

        return self.vectors

    def test(self, normalized_tt_data):
        winners = []
        for data_point in range(len(normalized_tt_data)):
            sample = normalized_tt_data[data_point]
            distances = np.linalg.norm(self.vectors - sample, axis = 1)
            winner = np.argmin(distances)
            winners.append(winner)

        return np.array(winners)

def plot(normalized_data, normalized_vectors, y, title, test=False):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        plt.scatter(normalized_data[:,0], normalized_data[:,1], c=y, cmap='summer', marker='o', s=100, label='data')
        plt.scatter(normalized_vectors[:,0], normalized_vectors[:,1], c='red', marker='x', s=200, label='vectors')
        if test == True:
            for i, t in enumerate(y):
                plt.annotate(t, (normalized_data[i, 0], normalized_data[i, 1]))

        plt.xlabel('characteristic x1')
        plt.ylabel('characteristic y1')
        plt.legend()
        plt.title(title)
        plt.show()
        
