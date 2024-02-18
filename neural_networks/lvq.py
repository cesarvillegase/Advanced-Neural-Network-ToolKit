import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


class LVQ:
    def __init__(self):
        self.vectors = None

    @staticmethod
    def norm_data(data):  # normalize the inputs and vectors to the range [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    @staticmethod
    def init_vectors(data, y):  # Init the vectors
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

                distances = np.linalg.norm(self.vectors - sample, axis=1)
                winner = np.argmin(distances)

                # output winner it´s equal to actual output
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
            distances = np.linalg.norm(self.vectors - sample, axis=1)
            winner = np.argmin(distances)
            winners.append(winner)

        return np.array(winners)


def plot(normalized_data, normalized_vectors, y, title, test=False):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=y, cmap='summer', marker='o', s=100, label='data')
        plt.scatter(normalized_vectors[:, 0], normalized_vectors[:, 1], c='red', marker='x', s=200, label='vectors')
        if test:
            for i, t in enumerate(y):
                plt.annotate(t, (normalized_data[i, 0], normalized_data[i, 1]))

        plt.xlabel('characteristic x1')
        plt.ylabel('characteristic y1')
        plt.legend()
        plt.title(title)
        plt.show()


# TESTING    

# Training data
X_train_lvq = np.array([[5.2, 6.7], [2.0, 3.0], [4.0, 5.0], [7.9, 6.1], [1.0, 2.0], [7.1, 8.9],
                        [2.0, 1.0], [5.1, 7.2], [3.0, 3.0], [7.9, 5.2], [4.0, 5.0], [2.9, 3.2],
                        [4.1, 5.7], [1.2, 2.8], [7.9, 6.2], [5.2, 6.1], [9.2, 8.1], [2.2, 1.1],
                        [3.98, 2.0], [4.8, 5.2], [9.9, 8.2], [7.1, 5.7], [5.2, 7.9], [7.2, 8.1]])

# Labels
y_train_lvq = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
                        1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])

# Test data
X_test_lvq = np.array([[5.0, 8.0], [9.0, 8.0], [2.0, 9.0], [4.0, 8.0], [4.0, 7.0],
                       [2.0, 6.0], [3.0, 1.0], [1.0, 4.0], [1.0, 1.0], [4.0, 3.0],
                       [5.9, 6.2], [2.6, 3.2], [4.8, 5.1], [1.7, 2.2], [2.9, 1.5],
                       [4.5, 5.2], [9.0, 8.0], [7.2, 5.9], [5.1, 7.8], [7.2, 8.9]])

# Test labels
y_test_lvq = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                       0, 1, 1, 1, 1, 1, 0, 0, 0, 0])

model = LVQ()
norm_X_train_lvq = model.norm_data(X_train_lvq)
norm_vectors = model.init_vectors(norm_X_train_lvq, y_train_lvq)
plot(X_train_lvq, norm_vectors, y_train_lvq, title='Before the training')

trained_vectors_lvq = model.train(norm_X_train_lvq, y_train_lvq, delta=0.1, epoch_max=500)
plot(norm_X_train_lvq, trained_vectors_lvq, y_train_lvq, title='After the training')

norm_test_data_lvq = model.norm_data(X_test_lvq)
labels_predicted_lvq = model.test(norm_test_data_lvq)
plot(norm_test_data_lvq, trained_vectors_lvq, y_test_lvq, title='Test LVQ', test=True)

accuracy = accuracy_score(y_test_lvq, labels_predicted_lvq)
print(f"> Accuracy of the model: {accuracy:.2f}")
print("True labels:", y_test_lvq)
print("Predicted labels:", labels_predicted_lvq)
