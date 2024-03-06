import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class HopfieldNetwork:
    def __init__(self, epoch_max):
        self.weights_r = None
        self.weights_g = None
        self.weights_b = None
        self.epoch_max = epoch_max

    def _weights_matrix(self, patterns):
        """Calculate the weight matrix for given patterns."""
        p, n = patterns.shape
        weights = np.zeros((n, n))

        for pt in patterns:
            pt = pt.reshape(-1, 1)
            weights += np.dot(pt, pt.T)

        np.fill_diagonal(weights, 0)
        weights = weights / n

        return weights

    def _neuron_update(self, weights, data):
        """Update neurons based on weights and input patterns."""
        for _ in range(self.epoch_max):
            data = np.sign(np.dot(weights, data))
        return data

    def train(self, data):
        """Train the network with the provided data."""
        patterns_r = np.array([input[:,:,0].flatten() for input in data])
        patterns_g = np.array([input[:,:,1].flatten() for input in data])
        patterns_b = np.array([input[:,:,2].flatten() for input in data])

        # Calculate the weight matrices for each color channel
        self.weights_r = self._weights_matrix(patterns_r)
        self.weights_g = self._weights_matrix(patterns_g)
        self.weights_b = self._weights_matrix(patterns_b)

    def reconstruct(self, noisy_data):
        """Reconstruct images from noisy input"""
        rec_img_r = self._neuron_update(self.weights_r, noisy_data[:, :, 0].flatten()).reshape(noisy_data.shape[0], noisy_data.shape[1])
        rec_img_g = self._neuron_update(self.weights_g, noisy_data[:, :, 1].flatten()).reshape(noisy_data.shape[0], noisy_data.shape[1])
        rec_img_b = self._neuron_update(self.weights_b, noisy_data[:, :, 2].flatten()).reshape(noisy_data.shape[0], noisy_data.shape[1])
        recovered_image = np.dstack((rec_img_r, rec_img_g, rec_img_b))
        return recovered_image
