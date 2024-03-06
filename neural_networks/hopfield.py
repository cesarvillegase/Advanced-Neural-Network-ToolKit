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

    def plot_images(self, original_img, noisy_img, reconstructed_img):
        """Plot the original, noisy, and reconstructed images."""
        plt.figure(figsize=(12, 4))
        imgs = [original_img, noisy_img, reconstructed_img]
        titles = ['Original Image', 'Noisy Image', 'Reconstructed Image']
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(imgs[i].astype(np.uint8))
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

img_1_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_1.png"  # "/Users/cesarve/Documents/GitHub/Advanced-Neural-Network-ToolKit/neural_networks/images/img_1.png"
img_2_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_2.png"
img_3_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_3.png"

img_1_wn_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_1_noisy.png"
img_2_wn_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_2_noisy.png"
img_3_wn_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_3_noisy.png"

img_1 = Image.open(img_1_path)
img_2 = Image.open(img_2_path)
img_3 = Image.open(img_3_path)

img_1_wn = Image.open(img_1_wn_path)
img_2_wn = Image.open(img_2_wn_path)
img_3_wn = Image.open(img_3_wn_path)

# Make copies of the original images
img_1_original = img_1.copy()
img_2_original = img_2.copy()
img_3_original = img_3.copy()

img_1_array = np.array(img_1) / 255.0 * 2 - 1
img_2_array = np.array(img_2) / 255.0 * 2 - 1
img_3_array = np.array(img_3) / 255.0 * 2 - 1

# Make copies of the noisy images
img_1_wn_original = img_1_wn.copy()
img_2_wn_original = img_2_wn.copy()
img_3_wn_original = img_3_wn.copy()

img_1_wn_array = np.array(img_1_wn) / 255.0 * 2 - 1
img_2_wn_array = np.array(img_2_wn) / 255.0 * 2 - 1
img_3_wn_array = np.array(img_3_wn) / 255.0 * 2 - 1

# Create a Hopfield Network instance and train it
hop_net = HopfieldNetwork(epoch_max=1000)
hop_net.train([img_1_array])

# Reconstruct the noisy image
reconstructed_img = hop_net.reconstruct(img_1_wn_array)

# Plot the original, noisy, and reconstructed images
original_img = ((img_1_array + 1) / 2 * 255).astype(np.uint8)
noisy_img = ((img_1_wn_array + 1) / 2 * 255).astype(np.uint8)
rec_img = ((reconstructed_img + 1) / 2 * 255).astype(np.uint8)
hop_net.plot_images(original_img, noisy_img, rec_img)