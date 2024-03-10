import matplotlib.pyplot as plt
import numpy as np
import neurolab as nl
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
        print(recovered_image[0].shape)
        return recovered_image

def plot_images_hop(original_img, noisy_img, reconstructed_img):
    """Plot the original, noisy, and reconstructed images."""
    plt.figure(figsize=(12, 4))
    imgs = [original_img[0], noisy_img[0],
            reconstructed_img[0]]  # Access the first element since each is wrapped in a list
    titles = ['Original Image', 'Noisy Image', 'Reconstructed Image']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(imgs[i].astype(np.uint8))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class HopfieldNetworkNeurolab:
    def __init__(self):
        self.net = None

    def train(self, data):
        """
        Train the Hopfield network with the given patterns.
        :param data: np.ndarray, patterns to train the network.
        """
        pattern_shape = data.shape
        num_pixels = np.prod(pattern_shape[:-1])  # Calculate total number of pixels excluding channels
        patterns_combined = data.reshape(-1, num_pixels * pattern_shape[-1])

        # Ensure that the patterns are of type np.float32
        patterns_combined = patterns_combined.astype(np.float32)

        # Create the Hopfield network with neurolab
        self.net = nl.net.newhop(patterns_combined)

    def reconstruct(self, patterns_test):
        """
        Use the trained Hopfield network to reconstruct the patterns.
        :param patterns_test: np.ndarray, patterns to reconstruct.
        :return: np.ndarray, reconstructed patterns.
        """
        # Reshape the input patterns to be two-dimensional
        pattern_shape = patterns_test.shape
        num_pixels = np.prod(pattern_shape[:-1])  # Calculate total number of pixels excluding channels
        patterns_flat = patterns_test.reshape(-1, num_pixels * pattern_shape[-1])

        # Ensure patterns are of type np.float32
        patterns_flat = patterns_flat.astype(np.float32)

        # Simulate the network with the test patterns
        output = self.net.sim(patterns_flat)

        # Reshape the output to match the shape of the input patterns
        output = output.reshape(pattern_shape)

        return output



img_1_path = r"C:\Users\cvill\iCloudDrive\workspace\DeepL\Interface\img_1.png"
img_1 = Image.open(img_1_path) #.convert("RGB")
img_1_array = np.array(img_1) / 255.0 * 2 - 1

img_1_noisy_path = r"C:\Users\cvill\iCloudDrive\workspace\DeepL\Interface\img_1_noisy.png"
img_1_noisy = Image.open(img_1_noisy_path)
img_1_noisy_array = np.array(img_1_noisy) / 255.0 * 2 - 1

print(img_1_array[0].shape)
print(img_1_noisy_array[0].shape)
'''
model = HopfieldNetwork(1000)
model.train([img_1_array])
recontructed_image = model.reconstruct(img_1_array)
'''
hopfield_net = HopfieldNetworkNeurolab()
hopfield_net.train(img_1_array)
recontructed_image = hopfield_net.reconstruct(img_1_noisy_array)
print(recontructed_image[0].shape)
print(recontructed_image)

original_img = [((img_1_array + 1) / 2 * 255).astype(np.uint8)]
noisy_img = [((img_1_noisy_array + 1) / 2 * 255).astype(np.uint8)]
rec_img = [((recontructed_image + 1) / 2 * 255).astype(np.uint8)]

print(rec_img)

plot_images_hop(original_img, noisy_img, rec_img)


