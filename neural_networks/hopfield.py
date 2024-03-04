import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Hopfield:
    def __init__(self, data, epoch_max):
        self.weights_r = None
        self.weights_g = None
        self.weights_b = None
        self.data = data
        self.epoch_max = epoch_max

    @staticmethod
    def weights_matrix(patterns):
        p, n = patterns.shape
        weights = np.zeros((n, n))

        for pt in patterns:
            pt = pt.reshape(-1, 1)
            weights = weights + np.dot(pt, pt.T)

        np.fill_diagonal(weights, 0)
        weights = weights / n

        return weights

    @staticmethod
    def neuron(weights, data, epoch_max):
        for i in range(epoch_max):
            data = np.sign(np.dot(weights, data))

        return data

    def train(self):
        # Split the inputs into color channels
        patterns_r = np.array([inp[:, :, 0].flatten() for inp in self.data])
        patterns_g = np.array([inp[:, :, 1].flatten() for inp in self.data])
        patterns_b = np.array([inp[:, :, 2].flatten() for inp in self.data])

        # Calculate the weight matrices for each color channel
        self.weights_r = self.weights_matrix(patterns_r)
        self.weights_g = self.weights_matrix(patterns_g)
        self.weights_b = self.weights_matrix(patterns_b)

    def reconstruct(self, noisy_data):
        rec_images = []
        for noisy_img in noisy_data:
            rec_img_r = self.neuron(self.weights_r, noisy_img[:, :, 0].flatten(), self.epoch_max).reshape(16, 16)
            rec_img_g = self.neuron(self.weights_g, noisy_img[:, :, 1].flatten(), self.epoch_max).reshape(16, 16)
            rec_img_b = self.neuron(self.weights_b, noisy_img[:, :, 2].flatten(), self.epoch_max).reshape(16, 16)
            rec_img = np.dstack((rec_img_r, rec_img_g, rec_img_b))
            rec_images.append(rec_img)
        return rec_images


img_1_path = "/Users/cesarve/Documents/GitHub/Advanced-Neural-Network-ToolKit/neural_networks/images/img_1.png"
img_2_path = "/Users/cesarve/Documents/GitHub/Advanced-Neural-Network-ToolKit/neural_networks/images/img_2.png"
img_3_path = "/Users/cesarve/Documents/GitHub/Advanced-Neural-Network-ToolKit/neural_networks/images/img_3.png"

img_1_wn_path = "/Users/cesarve/Documents/GitHub/Advanced-Neural-Network-ToolKit/neural_networks/images/img_1_noisy.png"
img_2_wn_path = "/Users/cesarve/Documents/GitHub/Advanced-Neural-Network-ToolKit/neural_networks/images/img_2_noisy.png"
img_3_wn_path = "/Users/cesarve/Documents/GitHub/Advanced-Neural-Network-ToolKit/neural_networks/images/img_3_noisy.png"

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

inputs = [img_1_array, img_2_array, img_3_array]
noisy_inputs = [img_1_wn_array, img_2_wn_array, img_3_wn_array]

'''
# Create HopfieldNetwork instance
hopfield_net = Hopfield(inputs, epoch_max=1000)

# Train the network
hopfield_net.train()

# Reconstruct images
reconstructed_images = hopfield_net.reconstruct(noisy_inputs)


def plot_images(original_images, noisy_images):
    num_original = int(input("Enter the number of original images to plot: "))
    num_noisy = int(input("Enter the number of noisy images to plot: "))

    # Plotting the images
    plt.figure(figsize=(7, 7))
    for i in range(num_original):
        plt.subplot(num_original, 3, i * 3 + 1)
        plt.imshow(original_images[i], cmap='gray')
        plt.title(f"Original Image {i + 1}")

    for i in range(num_noisy):
        plt.subplot(num_noisy, 3, i * 3 + 2)
        plt.imshow(noisy_images[i], cmap='gray')
        plt.title(f"Noisy Image {i + 1}")

    for i in range(min(num_original, num_noisy)):
        rec_img = reconstructed_images[i]
        plt.subplot(min(num_original, num_noisy), 3, i * 3 + 3)
        plt.imshow(rec_img, cmap='gray')
        plt.title(f"Reconstructed Image {i + 1}")

    plt.tight_layout()
    plt.show()


# Example usage:
plot_images([img_1_original, img_2_original, img_3_original], [img_1_wn_original, img_2_wn_original, img_3_wn_original])
'''