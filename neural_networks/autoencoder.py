import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, losses, Model
from sklearn.metrics import accuracy_score


class AutoEncoder:
    def __init__(self):
        self.input_neurons = None
        self.hidden_neurons = None
        self.output_neurons = None
        self.epoch = 0
        self.loss = []

    @staticmethod
    def sigmoid(x):
        return float(1) / (float(1) + np.exp(-x))

    @staticmethod
    def sigmoid_dev(x):
        return x * (float(1) - x)

    def train(self, data, alpha, momentum, epoch_max):
        inputs = data / 255.0
        inputs = inputs.flatten()
        expected_output = inputs

        self.input_neurons = inputs.shape[0]
        self.hidden_neurons = 3
        self.output_neurons = self.input_neurons

        weights_input = 2 * np.random.random((self.input_neurons, self.hidden_neurons)) - 1
        weights_output = 2 * np.random.random((self.hidden_neurons, self.output_neurons)) - 1

        w_old_input = np.zeros_like(weights_input)
        w_new_input = np.zeros_like(weights_input)

        w_old_output = np.zeros_like(weights_output)
        w_new_output = np.zeros_like(weights_output)

        mse = float(2)
        prev_mse = float(0)

        while (self.epoch < epoch_max) and abs(mse - prev_mse) > 0.00001:
            prev_mse = mse

            hidden_lyr_input = np.dot(inputs, weights_input)
            hidden_lyr_output = self.sigmoid(hidden_lyr_input)

            output_lyr_input = np.dot(hidden_lyr_output, weights_output)
            output = self.sigmoid(output_lyr_input)

            output_error = expected_output - output
            mse = np.mean(output_error ** 2)
            self.loss.append(mse)
            gradient = output_error * self.sigmoid_dev(output)

            output_lyr_delta = gradient

            hidden_lyr_error = output_lyr_delta.dot(weights_output.T)
            hidden_lyr_delta = hidden_lyr_error * self.sigmoid_dev(hidden_lyr_output)

            w_new_output = weights_output + alpha * np.outer(hidden_lyr_output, output_lyr_delta) + momentum * (
                    weights_output - w_old_output)
            w_old_output = weights_output.copy()
            weights_output = w_new_output

            w_new_input = weights_input + alpha * np.outer(inputs, hidden_lyr_delta) + momentum * (
                    weights_input - w_old_input)
            w_old_input = weights_input.copy()
            weights_input = w_new_input

            if self.epoch % 200 == 0:
                print(f"Epoch: {self.epoch} Error: {mse}")

            self.epoch += 1

        latent_space: float = self.sigmoid(np.dot(inputs, weights_input))
        decoded_inputs = self.sigmoid(np.dot(latent_space, weights_output))
        decoded_inputs = (decoded_inputs * 255).astype(int)

        return self.loss, latent_space, decoded_inputs


def plot_loss_ac(loss_values):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        plt.plot(range(1, len(loss_values) + 1), loss_values, color='orange', label='MSE')
        plt.title("Training for the Auto encoder")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        print(loss_values[-1])


def plot_images_ac(original_image, reconstructed_img):
    """Plot the original, noisy, and reconstructed images."""
    plt.figure(figsize=(8, 4))

    # Reshape and convert the original image to a NumPy array
    original_image_array = np.array(original_image[0])
    original_image_array = original_image_array.astype(np.uint8)  # + 1) / 2 * 255

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_array)
    plt.title('Original Image')
    plt.axis('off')  # Turn off axes

    # Plot the reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img)
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


class SimpleAutoencoder(Model):
    def __init__(self, latent_dimensions, data_shape):
        super(SimpleAutoencoder, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.data_shape = data_shape
        self.flat_dim = np.prod(data_shape)

        # Encoder architecture using a Sequential model
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dimensions, activation='sigmoid'),
        ])

        # Decoder architecture using another Sequential model
        self.decoder = tf.keras.Sequential([
            layers.Dense(self.flat_dim, activation='sigmoid'),  # Use the precalculated flat dimension
            layers.Reshape(data_shape)
        ])

    # Forward pass method defining the encoding and decoding steps
    def call(self, input_data):
        encoded_data = self.encoder(input_data)
        decoded_data = self.decoder(encoded_data)
        return decoded_data

    def train(self, data, epochs=1200):
        self.compile(optimizer='adam', loss=losses.MeanSquaredError())
        training_data = (data / 255.0).astype(np.float32)  # Normalize and cast to float32
        training_data = tf.reshape(training_data, [-1, *self.data_shape])  # Reshape data
        history = self.fit(training_data, training_data,
                           epochs=epochs,
                           shuffle=True,
                           validation_data=(training_data, training_data))

        # Compute latent space
        encoded_data = self.encoder(training_data)
        # Decode the encoded data
        decoded_data = self.decoder(encoded_data)
        decoded_data = np.round(np.clip(decoded_data * 255, 0, 255)).astype(np.uint8)  # Clip and convert to uint8
        # Calculate loss
        loss = history.history['loss']
        return loss, encoded_data, decoded_data


'''
# TESTING

X_train_ac = np.array([[123, 32, 24], [72, 204, 52], [145, 56, 91]])
alpha_ac = 0.9  # alpha = 1
momentum_ac = 0.4  # Momentum = 0.4
epoch_max_ac = 1200

model = AutoEncoder()
loss_ac, latent_space, decoded_inputs_ac = model.train(X_train_ac, alpha_ac, momentum_ac, epoch_max_ac)

print(f"\nOriginal Inputs: \n{X_train_ac.flatten()}")
print(f"\nLatent Space: \n{latent_space}")
print(f"\nReconstructed Inputs: \n{decoded_inputs_ac}")


input_data_shape = X_train_ac.shape[1:]
model_simpleac = SimpleAutoencoder(latent_dimensions=1, data_shape=input_data_shape)
loss_ac, latent_space, decoded_inputs_ac = model_simpleac.train(X_train_ac, epoch_max_ac)

print(f"\nOriginal Inputs: \n{X_train_ac}")
print(f"\nLatent Space: \n{latent_space}")
print(f"\nReconstructed Inputs: \n{decoded_inputs_ac}")

'''
