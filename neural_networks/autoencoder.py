import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder:
    def __init__(self):
        self.input_neurons = None
        self.hidden_neurons = None
        self.output_neurons = None
        self.epoch = 0
        self.loss = []

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_dev(x):
        return x * (1 - x)

    def train(self, data, alpha, momentum, epoch_max):
        inputs = data / 255.0
        inputs = inputs.flatten()
        expected_output = inputs

        self.input_neurons = inputs.shape[0]
        self.hidden_neurons = 1
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

            if self.epoch % 100 == 0:
                print(f"Epoch: {self.epoch} Error: {mse}")

            self.epoch += 1

        latent_space: float = self.sigmoid(np.dot(inputs, weights_input))
        decoded_inputs = self.sigmoid(np.dot(latent_space, weights_output))
        decoded_inputs = (decoded_inputs * 255).astype(int)

        return self.loss, latent_space, decoded_inputs


# TESTING
'''
X_train_ac = np.array([[123, 32, 24], [72, 204, 52], [145, 56, 91]])
alpha_ac = 0.9  # alpha = 1
momentum_ac = 0.4  # Momentum = 0.4
epoch_max_ac = 10000

model = AutoEncoder()
loss_ac, latent_space, decoded_inputs_ac = model.train(X_train_ac, alpha_ac, momentum_ac, epoch_max_ac)

print(f"\nOriginal Inputs: \n{X_train_ac.flatten()}")
print(f"\nLatent Space: \n{latent_space}")
print(f"\nReconstructed Inputs: \n{decoded_inputs_ac}")


def plot_loss(loss_accuracy):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        plt.plot(range(1, len(loss_accuracy) + 1), loss_accuracy, color='orange', label='MSE')
        plt.title("Training for the Auto encoder")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        print(loss_accuracy[-1])
'''