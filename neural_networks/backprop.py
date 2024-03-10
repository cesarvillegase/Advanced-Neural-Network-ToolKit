import numpy as np
import matplotlib.pyplot as plt

# import tensorflow as tf
from tensorflow import keras


class Backpropagation:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        # Configuration of the neural network
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # Init of the weights with a bias 
        self.weights_input = 2 * np.random.random((self.input_neurons, self.hidden_neurons)) - 1
        self.weights_output = 2 * np.random.random((self.hidden_neurons + 1, self.output_neurons)) - 1

        self.epoch = 0
        # Mean Square Error
        self.mse = float(2)
        # The previous Mean Square Error
        self.prev_mse = float(0)
        # Then we store the loss in a list
        self.loss = []

    # Sigmoid function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid derivative function
    @staticmethod
    def sigmoid_dev(x):
        return x * (1 - x)

    # =============TRAIN PHASE=============
    def train(self, inputs, desired_output, alpha):
        while abs(self.mse - self.prev_mse) > 0.00001:
            self.prev_mse = self.mse

            # =============FORWARD PROPAGATION=============
            # Hidden Layer
            hidden_input = np.dot(inputs, self.weights_input)  # Z value
            hidden_output = self.sigmoid(hidden_input)
            # Addition of the bias to the output
            hidden_output = np.insert(hidden_output, 0, 1, axis=1)

            # Output Layer
            output_input = np.dot(hidden_output, self.weights_output)
            obtained_output = self.sigmoid(output_input)

            # =============BACK PROPAGATION=============
            # Compute the error of the output layer
            output_error = desired_output - obtained_output
            self.mse = np.mean(output_error ** 2)
            self.loss.append(self.mse)
            gradient = output_error * self.sigmoid_dev(obtained_output)

            # Compute the error for the hidden layer
            hidden_error = gradient.dot(self.weights_output[1:, :].T)
            hidden_delta = hidden_error * self.sigmoid_dev(hidden_output[:, 1:])

            # Update the weights 
            self.weights_output = self.weights_output + (alpha * hidden_output.T.dot(gradient))
            self.weights_input = self.weights_input + (alpha * inputs.T.dot(hidden_delta))

            # Print the error every 500 iterations
            if self.epoch % 500 == 0:
                print(f"Epoch: {self.epoch} Error: {self.mse} Accuracy: {1 - abs(np.mean(self.mse))}")

            self.epoch += 1

        print("Training completed.")

    # =============TEST PHASE=============
    def test(self, inputs, desired_output):
        global output_activation
        results = ""
        # Test for the neural network trained
        for i in range(inputs.shape[0]):
            hidden_activation = self.sigmoid(np.dot(inputs[i], self.weights_input))
            # Addition of the bias to the hidden layer
            hidden_activation_with_bias = np.insert(hidden_activation, 0, 1)
            output_activation = self.sigmoid(np.dot(hidden_activation_with_bias, self.weights_output))

            print("Input:", inputs[i, 1:], "| Expected:", desired_output[i],
                  "| Predicted:", np.round(output_activation))

            # Accumulate results
            results += f"Input: {inputs[i, 1:]} | Expected: {desired_output[i]} | Predicted: {np.round(output_activation)}\n"

        return self.loss, output_activation, results


# Function to plot the loss
def plot_loss(loss_value):
    plt.plot(range(1, len(loss_value) + 1), loss_value, color='blue', label='Mean Square Error')
    plt.title("Training loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


class BackpropagationKeras:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.history = None

        # Initialize the Keras model
        self.model = keras.Sequential([
            keras.layers.Dense(hidden_neurons, input_shape=(input_neurons,), activation='sigmoid'),
            keras.layers.Dense(output_neurons, activation='sigmoid')
        ])

    def train(self, data, label, learning_rate):
        # Compile and train the Keras model
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        # train the model
        self.history = self.model.fit(data, label, epochs=200, verbose=0)

    def test(self, data, labels):
        # Evaluate the model
        test_loss, test_acc = self.model.evaluate(data, labels)
        print(f'\nAccuracy on the test set: {test_acc}')

        results = []

        # Make predictions
        predictions = self.model.predict(data)
        print("\nPredictions:")
        for i in range(len(predictions)):
            print(
                f"Input: {data[i, 1:]} | Expected: {labels[i]},| Predicted: {np.round(predictions[i][0])}")

            results.append(f"Input: {data[i, 1:]} | Expected: {labels[i]} | Predicted: {np.round(predictions[i][0])}")

        return self.history, test_acc, results


'''
# TESTING

# input [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
# Desired Output Data [0], [1], [1], [0]
# Learning rate 0.7

'''
