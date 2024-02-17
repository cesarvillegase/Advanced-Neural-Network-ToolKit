import numpy as np
import matplotlib.pyplot as plt

def weights_matrix(patterns):
    # P - Number of patterns,
    # n - length of every pattern after the flatten
    p, n = patterns.shape
    
    weights = np.zeros((n,n))
    
    # For every pattern in the list of patterns
    for pt in patterns:
        # An automatic list of arrays
        pt = pt.reshape(-1,1)
        weights = weights + np.dot(pt, pt.T)
        
    np.fill_diagonal(weights, 0)
    weights = weights / n 
    
    return weights

def neuron(weights, inputs, epoch_max=1000):
    for i in range(epoch_max):
        # The inputs are the X, Y and the bias
        # Compute the output in the function of the input
        inputs = np.sign(np.dot(weights, inputs))
        
    return inputs


def plot_patterns(patterns, titles, pattern_shape):
    '''
    Plots subplots of patterns with titles.
    '''
    num_patterns = len(patterns)
    num_rows, num_columns = pattern_shape
    
    # Create the figure
    plt.figure(figsize=(7, 7))
    
    # Loop to create the subplots
    for i in range(num_patterns):
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(patterns[i], cmap='gray')
        plt.title(titles[i])
        
    # Adjust the spacing between the suplots
    plt.tight_layout()
    # Show the figure
    plt.show()

# List of patterns 
pattern_1 = np.array([
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1]
])

pattern_2 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1]
])

pattern_3 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, -1, -1, -1]
])

pattern_4 = np.array([
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1]
])

pattern_5 = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1]
])

pattern_6 = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, 1, 1, 1]
])

   # Images with noise

pattern_1_wn = np.array([
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, 1, -1, 1, -1, 1, 1, 1],
    [1, -1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, -1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, -1],
    [1, 1, 1, -1, 1, -1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, 1, 1, -1],
    [1, 1, 1, -1, -1, -1, 1, -1, 1]
])

pattern_2_wn = np.array([
    [-1, 1, 1, 1, 1, 1, 1, 1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, 1, 1, 1, 1, 1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, -1, 1, -1, -1, -1, -1]
])

pattern_3_wn = np.array([
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, 1, 1, 1, 1, 1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, -1, -1, -1],
    [1, 1, 1, 1, 1, -1, -1, -1, -1],
    [1, 1, 1, 1, -1, -1, -1, -1, -1]
])

# Example 
patterns = [pattern_1, pattern_2, pattern_3, pattern_4, pattern_5, pattern_6, 
            pattern_1_wn, pattern_2_wn, pattern_3_wn]

# Títulos de los subplots
titles = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Pattern 5", "Pattern 6",
          "Pattern 1 with noise", "Pattern 2 with noise", "Pattern 3 with noise"]

pattern_shape = (3, 3)
    
plot_patterns(patterns, titles, pattern_shape)
    