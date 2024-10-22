import numpy as np 

# Input data (features and target)
input_data = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) 
target_output = np.array(([92], [86], [89]), dtype=float)

# Normalize input data and target output
input_data = input_data / np.amax(input_data, axis=0)  
target_output = target_output / 100

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (used for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Training parameters
epochs = 1000        # Number of iterations
learning_rate = 0.2  # Learning rate

# Network architecture
num_input_neurons = 2      # Number of input neurons
num_hidden_neurons = 3     # Number of hidden neurons
num_output_neurons = 1     # Number of output neurons

# Initialize weights and biases for the hidden layer and output layer
weights_hidden = np.random.uniform(size=(num_input_neurons, num_hidden_neurons)) 
bias_hidden = np.random.uniform(size=(1, num_hidden_neurons))  

weights_output = np.random.uniform(size=(num_hidden_neurons, num_output_neurons)) 
bias_output = np.random.uniform(size=(1, num_output_neurons))

# Training loop (forward and backpropagation)
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(input_data, weights_hidden) + bias_hidden  # Input to hidden layer
    hidden_layer_activation = sigmoid(hidden_layer_input)  # Output of hidden layer
    
    output_layer_input = np.dot(hidden_layer_activation, weights_output) + bias_output  # Input to output layer
    predicted_output = sigmoid(output_layer_input)  # Final output (prediction)
    
    # Backpropagation (Error calculation and gradient update)
    error_output_layer = target_output - predicted_output  # Error at output layer
    output_gradient = sigmoid_derivative(predicted_output)  # Gradient of the output
    delta_output = error_output_layer * output_gradient  # Delta for output layer
    
    # Error propagated back to hidden layer
    error_hidden_layer = delta_output.dot(weights_output.T)  
    hidden_layer_gradient = sigmoid_derivative(hidden_layer_activation)  
    delta_hidden = error_hidden_layer * hidden_layer_gradient  # Delta for hidden layer
    
    # Update weights and biases using the gradients
    weights_output += hidden_layer_activation.T.dot(delta_output) * learning_rate  
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    
    weights_hidden += input_data.T.dot(delta_hidden) * learning_rate  
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

# Output the results
print("Normalized Input Data: \n" + str(input_data))
print("Actual Output (Target): \n" + str(target_output))
print("Predicted Output: \n" , predicted_output)
