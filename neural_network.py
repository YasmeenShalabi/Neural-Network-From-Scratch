#Basic Neural Network from Scratch using only numpy to demonstrate how a NN works under the hood
#Implemented using a simple feed-forward network with one hidden layer, and use sigmoid activation function and mean squared error as the loss function

import numpy as np #used for matrix operation and mathematical functions

#Sigmoid activation function and its derivative
#Sigmoid helps a neural network learn non‑linear patterns by squashing numbers into the range 0–1 and making neurons behave like smooth switches
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#sigmoid derivative tells NN how much a neuron should adjust its weights during learning->used during backpropagation, where network learns from its mistakes
def sigmoid_derivative(x):
    return x * (1-x)

#Mean Squared Error(MSE) Loss Function to measure model performance-> tells NN how wrong its predictions are so it knows how to adjust its weights during training
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

#Basic Neural Network Class
class BasicNeuralNetwork:
    #Initialize NN with random weights and biases
    
    #Weights tell the NN how strongly each input should influence a neuron’s output
    #Bias is an extra number the neuron adds so it doesn’t always start at zero; lets the neuron shift its starting point so it doesn’t depend only on the inputs
    #Hidden layer-> where NN learns patterns that are not obvious from the raw input
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) #weights btw input layer and hidden layer
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) #weights btw hidden layer and output layer
        self.bias_hidden = np.random.randn(1, hidden_size) #bias for hidden layer
        self.bias_output = np.random.randn(1, output_size) #bias for output layer

    #Forward Pass (input is passed through the network to produce an output)
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output
    
    #Backward pass(error is propogated back through network; and weights & biases updated using gradient descent
    def backward(self, X, y, output, learning_rate):
        output_error = y - output #difference btw actual output(y) and predicted output
        output_delta = output_error * sigmoid_derivative(output) #adjustments for output layer based on error and derivative of sigmoid function
        hidden_error= np.dot(output_delta, self.weights_hidden_output.T) #error for hidden layer, propogated from output layer. Takes the error from the output layer and sends it backward through the output weights (flipped), so each hidden neuron knows how much error it caused
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output) #adjustments for hidden layer
        
        self.weights_hidden_output +=np.dot(self.hidden_output.T, output_delta) * learning_rate #update self weights hidden output
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate #update self bias output
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        
    #Train the NN over multiple epochs to minimize loss
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            #Forward Pass
            output = self.forward(X)
            
            #Backward Pass to update weights and biases
            self.backward(X, y, output, learning_rate)
            
            #for every 100 epochs print loss
            if epoch % 100 == 0:
                loss = mean_squared_error(y, output)
                print(f"Epoch: {epoch}, Loss: {loss}")
        
#XOR dataset to test the NN
X = np.array([[0,0],[0,1], [1,0], [1,1]])
y = np.array([[0],[1],[1],[0]]) #expected output

#create NN instance
nn = BasicNeuralNetwork(input_size=2, hidden_size=2, output_size=1) #2 input neurons, 2 hidden neurons, 1 output neuron

nn.train(X, y, epochs=10000, learning_rate=0.1)

print("\nTest the trained neural network:")
for i in range(len(X)):
    print(f"input: {X[i]}, Predicted Output: {nn.forward(X[i])}, Actual Output: {y[i]}")
