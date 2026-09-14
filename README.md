<h1>Basic Neural Network from Scratch (NumPy)</h1>

This project builds a simple feed‑forward neural network from scratch using only NumPy. It’s designed to help me understand how neural networks work under the hood — including forward propagation, backpropagation, and gradient descent.

**What the network does**

Learns the XOR function

Uses one hidden layer

Uses the sigmoid activation function

Trains using mean squared error (MSE)

Updates weights with gradient descent

**Why I built this**

I wanted to learn the fundamentals of neural networks without relying on frameworks like TensorFlow or PyTorch. Building it manually helped me understand how data flows through a network and how weights/biases get updated during training.

**Project Structure**

sigmoid() and sigmoid_derivative()

mean_squared_error()

BasicNeuralNetwork class

    -initialization
    
    -forward pass
    
    -backward pass
    
    -training loop

XOR dataset and testing

**Training Results**

After training, the model correctly predicts the XOR outputs (values close to 0 or 1).
