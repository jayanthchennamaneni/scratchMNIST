import numpy as np
from activations import ReLU, softmax, ReLU_deriv

def init_params(input_size, hidden_size, output_size):
    """
    Initialize the parameters of the neural network.

    Parameters:
    input_size : int
        Number of input features.
    hidden_size : int
        Number of neurons in the hidden layer.
    output_size : int
        Number of output classes.

    Returns:
    W1 : numpy.ndarray
        Weight matrix for the input layer to the hidden layer.
    b1 : numpy.ndarray
        Bias vector for the hidden layer.
    W2 : numpy.ndarray
        Weight matrix for the hidden layer to the output layer.
    b2 : numpy.ndarray
        Bias vector for the output layer.
    """
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

def forward_propagation(W1, b1, W2, b2, X):
    """
    Perform forward propagation through the neural network.

    Parameters:
    W1 : numpy.ndarray
        Weight matrix for the input layer to the hidden layer.
    b1 : numpy.ndarray
        Bias vector for the hidden layer.
    W2 : numpy.ndarray
        Weight matrix for the hidden layer to the output layer.
    b2 : numpy.ndarray
        Bias vector for the output layer.
    X : numpy.ndarray
        Input data matrix.

    Returns:
    Z1 : numpy.ndarray
        Output of the linear transformation in the hidden layer.
    A1 : numpy.ndarray
        Output of the activation function in the hidden layer.
    Z2 : numpy.ndarray
        Output of the linear transformation in the output layer.
    A2 : numpy.ndarray
        Output of the softmax activation function in the output layer.
    """
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    """
    Perform backward propagation to compute gradients.

    Parameters:
    Z1 : numpy.ndarray
        Output of the linear transformation in the hidden layer.
    A1 : numpy.ndarray
        Output of the activation function in the hidden layer.
    Z2 : numpy.ndarray
        Output of the linear transformation in the output layer.
    A2 : numpy.ndarray
        Output of the softmax activation function in the output layer.
    W1 : numpy.ndarray
        Weight matrix for the input layer to the hidden layer.
    W2 : numpy.ndarray
        Weight matrix for the hidden layer to the output layer.
    X : numpy.ndarray
        Input data matrix.
    Y : numpy.ndarray
        True labels.

    Returns:
    dW1 : numpy.ndarray
        Gradient of the loss with respect to W1.
    db1 : numpy.ndarray
        Gradient of the loss with respect to b1.
    dW2 : numpy.ndarray
        Gradient of the loss with respect to W2.
    db2 : numpy.ndarray
        Gradient of the loss with respect to b2.
    """
    m = X.shape[1]
    one_hot_Y = np.eye(10)[Y].T
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = (W2.T @ dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Update the parameters of the neural network using gradient descent.

    Parameters:
    W1 : numpy.ndarray
        Weight matrix for the input layer to the hidden layer.
    b1 : numpy.ndarray
        Bias vector for the hidden layer.
    W2 : numpy.ndarray
        Weight matrix for the hidden layer to the output layer.
    b2 : numpy.ndarray
        Bias vector for the output layer.
    dW1 : numpy.ndarray
        Gradient of the loss with respect to W1.
    db1 : numpy.ndarray
        Gradient of the loss with respect to b1.
    dW2 : numpy.ndarray
        Gradient of the loss with respect to W2.
    db2 : numpy.ndarray
        Gradient of the loss with respect to b2.
    alpha : float
        Learning rate.

    Returns:
    W1 : numpy.ndarray
        Updated weight matrix for the input layer to the hidden layer.
    b1 : numpy.ndarray
        Updated bias vector for the hidden layer.
    W2 : numpy.ndarray
        Updated weight matrix for the hidden layer to the output layer.
    b2 : numpy.ndarray
        Updated bias vector for the output layer.
    """
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_accuracy(predictions, Y):
    """
    Compute the accuracy of predictions.

    Parameters:
    predictions : numpy.ndarray
        Predicted labels.
    Y : numpy.ndarray
        True labels.

    Returns:
    float
        Accuracy of the predictions.
    """
    return np.mean(predictions == Y)

def gradient_descent(X, Y, alpha, iterations):
    """
    Train the neural network using gradient descent.

    Parameters:
    X : numpy.ndarray
        Input data matrix.
    Y : numpy.ndarray
        True labels.
    alpha : float
        Learning rate.
    iterations : int
        Number of training iterations.

    Returns:
    W1 : numpy.ndarray
        Updated weight matrix for the input layer to the hidden layer.
    b1 : numpy.ndarray
        Updated bias vector for the hidden layer.
    W2 : numpy.ndarray
        Updated weight matrix for the hidden layer to the output layer.
    b2 : numpy.ndarray
        Updated bias vector for the output layer.
    """
    W1, b1, W2, b2 = init_params(input_size=X.shape[0], hidden_size=10, output_size=10)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = np.argmax(A2, axis=0)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration: {i}, Accuracy: {accuracy}")
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    """
    Make predictions using the trained neural network.

    Parameters:
    X : numpy.ndarray
        Input data matrix.
    W1 : numpy.ndarray
        Weight matrix for the input layer to the hidden layer.
    b1 : numpy.ndarray
        Bias vector for the hidden layer.
    W2 : numpy.ndarray
        Weight matrix for the hidden layer to the output layer.
    b2 : numpy.ndarray
        Bias vector for the output layer.

    Returns:
    numpy.ndarray
        Predicted labels.
    """
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = np.argmax(A2, axis=0)
    return predictions