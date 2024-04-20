import numpy as np

def ReLU(x):
    """
    Compute the Rectified Linear Unit (ReLU) activation function element-wise.

    ReLU(x) returns the element-wise maximum of x and 0.

    Parameters:
    x : numpy.ndarray
        Input array.

    Returns:
    numpy.ndarray
        Element-wise maximum of x and 0.
    """
    return np.maximum(x, 0)

def softmax(x):
    """
    Compute the softmax activation function for the input array.

    softmax(x) computes the softmax function for each column in the input array x.

    Parameters:
    x : numpy.ndarray
        Input array.

    Returns:
    numpy.ndarray
        Softmax values for each column in the input array.
    """
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def ReLU_deriv(x):
    """
    Compute the derivative of the ReLU activation function.

    ReLU_deriv(x) returns a boolean array indicating which elements of x are greater than 0.

    Parameters:
    x : numpy.ndarray
        Input array.

    Returns:
    numpy.ndarray
        Boolean array indicating which elements are greater than 0.
    """
    return x > 0

