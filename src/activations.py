import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def ReLU_deriv(x):
    return x > 0
