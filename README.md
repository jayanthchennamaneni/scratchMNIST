## Scratch MNIST

This project implements a simple neural network using only NumPy and Pandas to classify handwritten digits from the MNIST dataset(the classic dataset of handwritten images). The model architecture consists of an input layer, a hidden layer with ReLU activation, and an output layer with softmax activation.

## Project Description

- **Libraries Used**: NumPy, Pandas
- **Model Architecture**: Single-hidden-layer neural network(Multilayer perceptron)
- **Activation Functions**: ReLU (hidden layer), Softmax (output layer)
- **Optimizer**: Gradient Descent(Learning rate = 0.1)
- **Epochs**: 500
- **Training Accuracy**: [91.08 %]
- **Test Accuracy**: [91.2 %]
- **Time**: [1m:07.87s]

## Usage
1. Clone the repository: https://github.com/jayanthchennamaneni/scratchMNIST.git
2. cd scratchMNIST
3. Install dependencies: `pip install -r requirements.txt` and the data can be accessed at https://www.kaggle.com/competitions/digit-recognizer
4. Run training: python src/train.py

## File

````
├── src/                  # Source code directory.
├──── activations.py      # Script for activations functions
├──── model.py            # Implementation of the neural network model
├──── train.py            # Script for training the model
└─ requirements.txt       # List of Python dependencies
````




