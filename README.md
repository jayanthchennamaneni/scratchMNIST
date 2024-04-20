## scratch MNIST

This project implements a simple neural network using only NumPy and Pandas to classify handwritten digits from the MNIST dataset. The model architecture consists of an input layer, a hidden layer with ReLU activation, and an output layer with softmax activation.

## Project Description

- **Libraries Used**: NumPy, Pandas
- **Model Architecture**: Single-hidden-layer neural network(Multilayer perceptron)
- **Activation Functions**: ReLU (hidden layer), Softmax (output layer)
- **Training Method**: Gradient Descent
- **Iterations**: 500
- **Accuracy**: []
- **Time**: []

## Usage
1. Clone the repository: https://github.com/jayanthchennamaneni/scratchMNIST.git
2. cd scratchMNIST
3. Install dependencies: pip install -r requirements.txt and the data can be accessed at https://www.kaggle.com/competitions/digit-recognizer
4. Run training: python src/train.py

## File
- `src/`: Source code directory.
- `activations.py`: Script for activations functions
- `model.py`: Implementation of the neural network model.
- `train.py`: Script for training the model.
- `requirements.txt`: List of Python dependencies.
