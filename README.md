# Implementing Neural Network with 2 Hidden Layers From Scratch

### Files Included:
* **NeuralNet.py**:
    * Implements the Neural Network with 2 hidden layers.
	* Used labelEncoder, min_max_scalar, SimpleImputer, MinMaxScaler for Pre-processing of the data.
	* labelEncoder: converts categorical to numerical attributes
	* SimpleImputer: Handles "NULL" values
	* MinMaxScaler: Standardizes and scales the attributes

## Requirements
* Python 3
* Command Line Interface
* Pandas * https://pandas.pydata.org/*
* Scikit Learn
* Numpy
* sklearn

## Steps to run the program
* Open the CLI and run the command, *python NeuralNet.py url_link max_iterations learning_rate activation_function*
* Replace the *url_link* in the above command with the url to the dataset
* Replace the *max_iterations* in the above command with the maximum number of iterations
* Replace the *learning_rate* in the above command with desired learning rate for the algorithm
* Replace the *activation_function* in the above command with the desired activation function( tanh, sigmoid, ReLu)

### Example Run commands
* **Car Dataset**: _python3 NeuralNet.py "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data" 1000 0.007 sigmoid
* **Iris Dataset**: _python3 NeuralNet.py "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data" 10000 0.05 sigmoid
* **Hepatitis Dataset**: _python NeuralNet.py https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data" 1000 0.005 sigmoid
* **Wine Dataset**: _python3 NeuralNet.py "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data" 10000 0.005 sigmoid