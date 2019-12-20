#####################################################################################################################
#   Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################

from sklearn import preprocessing
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import model_selection
import sys

class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train)
        #print("raw train input: " + str(raw_input))
        # TODO: Remember to implement the preprocess method
        train_dataset, test_dataset = model_selection.train_test_split(raw_input, test_size=0.2)
        train_dataset = self.preprocess(train_dataset)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random  weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
        self.test_dataset = test_dataset

    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        ## Inserted By Ekansh and Dishant
        elif activation  == "tanh":
            self.__tanh(self,x)

        elif activation == "relu":
            self.__relu(self,x)
    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)

        ## Inserted By Ekansh and Dishant
        elif activation  == "tanh":
            self.__tanh_derivative(self,x)

        elif activation == "relu":
            self.__relu_derivative(self,x)


    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    ## Inserted By Ekansh and Dishant
    def preprocess(self, X):
        '''
        input = X.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledInput = scaler.fit_transform(input)
        return rescaledInput
        '''
        labelEncoder = preprocessing.LabelEncoder()
        X=X.apply(labelEncoder.fit_transform)
        simpleInputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        simpleInputer.fit(X)
        '''
        for i in X.columns:
            X[i]= X[i].fillna(value = X[i].mean())
        '''
        #Normalization
        X = X.values #returns a numpy array
        minMaxScaler = MinMaxScaler(feature_range=(0, 1))
        rescaledInput = minMaxScaler.fit_transform(X)
        rescaledInput = pd.DataFrame(rescaledInput)
        rescaledInput= rescaledInput.drop_duplicates()
        #print("rescaled Input: " + str(rescaledInput))
        return rescaledInput
    # Below is the training function

    def train(self, activation, max_iterations, learning_rate):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            ## Inserted By Ekansh and Dishant
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)
            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )

        if activation == "sigmoid":
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        ## Inserted By Ekansh and Dishant
        elif activation == "tanh":
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        ## Inserted By Ekansh and Dishant
        elif activation == "relu":
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        output = out
        return output



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions
    ## Inserted By Ekansh and Dishant
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_derivative(self, x):
        return (1-np.square(x))

    def __relu(self, x):
        return np.maximum(0,x)

    def __relu_derivative(self, x):
        return 1.0*(x>0)

    def compute_output_delta(self, out, activation):
        delta_output = 0
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))

        ## Inserted By Ekansh and Dishant
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))

        ## Inserted By Ekansh and Dishant
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation):
        delta_hidden_layer2 = 0
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))

        ## Inserted By Ekansh and Dishant
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))

        ## Inserted By Ekansh and Dishant
        elif activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation):
        delta_hidden_layer1 = 0
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))

        ## Inserted By Ekansh and Dishant
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))

        ## Inserted By Ekansh and Dishant
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        self.delta12 = delta_hidden_layer1


    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, activation, header = True):
        # TODO: Remember to implement the preprocess method
        #raw_test_input = pd.read_csv(self.test_dataset)
        raw_test_input = self.test_dataset
        #print("Raw Test Input: " + str(raw_test_input))
        test_dataset = self.preprocess(raw_test_input)
        ncols = len(test_dataset.columns)
        nrows = len(test_dataset.index)
        self.X = test_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = test_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        out = self.forward_pass(activation)
        error = 0.5 * np.power((out - self.y), 2)
        print("Test Output error: "+ str(np.sum(error)))
        return error


if __name__ == "__main__":
    urlForDataset = sys.argv[1]
    numberOfIterations = int(sys.argv[2])
    learningRate = float(sys.argv[3])
    activationFunction = sys.argv[4]
    #neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data")
    #neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    #neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")
    #neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data")
    neural_network = NeuralNet(urlForDataset)
    neural_network.train(activation = activationFunction, max_iterations = numberOfIterations, learning_rate = learningRate)
    testError = neural_network.predict(activation = activationFunction)

