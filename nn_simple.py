#!/usr/local/bin/ python
# ******************************************************
#       _    _      ()_()
#      | |  | |    |(o o)
#   ___| | _| | ooO--`o'--Ooo
#  / __| |/ / |/ _ \ __|_  /
#  \__ \   <| |  __/ |_ / /
#  |___/_|\_\_|\___|\__/___|

# ******************************************************
# File name: nn_simple.py
# Author: skletz
# Date created: 15/07/2018
# Date last modified: 15/07/2018
# Python Version: 2.7
# ******************************************************
# Source:
# Blog: How to build a multi-layered neural network in Python (https://goo.gl/LZb46a)
# Blog: A Neural Network in 11 lines of Python (https://goo.gl/cxTp7B)
# ******************************************************

""" Simple Shallow Neural Network (Only one hidden layer)"""


import numpy as np
import matplotlib.pyplot as plt

debug_mode = 0

def activation_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activation_sigmoid_derivative(x):
    return x * (1 - x)


class TrainingSet:
    def __init__(self):
        self.input = []
        self.output = []

    def add(self, x1, x2, x3, y):
        self.input.append([x1, x2, x3])
        self.output.append([y])

    def print_data(self):
        print 'Inputs:'
        print(self.get_inputs())
        print 'Outputs:'
        print(self.get_outputs())

    def get_inputs(self):
        return np.array(self.input)

    def get_outputs(self):
        return np.array(self.output)


def main():
    print('Simple Shallow Neural Network')
    debug_mode = 1
    training_set = TrainingSet()
    training_set.add(0, 0, 1, 0)
    training_set.add(1, 1, 1, 1)
    training_set.add(1, 0, 1, 1)
    training_set.add(0, 1, 1, 0)

    X = training_set.get_inputs()
    y = training_set.get_outputs()

    training_set.print_data()

    if not debug_mode:
        np.random.seed(np.random.randint(500))
    else:
        np.random.seed(1)

    num_features = X.shape[1]
    num_neurons = y.shape[1]

    weights = 2 * np.random.random((num_features, num_neurons)) - 1
    print 'Initial weights:'
    print weights

    # Training
    for iter in xrange(10000):
        layer0 = X
        # generate guesses for the input
        guess = np.dot(layer0, weights)
        layer1 = activation_sigmoid(guess)

        # how much did we miss
        layer1_error = y - layer1

        # how much contribute layer1 to the error
        layer1_delta = layer1_error * activation_sigmoid_derivative(layer1)

        layer1_adjustment = np.dot(layer0.T, layer1_delta)

        weights += layer1_adjustment

    print "Output After Training:"
    print layer1
    print "Weights After Training"
    print weights

if __name__ == '__main__':
    main()
