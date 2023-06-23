import numpy as np
import pandas as pd
import math

TRAINING_STEPS = 1000
STEP_SIZE = 0.00001

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def accuracy(y, y_pred):
    correct_0, correct_1 = 0, 0
    count_0, count_1 = 0, 0
    for i in range(len(y)):
        if y[i] == 0:
            count_0 += 1
            if y_pred[i] == 0:
                correct_0 += 1
        if y[i] == 1:
            count_1 += 1
            if y_pred[i] == 1:
                correct_1 += 1
    print(f'Class 0: tested {count_0}, correctly classified {correct_0}')
    print(f'Class 1: tested {count_1}, correctly classified {correct_1}')
    print(f'Overall: tested {count_0 + count_1}, correctly classified {correct_0 + correct_1}')
    print(f'Accuracy = {(correct_0 + correct_1) / (count_0 + count_1)}')

def predict(X_test, thetas):
    y_pred = []
    for row in X_test:
        prediction = sigmoid(np.transpose(thetas) @ row)
        if prediction > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

def train(X, y, parameters):
    thetas = np.zeros(parameters)
    for step in range(TRAINING_STEPS):
        gradient = np.zeros(parameters)
        for i in range(len(y)):
            row = X[i]
            y_hat = sigmoid(np.transpose(thetas) @ row)
            for j in range(len(row)):
                gradient[j] += X[i][j] * (y[i] - y_hat)
        for k in range(len(thetas)):
            thetas[k] += STEP_SIZE * gradient[k]
    return thetas


def main():
    path = "./data/"
    training = np.genfromtxt(path + "train_cleaned.csv", delimiter=',', dtype=float, skip_header=1)
    # training = pd.read_csv('tt.csv')
    #training = np.delete(training, -2, 1)
    np.delete(training, 0, 1)
    # input outputs of training
    X = training[:, :-1]
    y = training[:, -1]
    examples = len(y)
    # initalize an x_0 = 1 for all of data
    X = np.insert(X, 0, np.array([1] * examples), axis=1)
    parameters = len(X[0])
    test = np.genfromtxt(path + "test_cleaned.csv", delimiter=',', dtype=int, skip_header=1)
    #test = np.delete(test, -2, 1)
    #input outputs of testing
    X_test = test[:, :-1]
    y_test = test[:, -1]
    test_examples = len(y_test)
    X_test = np.insert(X_test, 0, np.array([1] * test_examples), axis=1)
    # initialize thetas
    thetas = train(X, y, parameters)
    # predict
    y_pred = predict(X_test, thetas)
    accuracy(y_test, y_pred)

if __name__ == '__main__':
    main()

