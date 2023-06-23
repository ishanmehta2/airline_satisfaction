import csv
import numpy as np
import math

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.class_prior = {}
        self.mean = {}
        self.variance = {}
        
        # Calculate class priors
        for c in self.classes:
            self.class_prior[c] = np.mean(y == c)
        
        # Calculate mean and variance for each feature in each class
        for c in self.classes:
            class_data = X[y == c]
            num_samples = len(class_data)
            num_features = class_data.shape[1]
            
            self.mean[c] = np.mean(class_data, axis=0)
            self.variance[c] = np.var(class_data, axis=0) + 1e-10

    def predict(self, X):
        predictions = []
        
        for sample in X:
            probabilities = {}
            
            # Calculate likelihood probabilities for each class
            for c in self.classes:
                class_probability = self.class_prior[c]
                feature_probabilities = (1 / (np.sqrt(2 * np.pi * self.variance[c]))) * \
                                        np.exp(-((sample - self.mean[c]) ** 2) / (2 * self.variance[c]))
                class_probability *= np.prod(feature_probabilities)
                probabilities[c] = class_probability

            # Select the class with the highest probability
            prediction = max(probabilities, key=probabilities.get)
            predictions.append(prediction)
        
        return predictions

    def accuracy(self, y, y_pred):
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
        print(f'Accuracy = {(correct_0 + correct_1) / (count_0 + count_1)}')


# Read data from CSV file
def read_csv(file_path, excluded_features=None):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        feature_indices = [i for i, header in enumerate(headers) if header not in excluded_features]
        X = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=feature_indices)
        y = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=-1, dtype=int)
    return X, y


# File paths of the CSV files
path = "./data/"
train_csv_file_path = path + 'train_cleaned.csv'
test_csv_file_path = path + 'test_cleaned.csv'

# Excluded features
excluded_features = []
# Read training and testing data from CSV
X_train, y_train = read_csv(train_csv_file_path, excluded_features)
X_test, y_test = read_csv(test_csv_file_path, excluded_features)

# Create a Gaussian Naive Bayes classifier
classifier = GaussianNaiveBayes()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Predict the class labels for the test data
y_pred = classifier.predict(X_test)

# Calculate the accuracy
classifier.accuracy(y_test, y_pred)
