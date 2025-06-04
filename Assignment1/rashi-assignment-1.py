#Rashi Mullick 24B3967 Assignment 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

X = train_data.drop(columns=["MedHouseVal"]) #THIS is the input variable
y = train_data["MedHouseVal"] #This is the target variable

# print(X.shape)
# print("        ")
# print(y.shape)

X = (X - X.mean()) / X.std()
X_test = (test_data - X.mean()) / X.std()#normalising data

#Converting to numpy arrays
X = X.values
y = y.values.reshape(-1, 1)  # turn it into a column
test_X = X_test.values

#Setting bias b to 1 and adding a column of it to X and test_X
#X = np.hstack((np.ones((X.shape[0], 1)), X))
#test_X = np.hstack((np.ones((test_X.shape[0], 1)), test_X))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=40)

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None  # will store weights including bias
        self.bias = None

    def fit(self, X, y):
        no_samples, no_parameters = X.shape
        self.weights = np.zeros((no_parameters, 1))  # starting with zero weights
        self.bias = 0.0

        for i in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            j = (1 / (2 * no_samples)) * np.sum((y_pred - y)**2)
            dw = (1 / no_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / no_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        return y_pred.flatten()


model = LinearRegression(learning_rate=0.01, iterations=1000)
#testing on 0.2 of the train data and calculating rmse
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
#training on train data
model.fit(X, y)
predictions = model.predict(test_X)

submission = pd.DataFrame({
    "row_id": test_data.index,
    "MedHouseVal": predictions.flatten()
})
submission.to_csv("submission.csv", index=False)

