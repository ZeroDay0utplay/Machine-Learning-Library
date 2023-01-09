import pandas as pd
import numpy as np
from mlib.LinearRegression import LinearRegression
from scipy import stats


df = pd.read_csv("DS/LinearRegression/train.csv")
df_test = pd.read_csv("DS/LinearRegression/test.csv")

df = df.dropna()
df_test = df_test.dropna()

x_test = df_test["x"]
y_test = df_test["y"]

X_train = df["x"]
y_train = df["y"]


X_train = np.array([float(i) for i in X_train])
y_train = np.array([float(i) for i in y_train])
x_test = np.array([float(i) for i in x_test])


mx = max(X_train)
mn = min(X_train)

mx2 = max(x_test)
mn2 = min(x_test)

mx1 = max(y_train)
mn1 = min(y_train)


for i in range(X_train.shape[0]):
    X_train[i] = (X_train[i] - mn)/(mx-mn)


for i in range(x_test.shape[0]):
    x_test[i] = (x_test[i] - mn2)/(mx2-mn2)


lr = LinearRegression(X_train, y_train)
w, b = lr.train(epochs=100000, learning_rate=1e-6)
y_pred = lr.predict(x_test)
print(w, b)
print(y_pred)