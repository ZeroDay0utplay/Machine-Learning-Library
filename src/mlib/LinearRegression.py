import numpy as np

class LinearRegression:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        try:
            self.m, self.n = self.X_train.shape
        except ValueError:
            self.m, self.n = self.X_train.shape[0], 1
    

    def cost(self, w, b):
        J = 0
        for i in range(self.m):
            J += ((np.dot(w, self.X_train[i] + b) - self.y_train[i])**2)
        
        return J/2*self.m


    def train(self, learning_rate=0.01, nb_iters=100, _lambda=1):
        w = np.zeros(self.n)
        b = 0

        for k in range(nb_iters):
            for i in range(self.m):
                f_wb = np.dot(self.X_train[i], w)+b - self.y_train[i]
                w = w*(1 - learning_rate*_lambda/self.m) - learning_rate*f_wb*self.X_train[i]
                b = b - learning_rate*f_wb
            
            if not k%10:
                print(f"[+] Cost Function : {self.cost(w,b)}")

        return w, b