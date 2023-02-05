import numpy as np

class PolynomialRegression:
    def __init__(self, X_train, y_train, deg=1):
        self.X_train = X_train
        self.y_train = y_train
        self.deg = deg

        if len(X_train.shape) == 1:
            self.m, self.n = self.X_train.shape[0], 1
            self.m, self.n = self.y_train.shape[0], 1
        else:
            self.m, self.n = self.X_train.shape


    def cost(self, w, b):
        J = 0
        for i in range(self.m):
            J += ((np.dot(w, self.X_train[i] + b) - self.y_train[i])**2)
        
        return int(J/(2*self.m))
    

    def gradient(self, w, b):
        dj_dw = np.zeros((self.n, self.deg))
        dj_db = 0
        
        for i in range(self.m):
            for d in range(1,self.deg+1):
                f_wb = np.dot(self.X_train[i]**d, w)+b - self.y_train[i]
                dj_dw += f_wb*self.X_train[i]
                dj_db += f_wb
        
        return dj_dw/self.m, dj_db/self.m        