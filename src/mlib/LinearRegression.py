import numpy as np

class LinearRegression:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        if len(X_train.shape) == 1:
            self.X_train.shape = self.X_train.shape[0], 1
            self.y_train.shape = self.y_train.shape[0], 1

        self.m, self.n = self.X_train.shape
            
    

    def cost(self, w, b):
        J = 0
        for i in range(self.m):
            J += ((np.dot(w, self.X_train[i] + b) - self.y_train[i])**2)
        
        return int(J/2*self.m)


    def train(self, learning_rate=0.01, epochs=100, _lambda=0, ):
        w = np.zeros(self.n)
        b = 0

        dj_dw = np.zeros(self.n)
        dj_db = 0

        for k in range(epochs):
            for i in range(self.m):
                f_wb = np.dot(self.X_train[i], w)+b - self.y_train[i]
                dj_dw += f_wb*self.X_train[i]
                dj_db += f_wb
            
            dj_dw /= self.m
            dj_db /= self.m
               
            w = w*(1 - learning_rate*_lambda/self.m) - learning_rate*dj_dw
            b = b - learning_rate*dj_db
            
            if not k%10:
                print(f"[+] Epochs: {k}/{epochs} Cost Function : {self.cost(w,b)}")
        
        self.w, self.b = w, b

        return w, b
    

    def predict(self, X_test):
        return X_test*self.w + self.b