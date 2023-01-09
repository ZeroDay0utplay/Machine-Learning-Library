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
    

    def gradient(self, w, b):
        dj_dw = np.zeros(self.n)
        dj_db = 0
        
        for i in range(self.m):
            f_wb = np.dot(self.X_train[i], w)+b - self.y_train[i]
            dj_dw += f_wb*self.X_train[i]
            dj_db += f_wb
        
        return dj_dw/self.m, dj_db/self.m


    def train(self, learning_rate=0.01, epochs=100, _lambda=0, gradient_fn=None, b_init=None, w_init=None):
        if gradient_fn is None:
            gradient_fn = self.gradient
        
        if b_init is None:
            b_init = 0
        
        if w_init is None:
            w_init = np.zeros(self.n)

        w = w_init
        b = b_init

        for k in range(epochs):
            dj_dw, dj_db = gradient_fn(w, b)
               
            w = w*(1 - learning_rate*_lambda/self.m) - learning_rate*dj_dw
            b = b - learning_rate*dj_db
            
            if not k%10:
                print(f"[+] Epochs: {k}/{epochs} Cost Function : {self.cost(w,b)}")
        
        self.w, self.b = w, b

        return w, b
    

    def predict(self, X_test):
        return X_test*self.w + self.b
    
    def accuracy(self, x, y, diff=1):
        true = 0
        acc = abs(self.predict(x) - y)
        print(acc)
        for t in acc:
            if t <= diff:
                true +=1
        print(f"[+] Accuracy: {true/len(acc)*100}%")
        return true/len(acc)