import numpy as np

class LogisticRegression:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        if len(X_train.shape) == 1:
            self.m, self.n = self.X_train.shape[0], 1
            self.m, self.n = self.y_train.shape[0], 1
        else:
            self.m, self.n = self.X_train.shape
    

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    

    def loss(self, w, b): # Squared error cost
        L = 0
        for i in range(self.m):
            f_wb = self.sigmoid(np.dot(self.X_train[i], w)+b)
            L += (self.y_train[i]*np.log(f_wb) + (1-self.y_train[i])*np.log(1-f_wb))
        return -L/self.m
    

    def gradient(self, w, b):
        dj_dw = np.zeros(self.n)
        dj_db = 0

        for i in range(self.m):
            derr_f_wb = self.sigmoid(np.dot(self.X_train[i], w)+b) - self.y_train[i]
            dj_dw += derr_f_wb*self.X_train[i]
            dj_db += derr_f_wb
        
        dj_dw /= self.m
        dj_db /= self.m
        
        return dj_dw, dj_db


    def train(self, learning_rate=0.01, epochs=100, _lambda=0, optimizer=None, b_init=None, w_init=None):
        if optimizer is None:
            optimizer = self.gradient
        
        if b_init is None:
            b_init = 0
        
        if w_init is None:
            w_init = np.zeros(self.n)

        w = w_init
        b = b_init

        for k in range(epochs):
            dj_dw, dj_db = optimizer(w, b)
               
            w = w*(1 - learning_rate*_lambda/self.m) - learning_rate*dj_dw
            b = b - learning_rate*dj_db
            
            if not k%(epochs/10):
                print(f"[+] Epochs: {k}/{epochs} loss Function : {self.loss(w,b)}")
        
        self.w, self.b = w, b

        return w, b
    

    def predict(self, X_test, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        return self.sigmoid(np.dot(X_test, w) + b)
    

    def zdo_acc(self, y, y_pred):
        acc=0
        ln = len(y)
        for i in range(ln):
            acc += abs(y[i]-y_pred[i])
        return (1-acc/ln)
    

    def accuracy(self, x, y, w=None, b=None, error="zdo_acc"):
        y_pred = self.predict(x, w, b)
        if error == "zdo_acc":
            acc = self.zdo_acc(y, y_pred)
        elif error == "mae":
            acc = self.mae(y, y_pred)
        elif error == "mse":
            acc = self.mse(y, y_pred)
        elif error == "rmse":
            acc = self.rmse(y, y_pred)
        elif error == "mape":
            acc = self.mape(y, y_pred)
        elif error == "r2" or True:
            acc = self.r2(y, y_pred)
        print(f"[+] Accuracy: {acc*100}%")
        return acc
    

    def best_acc(self, x, y, error="zdo_acc"):
        best_acc = 0
        best_w = 0
        best_b = 0
        learning_rates = [0.0001*(10**i) for i in range(4)]
        lambdas = [0.01*(10**i) for i in range(4)]
        epochs = [10*(10**i) for i in range(4)]
        for alpha in learning_rates:
            for lambda_ in lambdas:
                for epoch in epochs:
                    print(f"\n\n[+] Trying learning rate: {alpha}, lambda: {lambda_}, epochs: {epoch}\n\n")
                    w, b = self.train(learning_rate=alpha, _lambda=lambda_, epochs=epoch)
                    acc = self.accuracy(x, y, w, b, error=error)
                    if acc > best_acc:
                        best_w = w
                        best_b = b
                        best_acc = acc
        print(f"\n\n[*] Best accuracy : {best_acc} with w: {best_w} and b: {best_b}")
        return best_w, best_b
