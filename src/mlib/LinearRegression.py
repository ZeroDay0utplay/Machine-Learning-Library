import numpy as np

class LinearRegression:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        if len(X_train.shape) == 1:
            self.m, self.n = self.X_train.shape[0], 1
            self.m, self.n = self.y_train.shape[0], 1
        else:
            self.m, self.n = self.X_train.shape
            
    

    def loss(self, w, b): # Squared error cost
        J = 0
        for i in range(self.m):
            J += ((np.dot(w, self.X_train[i] + b) - self.y_train[i])**2)
        
        return int(J/(2*self.m))
    

    def gradient(self, w, b):
        dj_dw = np.zeros(self.n)
        dj_db = 0
        
        for i in range(self.m):
            derr_f_wb = np.dot(self.X_train[i], w)+b - self.y_train[i]
            dj_dw += derr_f_wb*self.X_train[i]
            dj_db += derr_f_wb
        
        return dj_dw/self.m, dj_db/self.m


    def train(self, learning_rate=0.01, epochs=100, _lambda=0, optimizer=None, b_init=None, w_init=None, with_print="YES"):
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
            
            if not k%(epochs/10) and with_print == "YES":
                print(f"[+] Epochs: {k}/{epochs} loss Function : {self.loss(w,b)}")
        
        self.w, self.b = w, b

        return w, b
    

    def predict(self, X_test, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        return np.dot(X_test, w) + b
    

    def mae(self, y, y_pred):
        ae = 0
        ln = len(y)
        for i in range(ln):
            ae += abs((y[i]-y_pred[i]))
        return ae/ln

    def mse(self, y, y_pred):
        se = 0
        ln = len(y)
        for i in range(ln):
            se += (y[i]-y_pred[i])**2
        return se/ln
    

    def rmse(self, y, y_pred):
        return np.sqrt(self.mse(y, y_pred))


    def mape(self, y, y_pred):
        ae = 0
        ln = len(y)
        for i in range(ln):
            ae += abs((y[i]-y_pred[i])/y[i])
        return ae/ln
    
    def r2(self, y, y_pred):
        ln = len(y)
        y_bar = sum(y)/ln
        r2_up = 0
        r2_down = 0
        for i in range(ln):
            r2_up += (y[i]-y_pred[i])**2
            r2_down += (y[i]-y_bar)**2
        return 1 - (r2_up/r2_down)
    

    def zdo_acc(self, y, y_pred):
        zdo = 0
        ln = len(y)
        for i in range(ln):
            y_i = y[i]
            y_hat_i = y_pred[i]
            zdo += (min(y_i, y_hat_i)/max(y_i, y_hat_i))
        zdo /= ln
        return zdo


    def accuracy(self, x, y, w=None, b=None, error="r2"):
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
    

    def best_acc(self, x, y, error="r2"):
        best_acc = 0
        best_w = 0
        best_b = 0
        learning_rates = [0.0001*(10**i) for i in range(4)]
        lambdas = [0.01*(10**i) for i in range(4)]
        epochs = [10*(10**i) for i in range(4)]
        for alpha in learning_rates:
            for lambda_ in lambdas:
                for epoch in epochs:
                    print(f"\n[*] Trying learning rate: {alpha}, lambda: {lambda_}, epochs: {epoch}\n")
                    w, b = self.train(learning_rate=alpha, _lambda=lambda_, epochs=epoch)
                    acc = self.accuracy(x, y, w, b, error=error)
                    if acc > best_acc:
                        best_w = w
                        best_b = b
                        best_acc = acc
        print(f"\n\n[*] Best accuracy : {best_acc} with w: {best_w} and b: {best_b}")
        return best_w, best_b