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
            