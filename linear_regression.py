# v15861
class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            y_pred = [self.w * x + self.b for x in X]
            dw = -2/n * sum((y[i]-y_pred[i])*X[i] for i in range(n))
            db = -2/n * sum(y[i]-y_pred[i] for i in range(n))
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return [self.w * x + self.b for x in X]
