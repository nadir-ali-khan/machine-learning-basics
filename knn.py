# updated
from collections import Counter
import math

def euclidean(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        dists = sorted(range(len(self.X)), key=lambda i: euclidean(x, self.X[i]))
        return Counter(self.y[i] for i in dists[:self.k]).most_common(1)[0][0]
