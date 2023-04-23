import numpy

class KNearestNeighbor():
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _predict(self, X):
        distances = [numpy.sqrt(numpy.sum(X-x_train)**2) for x_train in self.X_train]
        
        k_indices = numpy.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        
        return most_common_label
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
