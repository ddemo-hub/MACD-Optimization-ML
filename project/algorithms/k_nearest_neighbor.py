import numpy

class KNearestNeighbor():
    def __init__(self, k: int):
        self.k = k
        
        self.X_train: numpy.ndarray
        self.y_train: numpy.ndarray

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        self.X_train = X
        self.y_train = y
    
    def _predict(self, x):
        distances = [numpy.sqrt(numpy.sum(x-x_train)**2) for x_train in self.X_train]
        
        k_indices = numpy.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        
        return most_common_label
    
    def predict(self, X: numpy.ndarray):
        predictions = [self._predict(x) for x in X]
        return predictions
