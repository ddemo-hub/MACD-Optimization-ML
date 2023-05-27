import numpy

class KNearestNeighbors():
    def __init__(self, k: int):
        self.k = k
        
        self.X_train: numpy.ndarray
        self.y_train: numpy.ndarray

        self.similarity_measure: str

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        self.X_train = X
        self.y_train = y
    
    def _predict(self, x: numpy.ndarray):
        if self.similarity_measure == "euclidean":
            distances = [numpy.sqrt(numpy.sum(x-x_train)**2) for x_train in self.X_train]
        elif self.similarity_measure == "cosine":
            distances = [(x @ x_train.T) / (numpy.linalg.norm(x) * numpy.linalg.norm(x_train)) for x_train in self.X_train]
        elif self.similarity_measure == "manhattan":
            distances = [numpy.sum(abs(x-x_train)) for x_train in self.X_train] 

        if self.similarity_measure == "cosine":
            k_indices = numpy.argsort(distances)[::-1][:self.k] # A greater cosine similarity value means a lower distance
        else:
            k_indices = numpy.argsort(distances)[:self.k] 

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        
        return most_common_label
    
    def predict(self, X: numpy.ndarray, similarity_measure: str="euclidean"):
        if similarity_measure not in ["euclidean", "cosine", "manhattan"]:
            raise Exception("The 'similarity_measure' parameter must be either one of: ['euclidean', 'cosine', 'manhattan']")
        self.similarity_measure = similarity_measure

        predictions = [self._predict(x) for x in X]
        return predictions
