import numpy

class NaiveBayes():
    def __init__(self, method: str):
        self.method = method

        self._classes: list

        self._prior: numpy.ndarray
        self._mean: numpy.ndarray
        self._var: numpy.ndarray

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        n_samples, n_features = X.shape

        self._classes = numpy.unique(y)
        n_classes = len(self._classes)

        # Calculate the mean, variance and the prior for each class
        self._prior = numpy.zeros(n_classes, dtype=numpy.float64)
        self._mean = numpy.zeros((n_classes, n_features), dtype=numpy.float64)
        self._var = numpy.zeros((n_classes, n_features), dtype=numpy.float64)

        for index, clss in enumerate(self._classes):
            X_class = X[y == clss]

            self._prior[index] = len(X_class) / n_samples
            self._mean[index, :] = X_class.mean(axis=0)
            self._var[index, :] = X_class.var(axis=0)
        

    def _gauss(self, class_index: int, x: numpy.ndarray):
        mean = self._mean[class_index]
        var = self._var[class_index]
        
        return numpy.exp(-((x - mean) ** 2) / (2 * var)) / numpy.sqrt(2 * numpy.pi * var)

    def _predict(self, x: numpy.ndarray):
        # Calculate the posterior for every class
        posteriors = [  
            numpy.log(self._prior[index]) # Prior
            +
            numpy.sum(numpy.log(self._gauss(index, x))) # Posterior
            for index in range(len(self._classes))                  
        ]

        return self._classes[numpy.argmax(posteriors)]

    def predict(self, X: numpy.ndarray):
        predictions = [self._predict(x) for x in X]
        return predictions