import numpy

class NaiveBayes():
    def __init__(self, method: str="gaussian"):
        if method not in ["gaussian", "categorical"]:
            raise Exception("Only Gaussian & Categorical Naive Bayes algorithms are supported. The 'method' parameter must be either 'gaussian' or 'categorical'.")
        self.method = method

        self._classes: list

        self._prior: numpy.ndarray

        if method == "gaussian":
            self._mean: numpy.ndarray
            self._var: numpy.ndarray
        elif method == "categorical":
            self._probs: list[list[dict]] = []

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        n_samples, n_features = X.shape

        self._classes = numpy.unique(y)

        # Initialize prior 
        self._prior = numpy.zeros(len(self._classes), dtype=numpy.float64)
        
        # If Gaussian Naive Bayes is used, initialize mean and variance
        if self.method == "gaussian":
            self._mean = numpy.zeros((len(self._classes), n_features), dtype=numpy.float64)
            self._var = numpy.zeros((len(self._classes), n_features), dtype=numpy.float64)
        elif self.method == "categorical":
            self._probs = [0] * len(self._classes)

        for index, label in enumerate(self._classes):
            X_label = X[y == label]

            # Calculate the prior for each class
            self._prior[index] = len(X_label) / n_samples
            
            # If Gaussian Naive Bayes is used, calculate mean and variance 
            if self.method == "gaussian":
                self._mean[index, :] = X_label.mean(axis=0)
                self._var[index, :] = X_label.var(axis=0)

            elif self.method == "categorical":
                feature_probs = [
                    {
                        category: sum(X_label[:, feature_idx] == category) 
                        / 
                        len(X_label[:, feature_idx]) 
                        for category in numpy.unique(X_label[:, feature_idx])
                    }
                    for feature_idx in range(X_label.shape[1])
                ]

                self._probs[index] = feature_probs

    def _gauss(self, class_index: int, x: numpy.ndarray):
        mean = self._mean[class_index]
        var = self._var[class_index]
        
        return numpy.exp(-((x - mean) ** 2) / (2 * var)) / numpy.sqrt(2 * numpy.pi * var)

    def _predict(self, x: numpy.ndarray):
        # In order to avoid overflow and underflow problems, log likelihood is used

        if self.method == "gaussian": 
            posteriors = [  
                numpy.log(self._prior[index]) # Prior
                +
                numpy.sum(numpy.log(self._gauss(index, x))) # Posterior
                for index in range(len(self._classes))                  
            ]

        elif self.method == "categorical":
            try:
                posteriors = [
                    numpy.log(self._prior[index]) # Prior
                    +
                    numpy.sum([numpy.log(self._probs[index][feature_index][x[feature_index]]) for feature_index in range(len(x))])    # Posterior
                    for index in range(len(self._classes))
                ]
            except KeyError:
                raise Exception(f"One of the categories in the set {x} was not encountered during training")

        return self._classes[numpy.argmax(posteriors)]

    def predict(self, X: numpy.ndarray):
        predictions = [self._predict(x) for x in X]
        return predictions