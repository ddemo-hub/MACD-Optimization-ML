from src.utils.logger import Logger

import numpy

class LogisticRegression():
    def __init__(self, num_features: int, regularization: bool=False, constant: int=1, stochastic: bool=False):
        self.weights = numpy.zeros(num_features+1)
        self.regularization = regularization
        self.constant = constant
        self.stochastic = stochastic

    def _sigmoid(self, X):
        return 1 / (1 + numpy.exp(-X))
    
    def _log_loss(self, pred, ground):
        return numpy.sum(ground * numpy.log(pred) + (1 - ground) * numpy.log(1 - pred)) / -len(pred)
    
    def train(self, X: numpy.ndarray, y: numpy.ndarray, lr: float):
        X = numpy.hstack((numpy.ones((X.shape[0], 1)), X))
        
        linear_pred = numpy.dot(X, self.weights)
        logistic_pred = self._sigmoid(linear_pred)
        
        if self.stochastic == True:
            rand = numpy.random.randint(low=0, high=X.shape[0]-1) 
            dWeights = X[rand] * (logistic_pred[rand]- y[rand])
        else:
            dWeights = (1 / X.shape[0]) * (X.T @ (logistic_pred - y))
        
        if self.regularization == True:
            dWeights -= (1 / X.shape[0]) * (self.constant * self.weights)
        
        self.weights -= lr * dWeights
        
        loss = self._log_loss(logistic_pred, y)        
        return loss

    def predict(self, X: numpy.ndarray, threshold: int, y: numpy.ndarray=None):
        X = numpy.hstack((numpy.ones((X.shape[0], 1)), X))
        
        linear_pred = numpy.dot(X, self.weights) 
        logistic_pred = self._sigmoid(linear_pred)
        
        pred_classes = [1 if pred >= threshold else 0 for pred in logistic_pred]
        
        if y is not None:    
            pred_loss = self._log_loss(logistic_pred, y)
            return pred_classes, pred_loss
        
        return pred_classes
    