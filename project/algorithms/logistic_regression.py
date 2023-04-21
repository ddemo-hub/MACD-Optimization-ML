from src.utils.logger import Logger

import numpy

class LogisticRegression():
    def __init__(self, num_features: int):
        self.weights = numpy.zeros(num_features)
        self.bias = 0

    def _sigmoid(self, X):
        return 1 / (1 + numpy.exp(-X))
    
    def _log_loss(self, pred, ground):
        return numpy.sum(ground * numpy.log(pred) + (1 - ground) * numpy.log(1 - pred)) / -len(pred)
    
    def train(self, X: numpy.ndarray, y: numpy.ndarray, lr: float):
        linear_pred = numpy.dot(X, self.weights) + self.bias
        logistic_pred = self._sigmoid(linear_pred)
        
        dWeights = (1 / X.shape[0]) * numpy.dot(X.T, (logistic_pred - y)) 
        dBias = (1 / X.shape[0]) * numpy.sum(logistic_pred - y) 

        self.weights -= lr * dWeights
        self.bias -= lr * dBias 
        
        loss = self._log_loss(logistic_pred, y)        
        return loss

    def predict(self, X: numpy.ndarray, threshold: int, y: numpy.ndarray=None):
        linear_pred = numpy.dot(X, self.weights) + self.bias
        logistic_pred = self._sigmoid(linear_pred)
        
        pred_classes = [1 if pred >= threshold else 0 for pred in logistic_pred]
        
        if y is not None:    
            pred_loss = self._log_loss(logistic_pred, y)
            return pred_classes, pred_loss
        
        return pred_classes
    