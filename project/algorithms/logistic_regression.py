from src.utils.logger import Logger

import numpy

class LogisticRegression():
    def __init__(self, num_features: int, regularization: bool=False, constant: int=1, stochastic: bool=False, class_weights: dict=None):
        self.weights = numpy.zeros(num_features+1)
        self.regularization = regularization
        self.constant = constant
        self.stochastic = stochastic
        self.cw = class_weights
        
    def _sigmoid(self, X):
        return 1 / (1 + numpy.exp(-X))
    
    def _log_loss(self, pred, ground):
        loss = numpy.sum(ground * numpy.log(pred) + (1 - ground) * numpy.log(1 - pred)) / -len(ground)

        if self.regularization == True:
            loss += ((self.constant / len(ground) / 2) * (self.weights.T @ self.weights))
        
        return loss
            
    def fit(self, X: numpy.ndarray, y: numpy.ndarray, lr: float):
        X = numpy.hstack((numpy.ones((X.shape[0], 1)), X))
        
        linear_pred = numpy.dot(X, self.weights)
        logistic_pred = self._sigmoid(linear_pred)
        
        if self.stochastic == True:
            rand = numpy.random.randint(low=0, high=X.shape[0]-1) 
            if self.cw is None:
                dWeights = X[rand] * (logistic_pred[rand]- y[rand])
            else:
                dWeights = X[rand] * ((self.cw[0]*y[rand]*(logistic_pred[rand]-1)) + (self.cw[1]*logistic_pred[rand]*(1-y[rand])))
        else:
            if self.cw is None:
                dWeights = (1 / X.shape[0]) * (X.T @ (logistic_pred - y))
            else:
                dWeights = (1 / X.shape[0]) * (X.T @ ((self.cw[0]*y*(logistic_pred-1)) + (self.cw[1]*logistic_pred*(1-y))))
        
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
    