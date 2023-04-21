from src.services.config_service import ConfigService
from src.services.data_service import DataService

from preprocess.preprocessor import Preprocessor

from algorithms.logistic_regression import LogisticRegression

from src.utils.logger import Logger

import numpy

class LogisticRegressionApp():
    def __init__(self, config_service: ConfigService, data_service: DataService, preprocessor: Preprocessor):
        self.config_service = config_service
        self.data_service = data_service
        self.preprocessor = preprocessor
        
    def main(self):
        # Set candle parameters
        start_ts = self.config_service.start_ts
        end_ts = self.config_service.end_ts
        symbol = self.config_service.symbol
        interval = self.config_service.interval
        
        # Set labeling parameters
        macd_fast = self.config_service.macd_fast
        macd_slow = self.config_service.macd_slow
        macd_signal = self.config_service.macd_signal
        
        # Get the preprocessed data
        model_inputs = self.preprocessor.prepare_model_inputs(
            start_ts=start_ts,
            end_ts=end_ts,
            symbol=symbol,
            interval=interval,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal
        )
        timestamps = model_inputs["timestamp"].to_numpy()
        labels = model_inputs["label"].to_numpy()
        features =  model_inputs[model_inputs.columns.drop(["timestamp", "label"])].to_numpy()
        
        # Non-Linear Transformation
        
        # Train-Validation-Test Split
        split_sizes = [
            int(features.shape[0]*self.config_service.training_size), 
            int(features.shape[0]*(self.config_service.training_size+self.config_service.validation_size))
        ]
        features_train, features_validation, features_test = numpy.split(features, split_sizes) 
        labels_train, labels_validation, labels_test = numpy.split(labels, split_sizes) 
        
        # Prepare mini-batches
        feature_batches = numpy.array_split(features_train, self.config_service.logistic_regression_batch_size)
        label_batches = numpy.array_split(labels_train, self.config_service.logistic_regression_batch_size)
        
        # Initialize Model
        model = LogisticRegression(num_features=features.shape[1])
        
        # Training Loop
        for epoch in range(self.config_service.logistic_regression_num_epochs):
            training_loss = 0
            for batch_index in range(len(feature_batches)):
                feature_batch = feature_batches[batch_index]
                label_batch = label_batches[batch_index]
                
                training_loss += model.train(feature_batch, label_batch, lr=0.00001) 
            
            # Validation        
            pred_classes, validation_loss = model.predict(features_validation, self.config_service.logistic_regression_threshold, labels_validation)
            Logger.info(f"[Logistic Regression] Epoch: {epoch} | Training Loss: {training_loss/features_train.shape[0]} | Validation Loss: {validation_loss/features_validation.shape[0]}") 
        
        Logger.info(f"[Logistic Regression] Weights: {model.weights}")
        Logger.info(f"[Logistic Regression] Bias: {model.bias}")
        
    def testing(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import f1_score
        
        dataset = load_breast_cancer()
        
        features = dataset.data
        labels = dataset.target
        
        features_train, features_test = numpy.split(features, [int(features.shape[0]*0.9)]) 
        labels_train, labels_test = numpy.split(labels, [int(features.shape[0]*0.9)]) 
        
        model = LogisticRegression(num_features=features.shape[1])
        for epoch in range(1000):
            training_loss = model.train(features_train, labels_train, lr=0.00005)
            pred_classes, test_loss = model.predict(features_test, 0.5, labels_test)
            
            print(f"[Logistic Regression] Epoch: {epoch} | Training Loss: {training_loss} | Validation Loss: {test_loss} | F1 Score: {f1_score(list(labels_test), pred_classes, average='macro')}") 

        print(f"[Logistic Regression] F1 Score: {f1_score(list(labels_test), pred_classes, average='macro')}") 
            