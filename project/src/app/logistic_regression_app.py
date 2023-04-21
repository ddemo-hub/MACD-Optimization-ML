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
        # Get the preprocessed data
        processed_df = self.preprocessor.prepare_model_inputs()
        timestamps = processed_df["timestamp"].to_numpy()
        labels = processed_df["label"].to_numpy()
        features =  processed_df[processed_df.columns.drop(["timestamp", "label"])].to_numpy()
        
        # Non-Linear Transformation
        
        # Train-Validation-Test Split
        features_train, features_validation, features_test = self.preprocessor.timeseries_split(features) 
        labels_train, labels_validation, labels_test = self.preprocessor.timeseries_split(labels)
        
        # Prepare mini-batches
        feature_batches = self.preprocessor.mini_batch(features_train)
        label_batches = self.preprocessor.mini_batch(labels_train)
        
        # Initialize Model
        model = LogisticRegression(num_features=features.shape[1], regularization=True, constant=1)
        
        # Training Loop
        for epoch in range(self.config_service.logistic_regression_num_epochs):
            training_loss = 0
            for batch_index in range(len(feature_batches)):
                feature_batch = feature_batches[batch_index]
                label_batch = label_batches[batch_index]
                
                training_loss += model.train(feature_batch, label_batch, lr=self.config_service.logistic_regression_learning_rate) 
            
            # Validation Step       
            validation_preds, validation_loss = model.predict(features_validation, self.config_service.logistic_regression_threshold, labels_validation)
            
            Logger.info(f"[Logistic Regression] Epoch: {epoch} | Training Loss: {training_loss} | Validation Loss: {validation_loss}") 
        
        test_preds = model.predict(features_test, self.config_service.logistic_regression_threshold)        
        
        Logger.info(f"[Logistic Regression] Weights: {model.weights}")
        Logger.info(f"[Logistic Regression] Bias: {model.bias}")
        
    def breast_cancer(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import f1_score
        
        dataset = load_breast_cancer()
        
        features = dataset.data
        labels = dataset.target
        
        features_train, features_validation, features_test = self.preprocessor.timeseries_split(features) 
        labels_train, labels_validation, labels_test = self.preprocessor.timeseries_split(labels) 

        feature_batches = self.preprocessor.mini_batch(features_train)
        label_batches = self.preprocessor.mini_batch(labels_train)
        
        model = LogisticRegression(num_features=features.shape[1], regularization=True, constant=1)
        
        for epoch in range(self.config_service.logistic_regression_num_epochs):
            training_loss = 0
            for batch_index in range(len(feature_batches)):
                feature_batch = feature_batches[batch_index]
                label_batch = label_batches[batch_index]
                
                training_loss += model.train(feature_batch, label_batch, lr=self.config_service.logistic_regression_learning_rate) 
            
            validation_preds, validation_loss = model.predict(features_validation, self.config_service.logistic_regression_threshold, labels_validation)
            validation_f1 = f1_score(list(labels_validation), validation_preds, average='macro')
            
            print(f"[Logistic Regression] Epoch: {epoch} | Training Loss: {training_loss} | Validation Loss: {validation_loss} | Validation F1 Score: {validation_f1}") 

        test_preds = model.predict(features_test, self.config_service.logistic_regression_threshold)
        print(f"[Logistic Regression] F1 Score: {f1_score(list(labels_test), test_preds, average='macro')}") 
            