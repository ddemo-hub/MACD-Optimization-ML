from .base_app import BaseApp

from algorithms.naive_bayes import NaiveBayes

from src.utils.logger import Logger

from algorithms.commons.evaluators import f1_macro, confusion_matrix

import numpy


class NaiveBayesApp(BaseApp):
    def __init__(self, config_service, preprocessor):
        super().__init__(config_service=config_service, preprocessor=preprocessor)

    def train(self):
        # Get the preprocessed data
        processed_df = self.preprocessor.prepare_features()

        timestamps = processed_df["timestamp"].to_numpy()
        labels = processed_df["label"].to_numpy()
        features =  processed_df[processed_df.columns.drop(["timestamp", "label"])].to_numpy()
        
        # Train-Validation-Test Split
        features_train, features_validation, features_test = self.preprocessor.timeseries_split(features) 
        labels_train, labels_validation, labels_test = self.preprocessor.timeseries_split(labels)
        features_train = numpy.concatenate((features_train, features_validation))
        labels_train = numpy.concatenate((labels_train, labels_validation))

        # Initialize model
        model = NaiveBayes(method=self.config_service.naive_bayes_method)
        
        # Train Model
        model.fit(features_train, labels_train)

        # Test & Evaluate Model
        predictions = model.predict(features_test)

        test_f1 = f1_macro(labels_test, numpy.array(predictions))
        test_confusion_matrix = confusion_matrix(labels_test, numpy.array(predictions))        
        Logger.info(f"[Naive Bayes] Test F1 Score: {test_f1}") 
        Logger.info(f"[Naive Bayes] Test Confusion Matrix: \n{test_confusion_matrix}") 

        self.save_configs()

    def breast_cancer(self):
        # Read data
        from sklearn.datasets import load_breast_cancer
        
        dataset = load_breast_cancer()
        
        features = dataset.data
        labels = dataset.target

        # Split data
        features_train, features_validation, features_test = self.preprocessor.timeseries_split(features) 
        labels_train, labels_validation, labels_test = self.preprocessor.timeseries_split(labels) 
        features_train = numpy.concatenate((features_train, features_validation))
        labels_train = numpy.concatenate((labels_train, labels_validation))

        # Initialize Model
        model = NaiveBayes()

        # Train Model
        model.fit(features_train, labels_train)

        # Test & Evaluate Model
        predictions = model.predict(features_test)

        test_f1 = f1_macro(labels_test, numpy.array(predictions))
        test_confusion_matrix = confusion_matrix(labels_test, numpy.array(predictions))        
        Logger.info(f"[Naive Bayes] Test F1 Score: {test_f1}") 
        Logger.info(f"[Naive Bayes] Test Confusion Matrix: \n{test_confusion_matrix}") 
        
        self.save_configs()