from src.services.config_service import ConfigService

from preprocess.preprocessor import Preprocessor

from algorithms.naive_bayes import NaiveBayes

from src.utils.globals import Globals
from src.utils.logger import Logger

from algorithms.commons.evaluators import f1_macro, confusion_matrix, plot_loss, plot_f1

import shutil
import numpy


class NaiveBayesApp():
    def __init__(self, config_service: ConfigService, preprocessor: Preprocessor):
        self.config_service = config_service
        self.preprocessor = preprocessor

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

        model = NaiveBayes()

        model.fit(features_train, labels_train)

        ...

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
        
        shutil.copyfile(Globals.project_path.joinpath("src", "configs", "config.yaml"), Globals.artifacts_path.joinpath("config.yamlignore"))

    def toy(self):
        import pandas
        dataset = pandas.read_csv("toy_dataset.csv")
        dataset = dataset[97:115].reset_index()

        features = dataset[["City", "Gender"]].to_numpy()
        labels = dataset["Illness"].to_numpy()

        features_train, features_test = numpy.array_split(features, [int(len(features) * 0.8)])
        labels_train, labels_test = numpy.array_split(labels, [int(len(features) * 0.8)])
        
        model = NaiveBayes(method="categorical")

        model.fit(features_train, labels_train)
        
        model.predict(features_test)

        ...