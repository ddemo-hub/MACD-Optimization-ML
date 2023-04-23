from src.services.config_service import ConfigService

from preprocess.preprocessor import Preprocessor

from algorithms.k_nearest_neighbor import KNearestNeighbor

from src.utils.globals import Globals
from src.utils.logger import Logger

from algorithms.commons.evaluators import f1_macro, confusion_matrix, plot_loss

import shutil
import numpy


class KNearestNeighborApp():
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
        
        if self.config_service.k_nearest_neighnor_oversample == True:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            features_train, labels_train = ros.fit_resample(features_train, labels_train)
        elif self.config_service.k_nearest_neighnor_undersample == True:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            features_train, labels_train = rus.fit_resample(features_train, labels_train)
        
        # Initialize Model
        model = KNearestNeighbor(k=self.config_service.k_nearest_neighnor_k)

        # Train model
        model.fit(features_train, labels_train)
        
        # Test Model
        predictions = model.predict(features_test)
        
        test_f1 = f1_macro(labels_test, numpy.array(predictions))
        test_confusion_matrix = confusion_matrix(labels_test, numpy.array(predictions))        
        Logger.info(f"[K Nearest Neighbor] Test F1 Score: {test_f1}") 
        Logger.info(f"[K Nearest Neighbor] Test Confusion Matrix: \n{test_confusion_matrix}") 
        
        shutil.copyfile(Globals.project_path.joinpath("src", "configs", "config.yaml"), Globals.artifacts_path.joinpath("config.yamlignore"))


    def breast_cancer(self):
        from sklearn.datasets import load_breast_cancer
        
        dataset = load_breast_cancer()
        
        features = dataset.data
        labels = dataset.target
        
        features_train, features_validation, features_test = self.preprocessor.timeseries_split(features) 
        labels_train, labels_validation, labels_test = self.preprocessor.timeseries_split(labels) 
        features_train = numpy.concatenate((features_train, features_validation))
        labels_train = numpy.concatenate((labels_train, labels_validation))
        
        if self.config_service.k_nearest_neighnor_oversample == True:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            features_train, labels_train = ros.fit_resample(features_train, labels_train)
        elif self.config_service.k_nearest_neighnor_undersample == True:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            features_train, labels_train = rus.fit_resample(features_train, labels_train)
        
        # Initialize Model
        model = KNearestNeighbor(k=self.config_service.k_nearest_neighnor_k)

        # Train model
        model.fit(features_train, labels_train)
        
        # Test Model
        predictions = model.predict(features_test)
        
        test_f1 = f1_macro(labels_test, numpy.array(predictions))
        test_confusion_matrix = confusion_matrix(labels_test, numpy.array(predictions))        
        Logger.info(f"[K Nearest Neighbor] Test F1 Score: {test_f1}") 
        Logger.info(f"[K Nearest Neighbor] Test Confusion Matrix: \n{test_confusion_matrix}") 
        
        shutil.copyfile(Globals.project_path.joinpath("src", "configs", "config.yaml"), Globals.artifacts_path.joinpath("config.yamlignore"))
