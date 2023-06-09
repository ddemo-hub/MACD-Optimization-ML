from .base_app import BaseApp

from algorithms.k_nearest_neighbors import KNearestNeighbors

from src.utils.globals import Globals
from src.utils.logger import Logger

from algorithms.commons.evaluators import f1_macro, confusion_matrix, plot_f1

import numpy


class KNearestNeighborsApp(BaseApp):
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
        
        if self.config_service.k_nearest_neighnors_oversample == True:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            features_train, labels_train = ros.fit_resample(features_train, labels_train)
        elif self.config_service.k_nearest_neighnors_undersample == True:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            features_train, labels_train = rus.fit_resample(features_train, labels_train)
        
        # Initialize Model
        k_config = self.config_service.k_nearest_neighnors_k
        similarity_measure = self.config_service.k_nearest_neighnors_similarity_measure
        if type(k_config) == int:
            model = KNearestNeighbors(k=k_config)

            # Train model
            model.fit(features_train, labels_train)
            
            # Test Model
            predictions = model.predict(features_test, similarity_measure=similarity_measure)
            
            test_f1 = f1_macro(labels_test, numpy.array(predictions))
            test_confusion_matrix = confusion_matrix(labels_test, numpy.array(predictions))        
            Logger.info(f"[K Nearest Neighbor] Test F1 Score: {test_f1}") 
            Logger.info(f"[K Nearest Neighbor] Test Confusion Matrix: \n{test_confusion_matrix}") 
            
        elif type(k_config) == list:
            test_f1_per_k = []
            for k in range(k_config[0], k_config[1]+1):
                model = KNearestNeighbors(k=k)

                # Train model
                model.fit(features_train, labels_train)
                
                # Test Model
                predictions = model.predict(features_test, similarity_measure=similarity_measure)
                
                test_f1 = f1_macro(labels_test, numpy.array(predictions))
                test_confusion_matrix = confusion_matrix(labels_test, numpy.array(predictions))   
                Logger.info(f"[K Nearest Neighbor] K = {k} | Test F1 Score: {test_f1}") 
                Logger.info(f"[K Nearest Neighbor] K = {k} | Test Confusion Matrix: \n{test_confusion_matrix}") 
                
                test_f1_per_k.append(test_f1)      

            plot_f1(
                f1_scores=test_f1_per_k,
                num_epoch=k_config[1],
                savefig_path=Globals.artifacts_path.joinpath("f1.png")
            )

        self.save_configs()    

    def breast_cancer(self):
        from sklearn.datasets import load_breast_cancer
        
        dataset = load_breast_cancer()
        
        features = dataset.data
        labels = dataset.target
        
        features_train, features_validation, features_test = self.preprocessor.timeseries_split(features) 
        labels_train, labels_validation, labels_test = self.preprocessor.timeseries_split(labels) 
        features_train = numpy.concatenate((features_train, features_validation))
        labels_train = numpy.concatenate((labels_train, labels_validation))
        
        if self.config_service.k_nearest_neighnors_oversample == True:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            features_train, labels_train = ros.fit_resample(features_train, labels_train)
        elif self.config_service.k_nearest_neighnors_undersample == True:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            features_train, labels_train = rus.fit_resample(features_train, labels_train)
        
        # Initialize Model
        k_config = self.config_service.k_nearest_neighnors_k
        similarity_measure = self.config_service.k_nearest_neighnors_similarity_measure
        if type(k_config) == int:
            model = KNearestNeighbors(k=k_config)

            # Train model
            model.fit(features_train, labels_train)
            
            # Test Model
            predictions = model.predict(features_test, similarity_measure=similarity_measure)
            
            test_f1 = f1_macro(labels_test, numpy.array(predictions))
            test_confusion_matrix = confusion_matrix(labels_test, numpy.array(predictions))        
            Logger.info(f"[K Nearest Neighbor] Test F1 Score: {test_f1}") 
            Logger.info(f"[K Nearest Neighbor] Test Confusion Matrix: \n{test_confusion_matrix}") 
            
        elif type(k_config) == list:
            test_f1_per_k = []
            for k in range(k_config[0], k_config[1]+1):
                model = KNearestNeighbors(k=k)

                # Train model
                model.fit(features_train, labels_train)
                
                # Test Model
                predictions = model.predict(features_test, similarity_measure=similarity_measure)
                
                test_f1 = f1_macro(labels_test, numpy.array(predictions))
                test_confusion_matrix = confusion_matrix(labels_test, numpy.array(predictions))   
                Logger.info(f"[K Nearest Neighbor] K = {k} | Test F1 Score: {test_f1}") 
                Logger.info(f"[K Nearest Neighbor] K = {k} | Test Confusion Matrix: \n{test_confusion_matrix}") 
                
                test_f1_per_k.append(test_f1)      

            plot_f1(
                f1_scores=test_f1_per_k,
                num_epoch=k_config[1],
                savefig_path=Globals.artifacts_path.joinpath("f1.png")
            )
        
        self.save_configs()