from .base_app import BaseApp

from algorithms.logistic_regression import LogisticRegression

from src.utils.globals import Globals
from src.utils.logger import Logger

from algorithms.commons.evaluators import f1_macro, confusion_matrix, plot_loss, plot_f1

import numpy


class LogisticRegressionApp(BaseApp):
    def __init__(self, config_service, preprocessor):
        super().__init__(config_service=config_service, preprocessor=preprocessor)
    
    def RBF(self, X, gamma=None):
        if gamma == None:
            gamma = 1.0/X.shape[1]
                        
        K = numpy.exp(-gamma * numpy.sum((X - X[:,numpy.newaxis])**2, axis = -1))

        return K
    
    def POLY(self, X, degree=2):
        original_axis = X.shape[1]
        for d in range(2, degree+1):
            X = numpy.hstack((X, (X[:, :original_axis]**d)))
        
        return X
    
    def TRIG(self, X):
        sin = numpy.sin(X)
        cos = numpy.cos(X)
        
        X = numpy.hstack((X, sin, cos))
        
        return X
        
    def train(self):
        # Get the preprocessed data
        processed_df = self.preprocessor.prepare_features()

        timestamps = processed_df["timestamp"].to_numpy()
        labels = processed_df["label"].to_numpy()
        features =  processed_df[processed_df.columns.drop(["timestamp", "label"])].to_numpy()
        
        # Non-Linear Transformation
        for transformation in self.config_service.logistic_regression_transformation:
            if transformation == "RBF":
                features = self.RBF(features)
            elif transformation == "TRIG":
                features = self.TRIG(features)
            elif transformation == "POLY":
                features = self.POLY(features)
        
        # Train-Validation-Test Split
        features_train, features_validation, features_test = self.preprocessor.timeseries_split(features) 
        labels_train, labels_validation, labels_test = self.preprocessor.timeseries_split(labels)
        
        # Prepare mini-batches
        feature_batches = self.preprocessor.mini_batch(features_train)
        label_batches = self.preprocessor.mini_batch(labels_train)
        
        # Initialize Model
        class_weights = None
        if self.config_service.logistic_regression_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(class_weight="balanced", classes=numpy.unique(labels_train), y=labels_train)
        
        model = LogisticRegression(
            num_features=features.shape[1], 
            regularization=self.config_service.logistic_regression_regularization, 
            constant=self.config_service.logistic_regression_constant,
            stochastic=self.config_service.logistic_regression_stochastic,
            class_weights=class_weights
        )
        
        # Training Loop
        learning_rate = self.config_service.logistic_regression_learning_rate
        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        validation_f1_score_per_epoch = []
        for epoch in range(1, self.config_service.logistic_regression_num_epochs+1):
            training_loss = 0
            for batch_index in range(len(feature_batches)):
                feature_batch = feature_batches[batch_index]
                label_batch = label_batches[batch_index]
                
                training_loss += model.fit(feature_batch, label_batch, lr=learning_rate) 
            
            # Calculate training loss
            training_loss /= batch_index+1
            
            # Validation Step       
            validation_preds, validation_loss = model.predict(features_validation, self.config_service.logistic_regression_threshold, labels_validation)
            validation_f1 = f1_macro(labels_validation, numpy.array(validation_preds))
            
            # End epoch
            Logger.info(f"[Logistic Regression] Epoch: {epoch} | Training Loss: {training_loss} | Validation Loss: {validation_loss} | Validation F1 Score: {validation_f1}") 
            training_loss_per_epoch.append(training_loss)
            validation_loss_per_epoch.append(validation_loss)
            validation_f1_score_per_epoch.append(validation_f1)

        with numpy.printoptions(threshold=numpy.inf):
            Logger.info(f"[Logistic Regression] Weights: \n{model.weights}")
            Logger.info(f"[Logistic Regression] Weights Sorted: \n{numpy.argsort(model.weights)}")
            
        # Test Model
        test_preds = model.predict(features_test, self.config_service.logistic_regression_threshold)        
        test_confusion_matrix = confusion_matrix(labels_test, numpy.array(test_preds))
        test_f1 = f1_macro(labels_test, numpy.array(test_preds))
        Logger.info(f"[Logistic Regression] Test F1 Score: {test_f1}") 
        Logger.info(f"[Logistic Regression] Test Confusion Matrix: \n{test_confusion_matrix}") 
        
        plot_loss(
            training_loss=training_loss_per_epoch, 
            validation_loss=validation_loss_per_epoch, 
            num_epoch=self.config_service.logistic_regression_num_epochs, 
            savefig_path=Globals.artifacts_path.joinpath("loss.png")
        )
        plot_f1(
            f1_scores=validation_f1_score_per_epoch,
            num_epoch=self.config_service.logistic_regression_num_epochs, 
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

        feature_batches = self.preprocessor.mini_batch(features_train)
        label_batches = self.preprocessor.mini_batch(labels_train)
        
        class_weights = None
        if self.config_service.logistic_regression_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(class_weight="balanced", classes=numpy.unique(labels_train), y=labels_train)
        
        model = LogisticRegression(
            num_features=features.shape[1], 
            regularization=self.config_service.logistic_regression_regularization, 
            constant=self.config_service.logistic_regression_constant,
            stochastic=self.config_service.logistic_regression_stochastic,
            class_weights=class_weights
        )
        
        learning_rate = self.config_service.logistic_regression_learning_rate
        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        validation_f1_score_per_epoch = []
        for epoch in range(1, self.config_service.logistic_regression_num_epochs+1):
            training_loss = 0
            for batch_index in range(len(feature_batches)):
                feature_batch = feature_batches[batch_index]
                label_batch = label_batches[batch_index]
                
                training_loss += model.fit(feature_batch, label_batch, lr=learning_rate) 
            
            training_loss /= batch_index+1
            
            validation_preds, validation_loss = model.predict(features_validation, self.config_service.logistic_regression_threshold, labels_validation)
            validation_f1 = f1_macro(labels_validation, numpy.array(validation_preds))
            
            Logger.info(f"[Logistic Regression] Epoch: {epoch} | Training Loss: {training_loss} | Validation Loss: {validation_loss} | Validation F1 Score: {validation_f1}") 
            training_loss_per_epoch.append(training_loss)
            validation_loss_per_epoch.append(validation_loss)
            validation_f1_score_per_epoch.append(validation_f1)
        
        with numpy.printoptions(threshold=numpy.inf):
            Logger.info(f"[Logistic Regression] Weights: \n{model.weights}")
            Logger.info(f"[Logistic Regression] Weights Sorted: \n{numpy.argsort(model.weights)}")

        test_preds = model.predict(features_test, self.config_service.logistic_regression_threshold)
        Logger.info(f"[Logistic Regression] Test F1 Score: {f1_macro(labels_test, numpy.array(test_preds))}") 
        Logger.info(f"[Logistic Regression] Test Confusion Matrix: \n{confusion_matrix(labels_test, numpy.array(test_preds))}") 
        
        plot_loss(
            training_loss=training_loss_per_epoch, 
            validation_loss=validation_loss_per_epoch, 
            num_epoch=self.config_service.logistic_regression_num_epochs, 
            savefig_path=Globals.artifacts_path.joinpath("loss.png")
        )
        plot_f1(
            f1_scores=validation_f1_score_per_epoch,
            num_epoch=self.config_service.logistic_regression_num_epochs, 
            savefig_path=Globals.artifacts_path.joinpath("f1.png")
        )

        self.save_configs()