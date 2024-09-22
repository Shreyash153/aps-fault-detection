from sensor.entity import artifact_entity, config_entity
from sensor.exception import SensorException
from sensor.logger import logging
import pandas as pd
import os, sys
import numpy as np
from xgboost import XGBClassifier
from sensor import utils
from sklearn.metrics import f1_score


class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                 data_transformation_artifact:artifact_entity.DataTrasformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = config_entity.ModelTrainerConfig
            self.data_transformation_artifact = artifact_entity.DataTrasformationArtifact


        except Exception as e:
            raise SensorException(e, sys)
        

    def train_model(self,X, y):
        xgb_clf = XGBClassifier()
        xgb_clf.fit(X,y)
        return xgb_clf
    

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("Loading numpy array data")
            train_arr =utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr =utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info("Splitting data")
            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            logging.info("Model training")
            model = ModelTrainer.train_model(x_train, y_train)

            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)
            logging.info(f"F1 train score: {f1_train_score}")

            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)
            logging.info(f"F1 test score: {f1_test_score}")

            #check for overfitting and underfitting or expected score
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                                expected accuracy: {self.model_trainer_config.expected_score}, model actual score is {f1_test_score}")
            

            diff = abs(f1_train_score-f1_test_score)
            if diff> self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than Overfitting threshold: {self.model_trainer_config.overfitting_threshold} d")

            #save the trained model
            logging.info(f"save the trained model")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
                                                 f1_train_score= f1_train_score, 
                                                 f1_test_score= f1_test_score)
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact


        except Exception as e:
            raise SensorException(e, sys)