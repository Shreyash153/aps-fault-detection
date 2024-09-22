from sensor.predictor import ModelResolver
from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import os, sys
from sensor.utils import load_object
from sklearn.metrics import f1_score
import pandas as pd
from sensor.config import TARGET_COLUMN


class ModelEvaluation:
    
    def __init__(self, model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTrasformationArtifact, 
                 model_trainer_artifact : artifact_entity.ModelTrainerArtifact
                 ):
        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact= data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()


        except Exception as e:
            raise SensorException(e, sys)
        

    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            #if saved model folder has model then we will compare
            #which model is best trained or the model from saved model folder
            logging.info(f"Getting latest dir path")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            logging.info(f"latest_dit_path: {latest_dir_path}")
            if latest_dir_path == None:
                model_eval_artifact =artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                             improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Finding locaion of transformer, model and encoder
            transformer_path =self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            logging.info("Loading previous trained object")
            # Previous trained objects
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)
            logging.info("Loaded successfully! ")

            logging.info("Loading currently trained object")
            # currently trained model objects
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            logging.info("Loaded successfully! ")

            logging.info(f"loading test_df")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            targer_df = test_df[TARGET_COLUMN]
            test_df.drop(TARGET_COLUMN, axis=1, inplace=True)
            y_true =target_encoder.transform(targer_df)

            # accuracy using previous trained 
            logging.info("Predicting on test data using previous model")
            input_arr = transformer.transform(test_df)
            y_pred = model.predict(input_arr)
            print(f"Prediction using previous model :{target_encoder.inverse_transform(y_pred[:5])}")

            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using previous model: {previous_model_score}")

            #accuracy using current trainde model
            logging.info("Predicting on test data using current model")
            input_arr = current_transformer.transform(test_df)
            y_pred = current_model.predict(input_arr)
            y_true = current_target_encoder.transform(targer_df)
            print(f"Prediction using trained model :{current_target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current model: {current_model_score}")

            if current_model_score< previous_model_score:
                logging.info("Current trained model is not better than previou model")
                raise Exception("Current trained model is not better than previou model")
            
            improved_accuracy = current_model_score -previous_model_score
            
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
                                                                          improved_accuracy=improved_accuracy)
            
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
           



        except Exception as e:
            raise SensorException(e, sys)