from datetime import datetime
import os
from sensor.exception import SensorException
import sys

# this file defines input entities for each component

FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pk1"
TARGET_ENCODER_OBJECT_FILE_NAME = "targer_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE =0.8
OVER_FITTING_THRESHOLD = 0.1

class TrainingPipelineConfig:
    def __init__(self):
        self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%y__%H%M%S')}")


class DataIngestionConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name ='aps'
            self.collection_name='sensor'
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feture_store", FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
            self.test_size =  0.2

        except Exception as e:
            raise SensorException(e, sys)
        
    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e, sys)

class DataValidationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_valition_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        self.report_file_path = os.path.join(self.data_valition_dir, "report.yaml")
        self.missing_threshold:float = 0.2
        self.base_file_path = os.path.join('aps_failure_training_set1.csv')

class DataTransformationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig) -> None:
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        self.transform_object_path = os.path.join(self.data_transformation_dir, "transformer", TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path = os.path.join(self.data_transformation_dir,"transformed", TRAIN_FILE_NAME.replace("csv","npz"))
        self.transformed_test_path = os.path.join(self.data_transformation_dir,"transformed", TEST_FILE_NAME.replace("csv","npz"))
        self.target_encoder_path = os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)




class ModelTrainerConfig:
      def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir,"model_trainer")
        self.model_path: str = os.path.join(self.model_trainer_dir,"trained_model",MODEL_FILE_NAME )
        self.expected_score: float = MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_threshold = OVER_FITTING_THRESHOLD




class ModelEvaluationConfig:...
class ModelPusherConfig:...