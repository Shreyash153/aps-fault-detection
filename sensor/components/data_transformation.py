from sensor.entity import artifact_entity, config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
from sensor.config import TARGET_COLUMN
import pandas as pd
import os, sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sensor import utils


class DataTransformation:

    def __init__(self,
                 data_transformaation_config:config_entity.DataTransformationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact ) -> None:
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformaation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e,sys)
        
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer= SimpleImputer(strategy = "constant", fill_value=0)
            robust_scaler =RobustScaler()

            # Create a pipeline with simple imputer with strategy constant and fill value 0
            pipeline = Pipeline(steps=[
                ('Imputer', simple_imputer),
                ('RobustScaler', robust_scaler)
            ])
            return pipeline

        except Exception as e:
            raise SensorException(e,sys)
        

    def initiate_data_transformation(self,)->artifact_entity.DataTrasformationArtifact:
        try:
            #reading training and testing data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #Selecting input features for train and test dataset
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            #selecting targer feature for train and test dataset
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformation on target column
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            #Transforming input features
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            smt = SMOTETomek(sampling_strategy="minority")
            logging.info(f"Before resampling in tarining set,Input: {input_feature_train_arr.shape}, Target: {target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in tarining set,Input: {input_feature_train_arr.shape}, Target: {target_feature_train_arr.shape}")

            logging.info(f"Before resampling in test set,Input: {input_feature_test_arr.shape}, Target: {target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in test set,Input: {input_feature_test_arr.shape}, Target: {target_feature_test_arr.shape}")

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            #Save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, 
                                        array=train_arr )
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, 
                                        array=test_arr )            
            
            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                              obj=transformation_pipeline)
            
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
                              obj=label_encoder)
            
            data_transformation_artifact = artifact_entity.DataTrasformationArtifact(
                data_transformation_dir = self.data_transformation_config.data_transformation_dir,
                transform_object_path = self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path
            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e,sys)

    