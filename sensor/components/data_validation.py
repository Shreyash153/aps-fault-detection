from sensor.entity import artifact_entity, config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
import pandas as pd
import os, sys
from sensor import utils
import numpy as np
from sensor.config import TARGET_COLUMN

class DataValidation:

    def __init__(self, 
                 data_validation_config:config_entity.DataValidationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()

        except Exception as e:
            raise SensorException(e, sys)


    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop columns which contains missing values more than specified threshold

        df: Accepts a pandas dataframe
        threshold: Percentage criteria to drop a column
        =========================================================================================
        returns Pandas Dataframe if atleast a single column available after dropping missing columns or none-
        """
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report= df.isna().sum()/df.shape[0] 
            #selecting column names which has null values
            logging.info(f"selecting column names which contains null values above {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name] = list(drop_column_names)
            df.drop(list(drop_column_names), axis=1, inplace=True)

            #return None if no column left
            if len(df.columns)==0:
                logging.info('#return None if no column left')
                return None
            
            return df

        except Exception as e:
            raise SensorException(e,sys)
    
    def is_required_columns_exist(self,base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns =[]
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: {base_column} is not available.")
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name]= list(missing_columns)
                return False
            
            return True

        except Exception as e:
            raise SensorException(e, sys)

    
    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report = dict()
            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data, current_data =base_df[base_column].dropna(), current_df[base_column].dropna()
                #Null hypothesis is that both column data drawn from same distribution
                # logging.info(f"Hypothesis : {base_column}: {base_data.dtype}, {current_data.dtype}")
                same_distribution = ks_2samp(base_data, current_data)

                # Checking p-value to decide if data drift occurred
                if float(same_distribution.pvalue) > 0.05:
                    #we are accepting null hypothesis
                    drift_report[base_column]={
                        'pvalue':float(same_distribution.pvalue),
                        'same_distribution': True
                    }
                else:
                    drift_report[base_column]={
                        'pvalue':float(same_distribution.pvalue),
                        'same_distribution': False
                    }
            self.validation_error[report_key_name]= drift_report

        except Exception as e:
            raise SensorException(e, sys)



    def initiate_data_validation(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            #base_df has na as null
            logging.info(f"Replace na value in base df")
            base_df.replace({'na': np.NAN}, inplace=True)

            logging.info("Drop null values columns from base dataframe")
            base_df=self.drop_missing_values_columns(df=base_df, report_key_name="missing_values_within_base_dataset")
            
            logging.info("Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)

            logging.info("Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("Drop null values columns from train dataframe")
            train_df = self.drop_missing_values_columns(df=train_df,report_key_name="missing_values_within_train_dataset")
            
            logging.info("Drop null values columns from test dataframe")
            test_df = self.drop_missing_values_columns(df=test_df,report_key_name="missing_values_within_test_dataset")


            exclude_columns = TARGET_COLUMN
            utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)

            logging.info("Is all required columns are present in train df")
            train_df_cols_status = self.is_required_columns_exist(base_df=base_df, current_df=train_df,report_key_name="missing_columns_within_train_dataset")
            logging.info("Is all required columns are present in test df")
            test_df_cols_status = self.is_required_columns_exist(base_df=base_df, current_df=test_df,report_key_name="missing_columnns_within_test_dataset")

            if train_df_cols_status:
                logging.info(f"All columns are available in train df detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_for_train_dataset")

            if test_df_cols_status:
                logging.info(f"All columns are available in test df detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df,report_key_name="data_drift_for_test_dataset")
            
            #write the report
            logging.info("Write report in yml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path, 
                                  data= self.validation_error)
            
            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e,sys)


