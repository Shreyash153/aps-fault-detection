from sensor.exception import SensorException
from sensor.logger import logging
import os, sys
from sensor.entity.config_entity import MODEL_FILE_NAME,  TARGET_ENCODER_OBJECT_FILE_NAME,TRANSFORMER_OBJECT_FILE_NAME
import glob
from typing import Optional

class ModelResolver:

    def __init__(self,model_registry:str ="saved_models",
                 transformer_dir_name="transformer",
                 target_encoder_dir_name = "target_encoder",
                 model_dir_name= "model"):
        
        self.model_registery = model_registry
        os.makedirs(self.model_registery, exist_ok=True)
        self.tansformer_dir_name =transformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name


    def get_latest_dir_path(self,)->Optional[str]:
        try:
            dir_names = os.listdir(self.model_registery)
            if len(dir_names)==0:
                return None
            dir_names = list(map(int, dir_names))
            latest_dir_name = max(dir_names)
            return os.path.join(self.model_registery, f"{latest_dir_name}")
 
        except Exception as e:
            raise SensorException(e, sys)
        
    def get_latest_model_path(self,): 
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise SensorException(e,sys)

    def get_latest_transformer_path(self,): 
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transformer is not available")
            return os.path.join(latest_dir, self.tansformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e,sys)

    def get_latest_target_encoder_path(self,): 
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Latest target encoder is not available")
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e,sys)
        
    # def get_latest_save_dir_path(self,):
    #     try:
    #         latest_dir = self.get_latest_dir_path()
    #         if latest_dir==None:
    #             os.path.join(self.model_registery,f"{0}")
    #         latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
    #         return os.path.join(self.model_registery,f"{latest_dir_num+1}")

    #     except Exception as e:
    #         raise SensorException(e, sys)
        
    def get_latest_save_dir_path(self,):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir == None:
                # When no directory exists, return the path for the first model (directory 0)
                return os.path.join(self.model_registery, f"{0}")
            
            # Use the already fetched latest_dir instead of calling get_latest_dir_path() again
            latest_dir_num = int(os.path.basename(latest_dir))
            return os.path.join(self.model_registery, f"{latest_dir_num+1}")

        except Exception as e:
            raise SensorException(e, sys)

        
    def get_latest_save_model_path(self,): 
        try:
            latest_dir = self.get_latest_save_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise SensorException(e,sys)

    def get_latest_save_transformer_path(self,): 
        try:
            latest_dir = self.get_latest_save_dir_path()
            if latest_dir is None:
                raise Exception(f"Transformer is not available")
            return os.path.join(latest_dir, self.tansformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e,sys)

    def get_latest_save_target_encoder_path(self,): 
        try:
            latest_dir = self.get_latest_save_dir_path()
            if latest_dir is None:
                raise Exception(f"Latest target encoder is not available")
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e,sys)
    
    

    
class Predictor:

    def __init__(self, model_resolver: ModelResolver) -> None:
        pass