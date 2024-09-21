from dataclasses import dataclass

# this file defines output entities for each component


@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    report_file_path:str


@dataclass
class DataTrasformationArtifact:
    data_transformation_dir:str
    transform_object_path:str
    tansform_train_path :str
    tansform_test_path:str
    target_encoder_path:str

    
class ModelTrainerArtifact:...
class ModelEvaluationArtifact:...
class ModelPusherArtifact:...