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
    transformed_train_path :str
    transformed_test_path:str
    target_encoder_path:str

    
@dataclass
class ModelTrainerArtifact:
    model_trainer_dir:str
    model_path:str
    f1_train_score:float
    f1_test_score:float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float

@dataclass
class ModelPusherArtifact:
    pusher_model_dir:str
    saved_model_dir:str