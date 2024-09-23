from sensor.pipeline import training_pipeline
from sensor.exception import SensorException
import sys
from sensor.pipeline import batch_prediction

print(__name__)

if __name__ =="__main__":
    try:
        # training_pipeline.start_training_pipeline()
        output_file = batch_prediction.start_batch_prediction("aps_failure_training_set1.csv")

    except Exception as e:
        raise SensorException(e, sys)