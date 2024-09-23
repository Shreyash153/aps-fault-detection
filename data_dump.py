import pymongo
import pandas as pd
import json
import os
from sensor.config import mongo_client


# client = pymongo.MongoClient(MONGO_DB_URL)

DATA_FILE_PATH = "aps_failure_training_set1.csv"
DATABASE_NAME = "aps"
COLLECTION_NAME = "sensor"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns : {df.shape}")

    #convert dataframe to json so that we can dump data in mongo db
    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    #insert converted json record to mongo db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
