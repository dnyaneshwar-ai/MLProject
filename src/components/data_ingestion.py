# Import necessary libraries and modules
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# Define a data class for configuration related to data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Define a class for data ingestion
class DataIngestion:
    def __init__(self):
        # Initialize an instance of the DataIngestionConfig class
        self.ingestion_config = DataIngestionConfig()

    # Define a method to initiate the data ingestion process
    def initiate_data_ingestion(self):
        # Log an informational message
        logging.info("Entered the data ingestion method or component")
        try:
            # Read a CSV file into a pandas DataFrame
            df = pd.read_csv('notebook\data\stud.csv')
            # Log a message indicating successful DataFrame creation
            logging.info('Read the dataset as a dataframe')

            # Create directories as needed
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the entire dataset to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Log a message indicating the initiation of train-test split
            logging.info("Train-test split initiated")
            # Perform train-test split on the DataFrame
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save the test set to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Log a message indicating the completion of data ingestion
            logging.info("Ingestion of the data is completed")

            # Return the paths of the training and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise a custom exception in case of an error
            raise CustomException(e, sys)

# Check if the script is being run directly
if __name__ == "__main__":
    # Create an instance of the DataIngestion class
    obj = DataIngestion()
    # Call the initiate_data_ingestion method to perform data ingestion
    train_data, test_data = obj.initiate_data_ingestion()

    # Data Transformation
    # Create an instance of the DataTransformation class
    data_transformation = DataTransformation()
    # Call the initiate_data_transformation method to transform the training and test datasets
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Model Trainer
    # Create an instance of the ModelTrainer class
    modeltrainer = ModelTrainer()
    # Print the result of the initiate_model_trainer method
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
