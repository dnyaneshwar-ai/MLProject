# Import necessary libraries and modules
import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

# Define a data class for configuration related to data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

# Define a class for data transformation
class DataTransformation:
    def __init__(self):
        # Initialize an instance of the DataTransformationConfig class
        self.data_transformation_config = DataTransformationConfig()

    # Define a method to obtain the data transformer object
    def get_data_transformer_object(self):
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Define a numerical pipeline with imputation and standard scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Define a categorical pipeline with imputation, one-hot encoding, and standard scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Log information about the columns
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Create a ColumnTransformer with both numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            # Return the preprocessor object
            return preprocessor
        
        except Exception as e:
            # Raise a custom exception in case of an error
            raise CustomException(e, sys)

    # Define a method to initiate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read training and testing data into pandas DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Log information about reading data
            logging.info("Read train and test data completed")

            # Log information about obtaining the preprocessing object
            logging.info("Obtaining preprocessing object")

            # Get the data transformer object using the defined method
            preprocessing_obj = self.get_data_transformer_object()

            # Specify the target column name and numerical columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Extract input and target features from the training and testing data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Log information about applying the preprocessing object
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Transform input features using the preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target features into arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Log information about saving the preprocessing object
            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing object using the save_object utility function
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed datasets and the path to the saved preprocessing object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Raise a custom exception in case of an error
            raise CustomException(e, sys)
