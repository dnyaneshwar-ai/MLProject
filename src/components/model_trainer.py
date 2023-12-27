# Import necessary libraries and modules
import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Define a data class for configuration related to model training
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# Define a class for model training
class ModelTrainer:
    def __init__(self):
        # Initialize an instance of the ModelTrainerConfig class
        self.model_trainer_config = ModelTrainerConfig()

    # Define a method to initiate the model training process
    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Log an informational message
            logging.info("Split training and test input data")
            # Split the training and test arrays into features (X) and target variable (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define a dictionary of model instances for various algorithms
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define a dictionary of hyperparameters for tuning specific algorithms
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models using the evaluate_models function
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            # Get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # Get the name of the best model from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            # Get the best model instance based on the name
            best_model = models[best_model_name]

            # If the best model score is below a certain threshold, raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            # Log a message indicating the best found model on both training and testing datasets
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions using the best model on the test set
            predicted = best_model.predict(X_test)

            # Calculate and return the R-squared score
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Raise a custom exception in case of an error
            raise CustomException(e, sys)


