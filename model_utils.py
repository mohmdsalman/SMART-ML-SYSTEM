# Import necessary libraries
import os  # For interacting with the operating system (e.g., file and directory handling)
import pandas as pd  # For reading and processing the dataset (CSV files)

# Import relevant functions from PyCaret for classification and regression tasks
from pycaret.classification import (
    setup as classification_setup,  # To set up the environment for classification
    compare_models as classification_compare,  # To compare different classification models
    save_model as save_classification_model,  # To save the best classification model
    get_config as get_classification_config,  # To get configuration settings for the classification setup
    pull as classification_pull,  # To pull the results of the model comparison
)

from pycaret.regression import (
    setup as regression_setup,  # To set up the environment for regression
    compare_models as regression_compare,  # To compare different regression models
    save_model as save_regression_model,  # To save the best regression model
    get_config as get_regression_config,  # To get configuration settings for the regression setup
    pull as regression_pull,  # To pull the results of the model comparison
)

# Define the function to train a model based on the provided dataset and target column
def train_model(file_path, target_column, problem_type):
    # Load the dataset from the provided file path (CSV format)
    data = pd.read_csv(file_path)

    # Check if the target column exists in the dataset
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")  # Raise an error if the target column is not found

    # Ensure the 'models/' directory exists to save the trained models later
    models_dir = os.path.join("static", "models")
    os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Check if the problem type is classification
    if problem_type == 'classification':
        # Set up the classification environment using PyCaret
        classification_setup(data=data, target=target_column, verbose=False, session_id=123)
        
        # Compare all available classification models and select the best one
        best_model = classification_compare()
        
        # Get the leaderboard dataframe of the classification models (sorted by performance)
        leaderboard_df = classification_pull()
        
        # Convert the leaderboard dataframe to an HTML table to display on the frontend
        leaderboard_html = leaderboard_df.to_html(classes='table table-striped', index=False)
        
        # Get the selected features (input variables) from the classification model
        selected_features = get_classification_config('X_train').columns.tolist()
        
        # Save the best classification model to the 'models/' directory
        save_classification_model(best_model, os.path.join(models_dir, 'best_model'))

    # Check if the problem type is regression
    elif problem_type == 'regression':
        # Set up the regression environment using PyCaret
        regression_setup(data=data, target=target_column, verbose=False, session_id=123)
        
        # Compare all available regression models and select the best one
        best_model = regression_compare()
        
        # Get the leaderboard dataframe of the regression models (sorted by performance)
        leaderboard_df = regression_pull()
        
        # Convert the leaderboard dataframe to an HTML table to display on the frontend
        leaderboard_html = leaderboard_df.to_html(classes='table table-striped', index=False)
        
        # Get the selected features (input variables) from the regression model
        selected_features = get_regression_config('X_train').columns.tolist()
        
        # Save the best regression model to the 'models/' directory
        save_regression_model(best_model, os.path.join(models_dir, 'best_model'))

    # If the problem type is neither 'classification' nor 'regression', raise an error
    else:
        raise ValueError(f"Unsupported problem type: '{problem_type}'. Use 'classification' or 'regression'.")

    # Return the training results as a dictionary
    return {
        "leaderboard": leaderboard_html,  # The leaderboard table in HTML format
        "best_model": str(best_model),  # The best model (converted to a string)
        "selected_features": selected_features,  # List of selected features used in the model
        "details": "Model training completed and saved successfully!"  # Status message indicating success
    }
