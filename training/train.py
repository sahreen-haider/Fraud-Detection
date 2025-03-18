import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import *
from logger_config import logger, set_log_file
# Configure Logging

# Set log file for the training process
set_log_file("training.log")
logging = logger 
# 🔹 Function to Compute Class Weights
def compute_class_weights(y):
    """
    Compute class weights for imbalanced datasets.

    Args:
        y (array-like): Target variable.

    Returns:
        dict: Class weights.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


# 🔹 Function to Save the Trained Model
def save_model(model, filename="saved models/lgbm_model.pkl"):
    """
    Saves the trained model to a file.

    Args:
        model: Trained model object.
        filename (str): File name to save the model.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


# 🔹 Function to Train LGBM Model
def train_lgbm(X, y, class_weights):
    """
    Trains an LGBMClassifier with GPU support.

    Args:
        X (DataFrame): Features.
        y (Series): Labels.
        class_weights (dict): Computed class weights.

    Returns:
        model: Trained LGBM model.
    """
    # Compute scale_pos_weight for LGBM (only for binary classification)
    scale_pos_weight = class_weights[1] / class_weights[0]

    # Initialize and train LGBM model with GPU support
    model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, device='gpu')
    model.fit(X, y)

    return model


# Define Features and Target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Compute Class Weights
class_weights = compute_class_weights(y)

# Train LGBM Model
model = train_lgbm(X, y, class_weights)

# Save the Trained Model
save_model(model)

logging.info("Training completed successfully, Model saved to 'saved models/lgbm_model.pkl'")
