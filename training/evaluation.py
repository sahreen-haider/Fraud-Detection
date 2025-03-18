import logging
from data_loader import df
from sklearn.metrics import classification_report, accuracy_score
from joblib import load
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(filename='training/eval.log', level=logging.INFO, format="%(levelname)s - %(asctime)s - %(module)s - %(funcName)s - %(message)s")

def load_model(model_path):
    """Load the trained model from a file."""
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.warning(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return None


def split_features_and_target(dataframe, target_column='target'):
    """Split the dataframe into features and target."""
    if target_column not in dataframe.columns:
        raise ValueError(f"The dataframe does not contain the '{target_column}' column.")
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    return X, y


def evaluate_model(model, X, y):
    """Evaluate the model on precision, recall, F1 score, and accuracy."""
    y_pred = model.predict(X)
    precision = precision_score(y, y_pred, pos_label=1)
    recall = recall_score(y, y_pred, pos_label=1)
    f1 = f1_score(y, y_pred, pos_label=1)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=['Class 0', 'Class 1'])
    
    return precision, recall, f1, accuracy, report


def main():
    # Load the trained model
    model_path = 'saved models\lgbm_model.pkl'
    model = load_model(model_path)

    # Split the data into features and target
    X, y = split_features_and_target(dataframe=df, target_column='is_fraud')

    # Evaluate the model
    precision, recall, f1, accuracy, report = evaluate_model(model, X, y)

    # Print evaluation metrics
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:")
    logging.info(report)


if __name__ == "__main__":
    main()