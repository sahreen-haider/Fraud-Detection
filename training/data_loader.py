import pickle
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Define feature columns and label
feature_columns = ['cc_num', 'merchant', 'category', 'amt', 'unix_time', 'merch_lat', 'merch_long', 'hour', 'day_of_week']
label_column = "is_fraud"

# Load dataset directly with Pandas (instead of tf.data)
df = pd.read_csv("data/fraudTrain_Feature_engineered.csv")

# Ensure only the required columns are used
df = df[feature_columns + [label_column]]

# 🔹 Encode Categorical Columns and Save Encoders
def encode_categorical_columns(dataframe, columns):
    encoders = {}
    categorical_columns = dataframe.select_dtypes(include=['object']).columns.intersection(columns)
    
    for column in categorical_columns:
        le = LabelEncoder()
        dataframe[column] = le.fit_transform(dataframe[column])
        encoders[column] = le  # Store encoder for later use
    
    # Save encoders
    with open("saved models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    
    return dataframe

df = encode_categorical_columns(df, ['merchant', 'category'])

# 🔹 Resample Data Using SMOTETomek
def resample_data(dataframe):
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(dataframe.drop(columns=[label_column]), dataframe[label_column])
    
    df_resampled = pd.DataFrame(X_resampled, columns=X_resampled.columns)
    df_resampled[label_column] = y_resampled  # Add label back
    
    print("Class Distribution after resampling:")
    print(df_resampled[label_column].value_counts())
    
    return df_resampled

df = resample_data(df)

# 🔹 Scale Numerical Features and Save Scaler
def scale_features(dataframe, num_cols):
    scaler = MinMaxScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    # Save the scaler
    with open("saved models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    return dataframe

# List of numerical columns
numerical_columns = ['amt', 'cc_num', 'unix_time', 'merch_lat', 'merch_long', 'hour']


df = scale_features(df, numerical_columns)

# 🔹 Convert to TensorFlow Dataset for Model Training
# def df_to_tf_dataset(dataframe, batch_size=32):
#     labels = dataframe.pop(label_column)
#     dataset = tf.data.Dataset.from_tensor_slices((dataframe.to_dict(orient="list"), labels))
#     dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
#     return dataset