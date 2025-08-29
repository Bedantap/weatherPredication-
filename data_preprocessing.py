import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Data cleaning
    data = data.dropna()  # Remove missing values
    data = data[data['temperature'] >= -50]  # Remove unrealistic temperature values

    # Feature extraction
    features = data[['humidity', 'wind_speed', 'precipitation']]
    labels = data['temperature']

    # Normalization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels

def split_data(features, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test