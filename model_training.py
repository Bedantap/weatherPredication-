import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Example preprocessing steps
    data = data.dropna()  # Remove missing values
    X = data.drop('target_column', axis=1)  # Features
    y = data['target_column']  # Target variable
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def save_model(model, filename):
    joblib.dump(model, filename)

def main():
    data = load_data('../data/india_weather_data.csv')
    X, y = preprocess_data(data)
    model, X_test, y_test = train_model(X, y)
    save_model(model, 'weather_model.pkl')

if __name__ == "__main__":
    main()