from data_preprocessing import load_and_preprocess_data
from model_training import train_model
from prediction import make_predictions

def main():
    # Load and preprocess the data
    data = load_and_preprocess_data('data/india_weather_data.csv')
    
    # Train the model
    model = train_model(data)
    
    # Make predictions
    predictions = make_predictions(model, data)
    
    # Output predictions
    print(predictions)

if __name__ == "__main__":
    main()