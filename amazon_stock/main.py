import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import os
import datetime
import wandb
from dotenv import load_dotenv  

# Load environment variables from .env file
load_dotenv()

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the path to save the model and results
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load the dataset
def load_data(file_path):
    """
    Load the Amazon stock data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    # Load the actual data using pd.read_csv(file_path)
    df = pd.read_csv(file_path)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# Define function to prepare the data for the model
def prepare_data(data, feature='Close', look_back=60, forecast_horizon=1, train_size=0.8):
    """
    Prepare the data for training and testing
    
    Args:
        data: DataFrame containing the stock data
        feature: Feature to use for prediction (default: 'Close')
        look_back: Number of previous time steps to use for prediction
        forecast_horizon: Number of steps to forecast into the future
        train_size: Proportion of data to use for training
        
    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    # Extract the feature to be used for prediction
    dataset = data[feature].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    
    # Create the data with look_back time steps
    X, y = [], []
    for i in range(len(dataset_scaled) - look_back - forecast_horizon + 1):
        X.append(dataset_scaled[i:(i + look_back), 0])
        y.append(dataset_scaled[i + look_back:(i + look_back + forecast_horizon), 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split the data into training and testing sets
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler, dataset

# Define function to create the LSTM model
def create_lstm_model(look_back, forecast_horizon, units=50, dropout=0.2):
    """
    Create an LSTM model for time series forecasting
    
    Args:
        look_back: Number of previous time steps to use for prediction
        forecast_horizon: Number of steps to forecast into the future
        units: Number of LSTM units
        dropout: Dropout rate
        
    Returns:
        Compiled LSTM model
    """
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(units=units, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(forecast_horizon))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Define function to create a bidirectional GRU model
def create_bidirectional_gru_model(look_back, forecast_horizon, units=50, dropout=0.2):
    """
    Create a bidirectional GRU model for time series forecasting
    
    Args:
        look_back: Number of previous time steps to use for prediction
        forecast_horizon: Number of steps to forecast into the future
        units: Number of GRU units
        dropout: Dropout rate
        
    Returns:
        Compiled bidirectional GRU model
    """
    model = Sequential()
    
    # Bidirectional GRU layers
    model.add(Bidirectional(GRU(units=units, return_sequences=True), input_shape=(look_back, 1)))
    model.add(Dropout(dropout))
    
    model.add(Bidirectional(GRU(units=units, return_sequences=False)))
    model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(forecast_horizon))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Define function to train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    Train the model
    
    Args:
        model: Model to train
        X_train: Training data
        y_train: Training labels
        X_test: Validation data
        y_test: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Trained model and training history
    """
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(os.path.join(results_dir, 'best_model.h5'), 
                                    save_best_only=True, monitor='val_loss')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

# Define function to evaluate the model
def evaluate_model(model, X_test, y_test, scaler, dataset):
    """
    Evaluate the model on the test data
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
        scaler: Scaler used to normalize the data
        dataset: Original dataset for plotting
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform the predictions and actual values
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv.reshape(-1), y_pred_inv.reshape(-1))
    
    # Print metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv.reshape(-1), label='Actual')
    plt.plot(y_pred_inv.reshape(-1), label='Predicted')
    plt.title('Amazon Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'prediction_vs_actual.png'))
    plt.close()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Define function to perform K-Fold cross validation
def perform_kfold_cv(data, feature='Close', look_back=60, forecast_horizon=1, 
                    k=5, epochs=50, batch_size=32, model_type='lstm'):
    """
    Perform K-Fold cross validation
    
    Args:
        data: DataFrame containing the stock data
        feature: Feature to use for prediction
        look_back: Number of previous time steps to use for prediction
        forecast_horizon: Number of steps to forecast into the future
        k: Number of folds
        epochs: Number of epochs
        batch_size: Batch size
        model_type: Type of model to use ('lstm' or 'gru')
        
    Returns:
        Dictionary containing evaluation metrics for each fold
    """
    # Extract the feature to be used for prediction
    dataset = data[feature].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    
    # Create the data with look_back time steps
    X, y = [], []
    for i in range(len(dataset_scaled) - look_back - forecast_horizon + 1):
        X.append(dataset_scaled[i:(i + look_back), 0])
        y.append(dataset_scaled[i + look_back:(i + look_back + forecast_horizon), 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize metrics
    metrics = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': []
    }
    
    # Perform K-Fold cross validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/{k}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create model
        if model_type == 'lstm':
            model = create_lstm_model(look_back, forecast_horizon)
        else:
            model = create_bidirectional_gru_model(look_back, forecast_horizon)
        
        # Train model
        model, _ = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        # Inverse transform the predictions and actual values
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv.reshape(-1), y_pred_inv.reshape(-1))
        
        # Store metrics
        metrics['mse'].append(mse)
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['r2'].append(r2)
        
        # Print metrics for this fold
        print(f"Fold {fold+1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Calculate mean and std of metrics
    for metric in metrics:
        print(f"\nMean {metric.upper()}: {np.mean(metrics[metric]):.4f} ± {np.std(metrics[metric]):.4f}")
    
    return metrics

# Define function for future prediction
def predict_future(model, last_sequence, scaler, steps=30):
    """
    Predict future stock prices
    
    Args:
        model: Trained model
        last_sequence: Last sequence from the dataset
        scaler: Scaler used to normalize the data
        steps: Number of future steps to predict
        
    Returns:
        Array of predicted future prices
    """
    future_predictions = []
    
    # Make a copy of the last sequence
    curr_sequence = last_sequence.copy()
    
    # Predict future steps
    for _ in range(steps):
        # Reshape for prediction
        curr_sequence_reshaped = curr_sequence.reshape(1, -1, 1)
        
        # Predict next step
        next_pred = model.predict(curr_sequence_reshaped)[0]
        
        # Add prediction to the list
        future_predictions.append(next_pred[0])
        
        # Update sequence for next prediction
        curr_sequence = np.append(curr_sequence[1:], next_pred[0])
    
    # Convert predictions to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions

# Define main function
def main():
    """
    Main function to run the stock price prediction pipeline
    """
    # Initialize WandB
    wandb_username = os.getenv("WANDB_USERNAME")  # Load username from .env
    wandb.init(project="amazon-stock-prediction", entity=wandb_username)

    # Load data
    file_path = '/Users/dimasmulya/Documents/Submission-Inosoft/amazon_stock/evaluating_models/data/Test Macine Learning Enginer Inosoft_AMZN.csv'  
    try:
        # Load the actual data file
        data = load_data(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Parameters
    feature = 'Close'
    look_back = 60  # Number of days to look back
    forecast_horizon = 1  # Number of days to forecast
    train_size = 0.8
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler, dataset = prepare_data(
        data, feature, look_back, forecast_horizon, train_size
    )
    
    # Create and train LSTM model
    print("\nTraining LSTM model...")
    lstm_model = create_lstm_model(look_back, forecast_horizon)
    lstm_model, lstm_history = train_model(lstm_model, X_train, y_train, X_test, y_test)
    
    # Evaluate LSTM model
    print("\nEvaluating LSTM model...")
    lstm_metrics = evaluate_model(lstm_model, X_test, y_test, scaler, dataset)
    
    # Log LSTM metrics to WandB
    wandb.log({"LSTM MSE": lstm_metrics['mse'],
                "LSTM RMSE": lstm_metrics['rmse'],
                "LSTM MAE": lstm_metrics['mae'],
                "LSTM R²": lstm_metrics['r2']})
    
    # Create and train GRU model
    print("\nTraining Bidirectional GRU model...")
    gru_model = create_bidirectional_gru_model(look_back, forecast_horizon)
    gru_model, gru_history = train_model(gru_model, X_train, y_train, X_test, y_test)
    
    # Evaluate GRU model
    print("\nEvaluating Bidirectional GRU model...")
    gru_metrics = evaluate_model(gru_model, X_test, y_test, scaler, dataset)
    
    # Log GRU metrics to WandB
    wandb.log({"GRU MSE": gru_metrics['mse'],
                "GRU RMSE": gru_metrics['rmse'],
                "GRU MAE": gru_metrics['mae'],
                "GRU R²": gru_metrics['r2']})

    # Compare models
    print("\nModel Comparison:")
    print(f"LSTM - RMSE: {lstm_metrics['rmse']:.4f}, R²: {lstm_metrics['r2']:.4f}")
    print(f"GRU - RMSE: {gru_metrics['rmse']:.4f}, R²: {gru_metrics['r2']:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
    plt.plot(gru_history.history['loss'], label='GRU Training Loss')
    plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    plt.close()
    
    # Choose the best model
    if lstm_metrics['rmse'] < gru_metrics['rmse']:
        best_model = lstm_model
        print("\nLSTM model performed better and will be used for future predictions.")
    else:
        best_model = gru_model
        print("\nGRU model performed better and will be used for future predictions.")
    
    # K-Fold cross validation
    print("\nPerforming K-Fold cross validation...")
    kfold_metrics = perform_kfold_cv(data, feature, look_back, forecast_horizon)
    
    # Predict future
    print("\nPredicting future stock prices...")
    last_sequence = X_test[-1]
    future_predictions = predict_future(best_model, last_sequence, scaler)
    
    # Create future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_predictions))
    
    # Plot future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(data[feature][-100:], label='Historical')
    plt.plot(future_dates, future_predictions, label='Predicted')
    plt.title('Amazon Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'future_predictions.png'))
    plt.close()
    
    # Save the best model
    best_model.save(os.path.join(results_dir, 'best_model.h5'))
    
    # Save the results to a text file
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write("Amazon Stock Price Prediction Results\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("LSTM Model Metrics:\n")
        f.write(f"MSE: {lstm_metrics['mse']:.4f}\n")
        f.write(f"RMSE: {lstm_metrics['rmse']:.4f}\n")
        f.write(f"MAE: {lstm_metrics['mae']:.4f}\n")
        f.write(f"R²: {lstm_metrics['r2']:.4f}\n\n")
        
        f.write("GRU Model Metrics:\n")
        f.write(f"MSE: {gru_metrics['mse']:.4f}\n")
        f.write(f"RMSE: {gru_metrics['rmse']:.4f}\n")
        f.write(f"MAE: {gru_metrics['mae']:.4f}\n")
        f.write(f"R²: {gru_metrics['r2']:.4f}\n\n")
        
        f.write("K-Fold Cross Validation Results:\n")
        f.write(f"Mean MSE: {np.mean(kfold_metrics['mse']):.4f} ± {np.std(kfold_metrics['mse']):.4f}\n")
        f.write(f"Mean RMSE: {np.mean(kfold_metrics['rmse']):.4f} ± {np.std(kfold_metrics['rmse']):.4f}\n")
        f.write(f"Mean MAE: {np.mean(kfold_metrics['mae']):.4f} ± {np.std(kfold_metrics['mae']):.4f}\n")
        f.write(f"Mean R²: {np.mean(kfold_metrics['r2']):.4f} ± {np.std(kfold_metrics['r2']):.4f}\n")

    # Finish the WandB run
    wandb.finish()

# Run the main function if the script is run directly
if __name__ == "__main__":
    main()