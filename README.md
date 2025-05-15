# Amazon Stock Price Forecasting using RNN

This repository contains a solution for forecasting Amazon stock prices using Recurrent Neural Networks (RNN), specifically LSTM and GRU models. The primary goal of this project is to develop a predictive model that can accurately forecast future stock prices based on historical data, enabling better investment decisions.

## Project Structure
amazon-stock-forecast/
│
├── main.py # Main script for training and evaluating models
├── amazon-stock-price-predication-using-lstm-a-9678cb.ipynb # Main script for training and evaluating models
├── requirements.txt # Required Python packages
├── README.md # Project documentation
├── data/ # Directory for datasets
│ └── amazon_stock_data.csv # Amazon historical stock data
│
└── results/ # Directory for results and saved models
├── best_model.h5 # Saved best model
├── prediction_vs_actual.png # Plot of predicted vs actual prices
├── training_history.png # Plot of training history
├── future_predictions.png # Plot of future predictions
└── results.txt # Detailed results and metrics


## Requirements
The project requires the following Python packages:
- numpy==1.24.3
- pandas==2.0.2
- matplotlib==3.7.2
- tensorflow==2.14.0
- scikit-learn==1.3.0

## Environment Setup
### Using pip
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS and Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### Using conda
```bash
# Create a conda environment
conda create -n amazon-stock-forecast python=3.9

# Activate the environment
conda activate amazon-stock-forecast

# Install the required packages
conda install numpy pandas matplotlib scikit-learn
conda install tensorflow
```

## Running the Code
Place the Amazon stock data CSV file in the `data/` directory and run the main script:
```bash
python main.py
```

## Model Architecture
This project implements two types of RNN models:
- **Basic RNN (Recurrent Neural Network)**: A standard RNN architecture with:
  - Two SimpleRNN layers with dropout for regularization
  - A dense output layer

- **Stacked RNN**: A deeper RNN architecture with:
  - Three SimpleRNN layers with dropout
  - A dense output layer

Both models are trained to predict the stock price based on the previous 60 days of data.

## Evaluation Metrics
The models are evaluated using the following metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

Additionally, K-Fold cross-validation is performed to assess the models' robustness and generalization capabilities.

## Analysis of Results
### Training and Testing Performance
The models were trained on 80% of the data and tested on the remaining 20%. The performances of both models are compared based on RMSE and R² metrics. Training history plots show the evolution of training and validation loss over epochs.

### Cross-Validation Results
K-Fold cross-validation was performed with k=5 to ensure the models' generalization capabilities. Mean and standard deviation of the metrics across the folds are reported to indicate the stability of the models.

### Future Predictions
Based on the best performing model, future stock prices are predicted for the next 30 days. These predictions are plotted alongside the historical data to visualize the forecasted trend.

### Summary of Results
1. [Predicted vs Actual Prices](amazon_stock/results/prediction_vs_actual.png): The first plot illustrates the predicted stock prices (orange) against the actual prices (blue). The close alignment of the two lines indicates that the model has effectively captured the underlying trends in the data.

2. [Training and Validation Data](amazon_stock/results/Train_Val_Prediction_output.png): The second plot shows the training data (blue), validation data (orange), and predictions (green). This visualization highlights how well the model generalizes to unseen data, with predictions closely following the validation data.

3. [Model Loss](amazon_stock/results/training_history.png): The third plot depicts the training and validation loss for both the LSTM and GRU models over epochs. The LSTM model shows a more stable and lower loss compared to the GRU model, indicating better performance during training.

## Why LSTM?
The LSTM model was chosen for this project due to its ability to capture long-term dependencies in time series data. Unlike traditional RNNs, LSTMs are designed to remember information for long periods, making them particularly effective for tasks like stock price prediction, where past values significantly influence future outcomes. The analysis shows that the LSTM model outperforms the GRU model, achieving lower error metrics and a higher R² value, indicating a better fit for the data.

## Further Improvements
- **Feature Engineering**: Include more features such as technical indicators or sentiment analysis from news articles.
- **Hyperparameter Tuning**: Optimize the model architecture and hyperparameters using techniques like Bayesian optimization.
- **Ensemble Methods**: Combine multiple models to improve prediction accuracy.
- **Attention Mechanisms**: Incorporate attention mechanisms to help the model focus on important time steps.
- **Data Augmentation**: Generate synthetic data to increase the training set size and improve generalization.
