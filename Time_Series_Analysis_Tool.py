import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from statsmodels.tsa.arima.model import ARIMA 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import numpy as np 

def import_data(url):
    """Imports time series data from a URL.

    parameters:
    url (str): URL of the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the time series data.
    """

    data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    print(data.head())
    return data

def plot_data(data):
    """
    Plots time series data.

    parameters:
    data (pd.DataFrame): DataFrame containing the time series data.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Passengers')
    plt.title('Monthly number of Air Passengers')
    plt.xlabel('Date')
    plt.ylabel('Number of Passengers')
    plt.legend()
    plt.show()

def decompose_series(data):
    """
    Decomposes time series data into trend, seasonal, and residual components.

    parameters:
    data (pd.DataFrame): DataFrame containing the time series data.

    Returns:
    DecomposeResult: Object containing the decomposition components.
    """
    decomposition = sm.tsa.seasonal_decompose(data, model='multiplicative')
    decomposition.plot()
    plt.show()
    return decomposition

def forecast(data, steps=12):
    """
    Forecasts future values using ARIMA. 

    Parameters:
    data (pd.DataFrame): DataFrame containing the time series data.
    steps (int): NUmber of steps to forecast.

    Returns:
    pd.Series: Forecasted values.    
    """
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='historical')
    plt.plot(forecast, label='forecast', color='red')
    plt.title('Forecasting Air Passengers')
    plt.xlabel('Date')
    plt.ylabel('Number of Passengers')
    plt.legend()
    plt.show()
    return forecast

def evaluate_forecast(data, forecast_values):
    """
    Evaluates the forecast using MAE, MSE, and RMSE.

    Parameters:
    data (pd.DataFrame): DataFrame containing the historical time series data.
    forecast_values (pd.Series): Forecasted values.

    Returns:
    dict: Dictionary containing the evaluation metrics.
    """
    historical_values = data[-len(forecast_values):].values
    mae = mean_absolute_error(historical_values, forecast_values)
    mse = mean_squared_error(historical_values, forecast_values)
    rmse = np.sqrt(mse)
    
    metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    print(metrics)
    return metrics

# Running the tool
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = import_data(url)
plot_data(data)
decomposition = decompse_seres(data)
forecast_values = forecast(data)
evaluate_forecast(data, forecast_values)