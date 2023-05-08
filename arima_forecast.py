# imports
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def get_data(ticker: str) -> pd.Series:
    
    # initialize Ticker object
    stock = yf.Ticker(ticker)
    
    # get Closing price dataframe
    df = stock.history(start="2016-01-01")[["Close"]]
    
    # create the log series for TSA 
    ser = np.log(df.Close)
    
    return ser
    
def train_test_split(data: pd.Series) -> pd.DataFrame:
    
    # creates size for train/test split
    size = int(np.round(len(data) * 0.75, 2))
    
    # splits the data into training and testing sets
    train, test = data[:size], data[size:]
    
    return train, test

def fit(train: pd.Series, test: pd.Series) -> tuple(np.array, ARIMA):
    
    # initializes historical data
    hist = [x for x in train]
    
    # creates empty list for predictions
    preds = []
    
    # generates predictions
    for t in range(len(test)):
        
        # reinitializes ARIMA model as we update historical data
        model = ARIMA(hist, order=(5, 2, 0))
        
        # fits model
        fitted_model = model.fit()
        
        # generates prediction
        pred = fitted_model.forecast()
        
        # appends prediction to prediction list
        preds.append(pred[0])
        
        # updates historical data
        hist.append(test[t])
        
    return preds, fitted_model

def rmse(obs: np.array, preds: np.array) -> float:
    
    # calculates rmse
    
    return np.sqrt(np.mean((obs - preds) ** 2))

def generate_forecasts(obs: pd.Series, preds: np.array, model: ARIMA, n: int) -> pd.DataFrame:
    
    # creates dataframe from obs data
    df = np.exp(obs).to_frame()
    
    # adds prediction data to our dataframe
    df["Predictions"] = np.exp(preds)
    
    # forecasts n steps 
    forecast = np.exp(model.forecast(n))
    for i in range(5):
        df.loc[df.index[-1] + pd.Timedelta('1day')] = [np.NaN, forecast[i]]
        
    return df