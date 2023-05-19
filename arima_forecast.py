# imports
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def get_data(ticker: str, start: datetime = None) -> pd.Series:
    
    if start is None:
        start = datetime.now() - relativedelta(years=5)
    # initialize Ticker object
    stock = yf.Ticker(ticker)
    
    # get Closing price dataframe
    df = stock.history(start=start)[["Close"]]
    
    # create the price data series for TSA 
    ser = df.Close
    
    return ser
    
def train_test_split(data: pd.Series, train_size: float = 0.75) -> pd.DataFrame:
    
    # creates size for train/test split
    size = int(np.round(len(data) * train_size, 2))
    
    # splits the data into training and testing sets
    train, test = data[:size], data[size:]
    
    return train, test

def fit(train: pd.Series, test: pd.Series) -> tuple[np.array, ARIMA]:
    
    # initializes ARIMA model
    model = ARIMA(train, order=(5, 2, 2))
    
    # fits model
    fitted_model = model.fit()
    
    # generates predictions
    preds = fitted_model.forecast(len(test)).to_numpy()
    
    # creates the full sequence of data by concatenating train and test
    full_data = pd.concat([train, test])
    
    # recreates a final model
    final_model = ARIMA(full_data, order=(5, 2, 2)).fit()
    
    return preds, final_model

def rmse(obs: np.array, preds: np.array) -> float:
    
    # calculates rmse
    
    return np.sqrt(np.mean((obs - preds) ** 2))

def validate(train: pd.Series, test: pd.Series, preds: np.array) -> pd.DataFrame:
    
    # creates dataframe of 
    df = pd.concat([pd.DataFrame(train), pd.DataFrame(test)])
    df["Predictions"] = pd.Series(preds, index=test.index)
    df.plot()
    return df

def generate_forecasts(obs: pd.Series, model: ARIMA, n: int) -> pd.DataFrame:
    
    # creates dataframe from obs data
    df = obs.to_frame()
    
    # adds prediction column to our dataframe
    nulls = np.empty(df.shape[0])
    nulls[:] = np.NaN
    df["Predictions"] = nulls
    
    
    # forecasts n steps 
    fc = model.forecast(n, alpha=0.05).to_numpy()
    for i in range(n):
        df.loc[df.index[-1] + pd.Timedelta('1day')] = [np.NaN, fc[i]]
    
    return df