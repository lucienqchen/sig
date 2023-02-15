import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib as plt
plt.style.use("ggplot")

def rolling_sharpe(start=dt.datetime(2016, 1, 1), end=dt.datetime.now(), ticker=None, window=20):
    
    RISK_FREE_RATE = yf.Ticker("^TYX").history()["Close"][-1] / 100 / 252
    # Gets closing prices of ticker for the past 
    close = yf.Ticker(ticker).history(start=start, end=end)["Close"]
    
    # Shifts closing prices by 1 day to get log returns; dropna drops the first value since there 
    # is no preceding number
    
    returns = close / close.shift(1).dropna()
    log_returns = np.log(returns)
    
    #Calculates standard deviation on an X rolling period and multiplies by sqrt to get volatility
    volatility = log_returns.rolling(window=window).std() * np.sqrt(window)
    
    #Calculate Rolling Sharpe
    sharpe = (log_returns.rolling(window=window).mean() - RISK_FREE_RATE) * window / volatility
    
    #Plots Rolling Sharpe with labels
    fig = sharpe.plot()
    fig.set_title(f"{ticker} Sharpe Ratio")
    fig.set_ylabel("Sharpe Ratio")    
    
    return fig