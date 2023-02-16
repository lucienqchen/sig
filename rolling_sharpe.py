import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def rolling_sharpe(start: dt.datetime, end: dt.datetime, ticker: str, window: int) -> pd.Series:
    
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
    
    return sharpe

def plot_rs(ticker: str, rs_ser: pd.Series) -> None:
    
    #Creates parameters for scaling the figure
    
    years = pd.to_datetime(rs_ser.index).year
    width = (years.max() - years.min()) * 2
    height= abs(rs_ser.max()) + abs(rs_ser.min())
    
    #Plots Rolling Sharpe with figsize
    
    fig = rs_ser.plot(figsize=(width, height))
    
    #Adds title, labels, and horizontal line at 0
    
    fig.set_title(f"{ticker} Sharpe Ratio")
    fig.set_ylabel("Sharpe Ratio")
    fig.axhline(y=0, color='black', linestyle='-', alpha=0.5)