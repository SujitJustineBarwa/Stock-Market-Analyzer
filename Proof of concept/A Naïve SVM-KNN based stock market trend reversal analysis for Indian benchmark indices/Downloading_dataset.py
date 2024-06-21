import yfinance as yf
import pandas as pd

# Define the tickers and date ranges
sensex_ticker = "^BSESN"
nifty_ticker = "^NSEI"
sensex_start_date = "1990-01-01"
sensex_end_date = "2014-08-31"
nifty_start_date = "1994-01-01"
nifty_end_date = "2014-08-31"

# Download the data
sensex_data = yf.download(sensex_ticker, start=sensex_start_date, end=sensex_end_date)
nifty_data = yf.download(nifty_ticker, start=nifty_start_date, end=nifty_end_date)

# Save to CSV files
sensex_data.to_csv("sensex_1990_to_2014.csv")
nifty_data.to_csv("nifty_1994_to_2014.csv")

print("Data downloaded and saved to CSV files.")
