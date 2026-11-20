# Calculate the 5-day and 20-day moving averages (MA) of Apple's share price.
# A moving average is the average price of the last few days. It smooths out
# the daily ups and downs so we can see the trend.

import pandas as pd
from datetime import datetime

def load_data():
    # Read Apple's daily prices from the CSV file (made by stock_data_generator.py)
    stock_data = pd.read_csv("apple_stock_test_v2.csv", encoding='utf-8')
    # Sort by date to ensure chronological order
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.sort_values('Date')
    return stock_data


def get_historical_context(current_date, stock_data):
        """Get only data BEFORE current date (no future leakage)"""
        # Keep only the days before current_date. A real trader cannot see
        # tomorrow's price, so our program must not use it either.
        historical_stock = stock_data[stock_data['Date'] < current_date]
        return historical_stock


def data_frame(historical_stock, short_window=5, long_window=20, price_column='Open'):
        data = historical_stock.copy()
        # Moving averages: for each day, the average opening price of
        #   SMA_Short = the last 5 days  (short-term trend, reacts fast)
        #   SMA_Long  = the last 20 days (long-term trend, reacts slowly)
        # If SMA_Short is above SMA_Long, the price trend is UP; below means DOWN.
        data['SMA_Short'] = data[price_column].rolling(window=short_window).mean()
        data['SMA_Long'] = data[price_column].rolling(window=long_window).mean()
        # The first rows show NaN (empty): there are not yet 5 (or 20) days to average
        print(f'df with ma: \n{data.head(22)}')
        print()

        # Remove the rows with empty averages, so every row has both averages
        data = data.dropna().reset_index(drop=True)
        print(f'Drop NA df: \n{data}')

        # Also remove the first 2 remaining rows
        data = data.drop(data.index[0:2])
        data = data.reset_index(drop=True)

        return data


if __name__ == "__main__":
        pre_data = load_data()
        # Pretend today is 28 Oct 2026: use only the prices before this day
        result = get_historical_context(datetime(2026, 10, 28), pre_data)
        print(result)
        result2 = data_frame(result)
