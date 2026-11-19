# Download Apple's daily share prices from Yahoo Finance and show them as a table.
import yfinance as yf    # tool that downloads stock prices from Yahoo Finance
import pandas as pd      # tool for working with tables of data


# "AAPL" is Apple's short name (ticker symbol) on the stock market
apple = yf.Ticker("AAPL")

# Get one price row per day (interval='1d') for one year
stock_data = apple.history(start="2025-10-29", end="2026-10-29", interval='1d')
stock_df = pd.DataFrame(stock_data)


# Turn the dates into a normal column called 'Date', so we can use them like any other column
stock_df = stock_df.reset_index()


df_datetime = stock_df.copy()


# Remove the time zone (e.g. "-05:00" for New York) from the dates
df_datetime['Datetime'] = df_datetime['Date'].dt.tz_localize(None)

# Split the date and the time into two separate columns
df_datetime['Date'] = df_datetime['Datetime'].dt.date
df_datetime['Time'] = df_datetime['Datetime'].dt.time
df_drop_datetime = df_datetime.drop('Datetime', axis=1)   # the combined column is not needed any more

# Show the final table: Date, Open, High, Low, Close, Volume, ... and Time
print(df_drop_datetime)
