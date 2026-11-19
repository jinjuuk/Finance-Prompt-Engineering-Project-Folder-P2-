# A simple trading rule that uses only the moving averages (no AI, no news):
# BUY at a "Golden Cross" and SELL at a "Death Cross".

from datetime import datetime
# Reuse the data loading from ma_example_1.py
from ma_example_1 import load_data, get_historical_context

def moving_average_algorithm(historical_stock, short_window=5, long_window=20, price_column='Open'):

    # Same as ma_example_1.py: the 5-day and 20-day average prices
    data = historical_stock.copy()
    data['SMA_Short'] = data[price_column].rolling(window=short_window).mean()
    data['SMA_Long'] = data[price_column].rolling(window=long_window).mean()
    data = data.dropna().reset_index(drop=True)

    data = data.drop(data.index[0:2])
    data = data.reset_index(drop=True)
    print(data)

    # Go through the days one by one and compare each day with the day before
    for i in range(1, len(data)):
        current_short = data.loc[i, 'SMA_Short']    # today's 5-day average
        current_long = data.loc[i, 'SMA_Long']      # today's 20-day average
        prev_short = data.loc[i-1, 'SMA_Short']     # yesterday's 5-day average
        prev_long = data.loc[i-1, 'SMA_Long']       # yesterday's 20-day average
        current_price = data.loc[i, price_column]   # today's price

        # BUY SIGNAL: Golden Cross (short MA crosses ABOVE long MA)
        # Yesterday the 5-day average was at or below the 20-day average, today it is above,
        # and today's price is above both: the price has started to go up.
        if (current_price > current_short and current_short > current_long and prev_short <= prev_long):
            print(f"[{i}]: Golden Cross => Buy Stock")

        # SELL SIGNAL: Death Cross (short MA crosses BELOW long MA)
        # Yesterday the 5-day average was at or above the 20-day average, today it is below,
        # and today's price is below both: the price has started to go down.
        elif (current_price < current_short and current_short < current_long and prev_short >= prev_long):
            print(f"[{i}]: Death Cross => Sell Stock")
        # On all other days nothing is printed: no signal, so we HOLD
    return data

if __name__ == "__main__":
    pre_data = load_data()
    # Pretend today is 28 Oct 2026: use only the prices before this day
    result = get_historical_context(datetime(2026, 10, 28), pre_data)
    result2 = moving_average_algorithm(result)

