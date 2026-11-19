# Trading simulator that uses only the moving average rule (no AI, no news):
# it starts with $10,000, BUYS at a Golden Cross, SELLS at a Death Cross,
# then shows the profit and a chart of the trades.

import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt    # tool for drawing charts


class MAStockSimulation:
    def __init__(self):
        self.stock_data = None
        self.current_index = 0
        self.predictions = []
        # Our "wallet": $10,000 cash, no shares yet, and a list that records every trade
        self.portfolio = {
            'cash': 10000,
            'shares': 0,
            'transactions': []
        }

    def load_data(self):
        self.stock_data = pd.read_csv("apple_stock_test_v2.csv", encoding='utf-8')
        
        # Sort by date to ensure chronological order
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        self.stock_data = self.stock_data.sort_values('Date')
        return self.stock_data
    
    def get_historical_context(self, current_date):
        """Get only data BEFORE current date (no future leakage)"""
        historical_stock = self.stock_data[self.stock_data['Date'] < current_date]
        return historical_stock
    
    def moving_average_algorithm(self, historical_stock, short_window=5, long_window=20, price_column='Open', position_size=1):
        """
        Simple Moving Average Crossover Strategy
        short_window: period for short-term moving average for 5 days
        long_window: period for long-term moving average for 20 days
        price_column: which price to use for calculations
        inital_cash: starting cash for simulation
        position_size: fraction of portfolio to trade each signal
        """
        # Same as ma_example_1.py: the 5-day and 20-day average prices,
        # without the first days that do not have enough days to average yet
        data = historical_stock.copy()
        data['SMA_Short'] = data[price_column].rolling(window=short_window).mean()
        data['SMA_Long'] = data[price_column].rolling(window=long_window).mean()
        data = data.dropna().reset_index(drop=True)

        data = data.reset_index(drop=True)

        cash = self.portfolio['cash']
        shares = self.portfolio['shares']

        # Go through the days one by one, like a trader checking the market every morning
        # (the same Golden Cross / Death Cross rule as ma_example_2.py)
        for i in range(1, len(data)):
            current_short = data.loc[i, 'SMA_Short']
            current_long = data.loc[i, 'SMA_Long']
            prev_short = data.loc[i-1, 'SMA_Short']
            prev_long = data.loc[i-1, 'SMA_Long']
            current_price = data.loc[i, price_column]
            
            # BUY SIGNAL: Golden Cross (short MA crosses ABOVE long MA)
            # ... and we only buy if we still have cash
            if (current_price > current_short and current_short > current_long and prev_short <= prev_long and cash > 0):

                # How many whole shares can we buy with our cash? (// rounds down: no half shares)
                buy_amount = cash * position_size
                shares_to_buy = buy_amount // current_price

                if shares_to_buy > 0:
                    # Pay for the shares: cash goes down, shares go up
                    cost = shares_to_buy * current_price
                    cash -= cost
                    shares += shares_to_buy

                    # Write the trade in our record (used later for the chart)
                    self.portfolio['transactions'].append({
                        'date': data.loc[i, 'Date'],
                        'type': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'total': cost
                    })
                    print(f"BUY: {shares_to_buy} shares at ${current_price:.2f} = ${cost:.2f}")
            
            # SELL SIGNAL: Death Cross (short MA crosses BELOW long MA)
            # SELL ALL SHARES
            # ... and we only sell if we own shares
            elif (current_price < current_short and current_short < current_long and prev_short >= prev_long and shares > 0):

                sell_amount = shares  # SELL ALL shares, not partial
                revenue = sell_amount * current_price   # money we get for the shares
                cash += revenue
                
                self.portfolio['transactions'].append({
                    'date': data.loc[i, 'Date'],
                    'type': 'SELL',
                    'shares': sell_amount,
                    'price': current_price,
                    'total': revenue
                })
                print(f"SELL ALL: {sell_amount} shares at ${current_price:.2f} = ${revenue:.2f}")
                
                shares = 0  # Reset shares to 0 after selling everything
            
            # LAST DAY: Liquidate all positions
            # (sell any shares we still own, so the result is all in cash and easy to compare)
            elif i == len(data) - 1 and shares > 0:
                sell_amount = shares
                revenue = sell_amount * current_price
                cash += revenue
                
                self.portfolio['transactions'].append({
                    'date': data.loc[i, 'Date'],
                    'type': 'SELL (Final)',
                    'shares': sell_amount,
                    'price': current_price,
                    'total': revenue
                })
                print(f"FINAL SELL: {sell_amount} shares at ${current_price:.2f} = ${revenue:.2f}")
                shares = 0
        
        # Update portfolio
        self.portfolio['cash'] = cash
        self.portfolio['shares'] = shares
        
        # Profit = what we have at the end minus the $10,000 we started with
        final_value = cash + (shares * data.loc[len(data)-1, price_column])
        profit = final_value - 10000
        profit_pct = (profit / 10000) * 100
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"Initial Capital: ${cash:.2f}")
        print(f"Final Value: ${final_value:.2f}")
        print(f"Profit/Loss: ${profit:.2f} ({profit_pct:+.2f}%)")
        print(f"Total Transactions: {len(self.portfolio['transactions'])}")
        print(f"{'='*60}")
        
        return data

    # Draw the chart: the price line, the two moving averages,
    # and a green ▲ for every BUY and a red ▼ for every SELL
    def ploting_results(self, data):
        plt.figure(figsize=(14, 7))
        plt.plot(data['Date'], data['Open'], label='Open Price', color='blue', alpha=0.5)
        plt.plot(data['Date'], data['SMA_Short'], label='5-Day SMA', color='orange')
        plt.plot(data['Date'], data['SMA_Long'], label='20-Day SMA', color='red')
        
        # Track if we've already added labels
        buy_label_added = False
        sell_label_added = False
        
        # Mark buy/sell signals
        for transaction in self.portfolio['transactions']:
            if transaction['type'] == 'BUY':
                # Only add label for the first buy signal
                if not buy_label_added:
                    plt.scatter(transaction['date'], transaction['price'], 
                            marker='^', color='green', s=100, label='Buy Signal')
                    buy_label_added = True
                else:
                    plt.scatter(transaction['date'], transaction['price'], 
                            marker='^', color='green', s=100)  
            
            elif 'SELL' in transaction['type']:
                # Only add label for the first sell signal
                if not sell_label_added:
                    plt.scatter(transaction['date'], transaction['price'], 
                            marker='v', color='red', s=100, label='Sell Signal')
                    sell_label_added = True
                else:
                    plt.scatter(transaction['date'], transaction['price'], 
                            marker='v', color='red', s=100)  
        
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title('Apple Stock - Moving Average Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Run the simulation: load the prices, trade with the moving average rule, then show the chart
if __name__ == "__main__":
    ma_sim = MAStockSimulation()
    ma_sim.load_data()
    ma_sim.get_historical_context(datetime(2026, 10, 28))
    a = ma_sim.moving_average_algorithm(historical_stock=ma_sim.stock_data)
    ma_sim.ploting_results(data=a)
