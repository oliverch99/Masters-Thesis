import pickle
import numpy as np

with open("data/model_data.pkl","rb") as file:
    data = pickle.load(file)

with open("model.pkl", "rb") as f:
    X_train_reshaped, X_test_reshaped, y_train, y_test, y_test_proba, y_test_preds, hour_model = pickle.load(f)

df = data[['Time', 'ticker', 'close', 'oneHourWinner']]
y_train_proba = hour_model.predict(X_train_reshaped)
y_train_preds = (y_train_proba > 0.5).astype(int)

df['predicted'] = list(y_train_preds[:,0]) + list(y_test_preds[:,0])


test_start_idx = int(len(df) * 0.8)

# Create the test_df with the last 20% of the data
test_df = df.iloc[test_start_idx:].copy()

import pandas as pd
from datetime import datetime, timedelta

# Initialize variables for the backtesting simulation
initial_cash = 100000  # Starting with $100,000
cash = initial_cash
portfolio = {}  # To track the positions: positive for long, negative for short
pnl_history = []  # To keep track of the portfolio value over time

# Helper functions
def update_portfolio_value(portfolio, close_prices):
    portfolio_value = 0
    for ticker, shares in portfolio.items():
        if ticker in close_prices:
            portfolio_value += shares * close_prices[ticker]
    return portfolio_value

def close_position(ticker, close_price):
    global cash, portfolio
    if ticker in portfolio:
        position = portfolio.pop(ticker)
        if position > 0:  # Long position
            cash += position * close_price
        elif position < 0:  # Short position
            cash -= position * close_price  # Closing a short position

def open_position(ticker, close_price, position_size):
    global cash, portfolio
    required_cash = abs(position_size * close_price)
    if cash >= required_cash:
        if position_size > 0:  # Opening a long position
            portfolio[ticker] = position_size
            cash -= required_cash
        elif position_size < 0:  # Opening a short position
            portfolio[ticker] = position_size
            cash += required_cash

# Run the backtesting simulation
for index, row in test_df.iterrows():
    current_time = row['Time']
    ticker = row['ticker']
    close_price = row['close']
    signal = row['predicted']
    
    if ticker in portfolio:
        position = portfolio[ticker]
        if position > 0:  # Long position
            if close_price >= (1 + 0.01) * abs(position) or signal == 0:
                close_position(ticker, close_price)
        elif position < 0:  # Short position
            if close_price <= (1 - 0.01) * abs(position) or signal == 1:
                close_position(ticker, close_price)
    
    if signal == 1:
        if ticker in portfolio and portfolio[ticker] < 0:  # Cover short position
            close_position(ticker, close_price)
        open_position(ticker, close_price, 100)  # Example position size
    elif signal == 0:
        if ticker in portfolio and portfolio[ticker] > 0:  # Close long position
            close_position(ticker, close_price)
        open_position(ticker, close_price, -100)  # Example position size
    
    close_prices = test_df.set_index('ticker')['close'].to_dict()
    portfolio_value = update_portfolio_value(portfolio, close_prices)
    total_value = cash + portfolio_value
    pnl_history.append({'Time': current_time, 'Cash': cash, 'Portfolio Value': portfolio_value, 'Total Value': total_value})

# Convert PnL history to DataFrame for analysis
pnl_df = pd.DataFrame(pnl_history)

# Display PnL history
import ace_tools as tools; tools.display_dataframe_to_user(name="PnL History", dataframe=pnl_df)




import pandas as pd
from datetime import datetime, timedelta

# Initialize variables for the backtesting simulation
initial_cash = 100000  # Starting with $100,000
cash = initial_cash
portfolio = {}  # To track the positions: positive for long, negative for short
pnl_history = []  # To keep track of the portfolio value over time
transaction_cost = 0  # Assume $10 per trade
last_trade_time = {}  # Track the last trade time for each ticker to limit frequency

# Helper functions
def update_portfolio_value(portfolio, close_prices):
    portfolio_value = 0
    for ticker, shares in portfolio.items():
        if ticker in close_prices:
            portfolio_value += shares * close_prices[ticker]
    return portfolio_value

def close_position(ticker, close_price):
    global cash, portfolio
    if ticker in portfolio:
        position = portfolio.pop(ticker)
        if position > 0:  # Long position
            cash += position * close_price - transaction_cost
        elif position < 0:  # Short position
            cash -= position * close_price - transaction_cost  # Closing a short position

def open_position(ticker, close_price, position_size):
    global cash, portfolio
    required_cash = abs(position_size * close_price) + transaction_cost
    if cash >= required_cash:
        portfolio[ticker] = position_size
        if position_size > 0:  # Opening a long position
            cash -= required_cash
        elif position_size < 0:  # Opening a short position
            cash += required_cash

# Run the backtesting simulation
for index, row in test_df.iterrows():
    current_time = row['Time']
    ticker = row['ticker']
    close_price = row['close']
    signal = row['predicted']
    
    if ticker in last_trade_time and (current_time - last_trade_time[ticker]).total_seconds() < 60:
        continue  # Skip if the last trade was within the last minute

    if ticker in portfolio:
        position = portfolio[ticker]
        if position > 0:  # Long position
            if close_price >= (1 + 0.01) * abs(position) or signal == 0:
                close_position(ticker, close_price)
                last_trade_time[ticker] = current_time
        elif position < 0:  # Short position
            if close_price <= (1 - 0.01) * abs(position) or signal == 1:
                close_position(ticker, close_price)
                last_trade_time[ticker] = current_time
    
    if signal == 1:
        if ticker in portfolio and portfolio[ticker] < 0:  # Cover short position
            close_position(ticker, close_price)
            last_trade_time[ticker] = current_time
        open_position(ticker, close_price, 100)  # Example position size
        last_trade_time[ticker] = current_time
    elif signal == 0:
        if ticker in portfolio and portfolio[ticker] > 0:  # Close long position
            close_position(ticker, close_price)
            last_trade_time[ticker] = current_time
        open_position(ticker, close_price, -100)  # Example position size
        last_trade_time[ticker] = current_time
    
    close_prices = test_df.set_index('ticker')['close'].to_dict()
    portfolio_value = update_portfolio_value(portfolio, close_prices)
    total_value = cash + portfolio_value
    pnl_history.append({'Time': current_time, 'Cash': cash, 'Portfolio Value': portfolio_value, 'Total Value': total_value})

# Convert PnL history to DataFrame for analysis
pnl_df = pd.DataFrame(pnl_history)






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("data/model_data.pkl","rb") as file:
    data = pickle.load(file)

with open("model.pkl", "rb") as f:
    X_train_reshaped, X_test_reshaped, y_train, y_test, y_test_proba, y_test_preds, hour_model = pickle.load(f)

df = data[['Time', 'ticker', 'close', 'oneHourWinner']]
y_train_proba = hour_model.predict(X_train_reshaped)
y_train_preds = (y_train_proba > 0.5).astype(int)

df['predicted'] = list(y_train_preds[:,0]) + list(y_test_preds[:,0])


test_start_idx = int(len(df) * 0.8)

# Create the test_df with the last 20% of the data
test_df = df.iloc[test_start_idx:].copy()




class BacktestingEngine:
    def __init__(self, data):
        self.data = data
        self.portfolio = {'cash': 1000000, 'positions': {}}
        self.trades = []
        self.risk_per_trade = 0.01 # Risk 1% of portfolio per trade
    def run_backtest(self):

        for index, row in self.data.iterrows():
            close_price = row['close']
            signal = row['predicted']

            if signal == 1:
                # Buy signal
                available_cash = self.portfolio['cash']
                position_size = available_cash * self.risk_per_trade / close_price
                self.execute_trade(symbol=row['ticker'], quantity=position_size, price=close_price)
            elif signal == 0:
                # Sell signal
                position_size = self.portfolio['positions'].get(row['ticker'], 0)
                if position_size > 0:
                    self.execute_trade(symbol=row['ticker'], quantity=-position_size, price=close_price)

    def calculate_performance(self):
        trade_prices = np.array([trade['price'] for trade in self.trades])
        trade_quantities = np.array([trade['quantity'] for trade in self.trades])

        trade_returns = np.diff(trade_prices) / trade_prices[:-1]
        trade_pnl = trade_returns * trade_quantities[:-1]

        total_pnl = np.sum(trade_pnl)
        average_trade_return = np.mean(trade_returns)
        win_ratio = np.sum(trade_pnl > 0) / len(trade_pnl)

        return total_pnl, average_trade_return, win_ratio

    def execute_trade(self, symbol, quantity, price):
        self.portfolio['cash'] -= quantity * price

        if symbol in self.portfolio['positions']:
            self.portfolio['positions'][symbol] += quantity
        else:
            self.portfolio['positions'][symbol] = quantity

        self.trades.append({'symbol': symbol, 'quantity': quantity, 'price': price})

    def get_portfolio_value(self, price_data):
        positions_value = sum(self.portfolio['positions'].get(symbol, 0) * price_data[symbol] for symbol in self.portfolio['positions'])
        return self.portfolio['cash'] + positions_value

    def get_portfolio_returns(self):
        portfolio_value = [self.get_portfolio_value(self.data.set_index('ticker')['close'].to_dict()) for _, row in self.data.iterrows()]
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        return returns

    def print_portfolio_summary(self):
        print('--- Portfolio Summary ---')
        print('Cash:', self.portfolio['cash'])
        print('Positions:')
        for symbol, quantity in self.portfolio['positions'].items():
            print(symbol + ':', quantity)

    def plot_portfolio_value(self):
        portfolio_value = [self.get_portfolio_value(self.data.set_index('ticker')['close'].to_dict()) for _, row in self.data.iterrows()]
        dates = self.data['Time']
        signals = self.data['predicted']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(dates, portfolio_value, label='Portfolio Value')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')

        ax2 = ax1.twinx()
        ax2.plot(dates, signals, 'r-', label='Buy/Sell Signal')
        ax2.set_ylabel('Signal')
        ax2.grid(None)

        fig.tight_layout()
        plt.show()

# Load your data
df = test_df  # Assuming test_df is already loaded with your data

# Ensure that 'Time' column is in datetime format
df['Time'] = pd.to_datetime(df['Time'])

# Instantiate the backtesting engine with your data
engine = BacktestingEngine(df)
initial_portfolio_value = engine.get_portfolio_value(df.set_index('ticker')['close'].to_dict())

# Run the backtest
engine.run_backtest()

# Evaluate Performance
final_portfolio_value = engine.get_portfolio_value(df.set_index('ticker')['close'].to_dict())
returns = engine.get_portfolio_returns()
total_returns = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
annualized_returns = (1 + total_returns) ** (252 / len(df)) - 1  # Assuming 252 trading days in a year
volatility = np.std(returns) * np.sqrt(252)
sharpe_ratio = (annualized_returns - 0.02) / volatility  # Assuming risk-free rate of 2%

# Visualize Results
engine.plot_portfolio_value()

# Calculate and print performance metrics
total_pnl, average_trade_return, win_ratio = engine.calculate_performance()
print('--- Performance Metrics ---')
print(f'Total Returns: {total_returns:.2%}')
print(f'Annualized Returns: {annualized_returns:.2%}')
print(f'Volatility: {volatility:.2%}')
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
print(f'Total P&L: {total_pnl:.2f}')
print(f'Average Trade Return: {average_trade_return:.2%}')
print(f'Win Ratio: {win_ratio:.2%}')







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load data
with open("data/model_data.pkl", "rb") as file:
    data = pickle.load(file)

with open("model.pkl", "rb") as f:
    X_train_reshaped, X_test_reshaped, y_train, y_test, y_test_proba, y_test_preds, hour_model = pickle.load(f)

df = data[['Time', 'ticker', 'close', 'oneHourWinner']]
y_train_proba = hour_model.predict(X_train_reshaped)
y_train_preds = (y_train_proba > 0.5).astype(int)

df['predicted'] = list(y_train_preds[:, 0]) + list(y_test_preds[:, 0])

test_start_idx = int(len(df) * 0.8)
test_df = df.iloc[test_start_idx:].copy()

class BacktestingEngine:
    def __init__(self, data):
        self.data = data
        self.portfolio = {'cash': 1000000, 'positions': {}}
        self.trades = []

    def run_backtest(self):
        for index, row in self.data.iterrows():
            close_price = row['close']
            signal = row['predicted']

            if signal == 1:
                # Buy signal
                self.execute_trade(symbol=row['ticker'], quantity=1, price=close_price)
            elif signal == 0:
                # Sell signal
                position_size = self.portfolio['positions'].get(row['ticker'], 0)
                if position_size > 0:
                    self.execute_trade(symbol=row['ticker'], quantity=-position_size, price=close_price)

    def calculate_performance(self):
        trade_prices = np.array([trade['price'] for trade in self.trades])
        trade_quantities = np.array([trade['quantity'] for trade in self.trades])

        trade_returns = np.diff(trade_prices) / trade_prices[:-1]
        trade_pnl = trade_returns * trade_quantities[:-1]

        total_pnl = np.sum(trade_pnl)
        average_trade_return = np.mean(trade_returns)
        win_ratio = np.sum(trade_pnl > 0) / len(trade_pnl)

        return total_pnl, average_trade_return, win_ratio

    def execute_trade(self, symbol, quantity, price):
        self.portfolio['cash'] -= quantity * price

        if symbol in self.portfolio['positions']:
            self.portfolio['positions'][symbol] += quantity
        else:
            self.portfolio['positions'][symbol] = quantity

        self.trades.append({'symbol': symbol, 'quantity': quantity, 'price': price})

    def get_portfolio_value(self, price_data):
        positions_value = sum(self.portfolio['positions'].get(symbol, 0) * price_data[symbol] for symbol in self.portfolio['positions'])
        return self.portfolio['cash'] + positions_value

    def get_portfolio_returns(self):
        portfolio_value = [self.get_portfolio_value(self.data.set_index('ticker')['close'].to_dict()) for _, row in self.data.iterrows()]
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        return returns

    def print_portfolio_summary(self):
        print('--- Portfolio Summary ---')
        print('Cash:', self.portfolio['cash'])
        print('Positions:')
        for symbol, quantity in self.portfolio['positions'].items():
            print(symbol + ':', quantity)

    def plot_portfolio_value(self):
        portfolio_value = [self.get_portfolio_value(self.data.set_index('ticker')['close'].to_dict()) for _, row in self.data.iterrows()]
        dates = self.data['Time']
        signals = self.data['predicted']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(dates, portfolio_value, label='Portfolio Value')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')

        ax2 = ax1.twinx()
        ax2.plot(dates, signals, 'r-', label='Buy/Sell Signal')
        ax2.set_ylabel('Signal')
        ax2.grid(None)

        fig.tight_layout()
        plt.show()

# Load your data
df = test_df  # Assuming test_df is already loaded with your data
df['predicted'] = df['oneHourWinner']
# Ensure that 'Time' column is in datetime format
df['Time'] = pd.to_datetime(df['Time'])

# Instantiate the backtesting engine with your data
engine = BacktestingEngine(df)
initial_portfolio_value = engine.get_portfolio_value(df.set_index('ticker')['close'].to_dict())

# Run the backtest
engine.run_backtest()

# Evaluate Performance
final_portfolio_value = engine.get_portfolio_value(df.set_index('ticker')['close'].to_dict())
returns = engine.get_portfolio_returns()
total_returns = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
annualized_returns = (1 + total_returns) ** (252 / len(df)) - 1  # Assuming 252 trading days in a year
volatility = np.std(returns) * np.sqrt(252)
sharpe_ratio = (annualized_returns - 0.02) / volatility  # Assuming risk-free rate of 2%

# Visualize Results
engine.plot_portfolio_value()

# Calculate and print performance metrics
total_pnl, average_trade_return, win_ratio = engine.calculate_performance()
print('--- Performance Metrics ---')
print(f'Total Returns: {total_returns:.2%}')
print(f'Annualized Returns: {annualized_returns:.2%}')
print(f'Volatility: {volatility:.2%}')
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
print(f'Total P&L: {total_pnl:.2f}')
print(f'Average Trade Return: {average_trade_return:.2%}')
print(f'Win Ratio: {win_ratio:.2%}')

