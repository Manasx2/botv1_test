import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import matplotlib.pyplot as plt

# Fetch Forex Data
def fetch_forex_data(symbol='TSLA', start='2024-10-01', end='2024-10-07', interval='1m'):
    try:
        data = yf.download(symbol, start=start, end=end, interval=interval)

        # Handling missing values
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Failed to download data: {e}")
        return None

# Define the advanced trading strategy
class AdvancedForexStrategy(Strategy):
    # Define parameters for optimization
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    ema_period = 200
    bollinger_window = 20
    bollinger_std = 2
    stoch_k = 14
    stoch_d = 3
    atr_period = 14
    risk_factor = 1 # Risk factor for position sizing

    def init(self):
        # Initialize indicators
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)

        # MACD
        macd = MACD(close, window_slow=self.macd_slow, window_fast=self.macd_fast, window_sign=self.macd_signal)
        self.macd = self.I(macd.macd_diff)
        self.macd_signal = self.I(macd.macd_signal)

        # RSI
        rsi = RSIIndicator(close, window=self.rsi_period)
        self.rsi = self.I(rsi.rsi)

        # EMA
        ema = EMAIndicator(close, window=self.ema_period)
        self.ema200 = self.I(ema.ema_indicator)

        # Bollinger Bands
        bollinger = BollingerBands(close, window=self.bollinger_window, window_dev=self.bollinger_std)
        self.bollinger_hband = self.I(bollinger.bollinger_hband)
        self.bollinger_lband = self.I(bollinger.bollinger_lband)

        # Stochastic Oscillator
        stoch = StochasticOscillator(high, low, close, window=self.stoch_k, smooth_window=self.stoch_d)
        self.stoch_k = self.I(stoch.stoch)
        self.stoch_d = self.I(stoch.stoch_signal)

        # ATR for risk management
        atr = AverageTrueRange(high, low, close, window=self.atr_period)
        self.atr = self.I(atr.average_true_range)

    def next(self):
        # Dynamic position sizing based on ATR
        risk_per_trade = self.equity * 0.01  # 1% risk per trade
        atr_value = self.atr[-1]
        price = self.data.Close[-1]
        stop_loss = 2 * atr_value

        if atr_value == 0:
            return  # Avoid division by zero

        # Calculate size_fraction ensuring it remains between 0.01 and 1
        size_fraction = risk_per_trade / (price * stop_loss)
        size_fraction = min(size_fraction, 1)  # Cap at 1 to avoid exceeding equity
        size_fraction = max(size_fraction, 0.01)  # Ensure a minimum size fraction

        # Debugging: Print key variables
        print(f"Equity: {self.equity:.2f}, Price: {price:.4f}, ATR: {atr_value:.4f}, Stop Loss: {stop_loss:.4f}, Size Fraction: {size_fraction:.4f}")

        # Entry Conditions
        long_condition = (
            (self.macd[-1] > self.macd_signal[-1] * 0.95) &
            (self.rsi[-1] < 85) &
            (self.data.Close[-1] > self.ema200[-1] * 0.99) &
            (self.stoch_k[-1] > self.stoch_d[-1] * 0.95) &
            (self.data.Close[-1] < self.bollinger_lband[-1] * 1.05) &
            (not self.position)
        )

        short_condition = (
            (self.macd[-1] < self.macd_signal[-1] * 0.2) &
            (self.rsi[-1] > 60 ) &
            (self.data.Close[-1] < self.ema200[-1]* 1.01) &
            (self.stoch_k[-1] < self.stoch_d[-1]* 1.05) &
            (self.data.Close[-1] > self.bollinger_hband[-1] *0.95) &
            (not self.position)
        )

        # Debugging: Print condition status
        print(f"Long Condition: {long_condition}")
        print(f"Short Condition: {short_condition}")

        # Exit Conditions
        exit_long = (
            (self.macd[-1] < self.macd_signal[-1] * 0.95) |
            (self.rsi[-1] > 60) |
            (self.data.Close[-1] < self.ema200[-1]) |
            (self.stoch_k[-1] < self.stoch_d[-1] * 0.95) |
            (self.data.Close[-1] > self.bollinger_hband[-1]* 1.05)
        )

        exit_short = (
            (self.macd[-1] > self.macd_signal[-1] * 0.95) |
            (self.rsi[-1] < 35) |
            (self.data.Close[-1] > self.ema200[-1] * 1.21) |
            (self.stoch_k[-1] > self.stoch_d[-1] * 0.95) |
            (self.data.Close[-1] < self.bollinger_lband[-1] * 0.95)
        )

        # Execute Trades
        if long_condition:
            print("Entering long position")
            self.buy(
                size=size_fraction,
                sl=price - stop_loss,
                tp=price + 4 * atr_value
            )

        elif short_condition:
            print("Entering short position")
            self.sell(
                size=size_fraction,
                sl=price + stop_loss,
                tp=price - 4 * atr_value
            )

        # Trailing Stop
        if self.position:
            if self.position.is_long:
                trailing_stop = price - atr_value
                if price < trailing_stop:
                    print("Trailing stop hit for long position")
                    self.position.close()
            elif self.position.is_short:
                trailing_stop = price + atr_value
                if price > trailing_stop:
                    print("Trailing stop hit for short position")
                    self.position.close()

        # Exit Conditions
        if self.position and self.position.is_long and exit_long:
            print("Exit condition hit for long position")
            self.position.close()

        if self.position and self.position.is_short and exit_short:
            print("Exit condition hit for short position")
            self.position.close()

# Fetch data and run backtest
df = fetch_forex_data()

if df is not None:
    # Ensure column names are in the correct format for backtesting
    df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Save Forex data to CSV
    df.to_csv('forex_data.csv')

    # Run backtest
    bt = Backtest(
        df,
        AdvancedForexStrategy,
        cash=10000,
        commission=.002,
        exclusive_orders=True,
        trade_on_close=False
    )

    # Parameter Optimization (optional)
    # Uncomment the following lines to perform optimization
    # optimized_params = bt.optimize(
    #     macd_fast=range(10, 20, 2),
    #     macd_slow=range(20, 40, 2),
    #     macd_signal=range(5, 15, 2),
    #     rsi_period=range(10, 20, 2),
    #     ema_period=range(100, 300, 50),
    #     bollinger_window=range(15, 25, 5),
    #     bollinger_std=[2, 2.5, 3],
    #     stoch_k=range(10, 20, 5),
    #     stoch_d=range(3, 6, 1),
    #     atr_period=range(10, 20, 5),
    #     maximize='Return [%]'
    # )
    # print(optimized_params)

    # Run backtest without optimization
    results = bt.run()

    # Save backtest results to CSV (extract key metrics)
metrics = {
        'Start': getattr(results, 'Start', 'N/A'),
        'End': getattr(results, 'End', 'N/A'),
        'Equity Final [$]': getattr(results, 'Equity Final [$]', 'N/A'),
        'Equity Peak [$]': getattr(results, 'Equity Peak [$]', 'N/A'),
        'Return [%]': getattr(results, 'Return [%]', 'N/A'),
        'Buy & Hold Return [%]': getattr(results, 'Buy & Hold Return [%]', 'N/A'),
        'Max. Drawdown [%]': getattr(results, 'Max. Drawdown [%]', 'N/A'),
        'Sharpe Ratio': getattr(results, 'Sharpe Ratio', 'N/A'),
        'Sortino Ratio': getattr(results, 'Sortino Ratio', 'N/A'),
        'Calmar Ratio': getattr(results, 'Calmar Ratio', 'N/A'),
        'Total Trades': len(getattr(results, 'Trades', [])),
        'Win Rate [%]': getattr(results, 'Win Rate [%]', 'N/A'),
        'Best Trade [%]': getattr(results, 'Best Trade [%]', 'N/A'),
        'Worst Trade [%]': getattr(results, 'Worst Trade [%]', 'N/A'),
        'Avg. Trade [%]': getattr(results, 'Avg. Trade [%]', 'N/A'),
        'Expectancy': getattr(results, 'Expectancy', 'N/A'),
    }
results_df = pd.DataFrame([metrics])
results_df.to_csv('backtest_results_advanced.csv', index=False)

    # Print results
print(results)

    # Custom plotting using matplotlib
        # Custom plotting using matplotlib
equity_curve = results._equity_curve

plt.figure(figsize=(12, 6))
plt.plot(equity_curve.index, equity_curve['Equity'], label='Equity')
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.savefig("equity_curve.png")
plt.close()

    # Plot drawdown
drawdown = (equity_curve['Equity'] / equity_curve['Equity'].cummax() - 1) * 100

plt.figure(figsize=(12, 6))
plt.plot(drawdown.index, drawdown, label='Drawdown')
plt.title('Drawdown')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.legend()
plt.grid(True)
plt.savefig("drawdown.png")
plt.close()

    # You can still try the original plot function, but with a single date format string
try:
        bt.plot(plot_equity=True, plot_drawdown=True, resample='2h', 
                datetime_format='%Y-%m-%d')
        plt.savefig("bt_original.png")
except Exception as e:
        print(f"Original plot function failed: {e}")

    # Additional debugging information
print("\nStrategy Parameters:")
for param, value in AdvancedForexStrategy.__dict__.items():
        if not param.startswith('__') and not callable(value):
            print(f"{param}: {value}")

print("\nData Summary:")
print(df.describe())

print("\nFirst few rows of data:")
print(df.head())

print("\nLast few rows of data:")
print(df.tail())

print("\nIndicator values for the last data point:")
print(f"MACD: {results._strategy.macd[-1]}")
print(f"MACD Signal: {results._strategy.macd_signal[-1]}")
print(f"RSI: {results._strategy.rsi[-1]}")
print(f"EMA200: {results._strategy.ema200[-1]}")
print(f"Bollinger Upper: {results._strategy.bollinger_hband[-1]}")
print(f"Bollinger Lower: {results._strategy.bollinger_lband[-1]}")
print(f"Stochastic K: {results._strategy.stoch_k[-1]}")
print(f"Stochastic D: {results._strategy.stoch_d[-1]}")
print(f"ATR: {results._strategy.atr[-1]}")

    # Check if any trades were executed
if len(results._trades) == 0:
        print("\nNo trades were executed during the backtest.")
        print("Possible reasons:")
        print("1. Entry conditions may be too strict.")
        print("2. The strategy may not be suitable for the given market conditions.")
        print("3. There might be an issue with the data or indicator calculations.")
        print("\nSuggestions:")
        print("1. Loosen the entry conditions in the strategy.")
        print("2. Check the indicator values throughout the backtest period.")
        print("3. Verify the data quality and completeness.")
else:
        print(f"\nTotal number of trades executed: {len(results._trades)}")
        print("\nTrade Statistics:")
        print(f"Win Rate: {results['Win Rate [%]']:.2f}%")
        print(f"Average Trade: {results['Avg. Trade [%]']:.2f}%")
        print(f"Best Trade: {results['Best Trade [%]']:.2f}%")
        print(f"Worst Trade: {results['Worst Trade [%]']:.2f}%")
        
        '''print("\nDetailed Trade Information:")
        for i, trade in enumerate(results._trades, 1):
            print(f"\nTrade {i}:")
            print(f"Entry Time: {trade.entry_time}")
            print(f"Exit Time: {trade.exit_time}")
            print(f"Entry Price: {trade.entry_price:.4f}")
            print(f"Exit Price: {trade.exit_price:.4f}")
            print(f"Size: {trade.size:.4f}")
            print(f"PnL: {trade.pnl:.2f}")
            print(f"Return: {trade.return_pct:.2f}%")'''

    # Additional analysis
print("\nMonthly Returns:")
monthly_returns = equity_curve['Equity'].resample('ME').last().pct_change()
print(monthly_returns)

print("\nAnnual Returns:")
annual_returns = equity_curve['Equity'].resample('YE').last().pct_change()
print(annual_returns)

# Plotting monthly returns
plt.figure(figsize=(12, 6))
monthly_returns.plot(kind='bar')
plt.title('Monthly Returns')
plt.xlabel('Date')
plt.ylabel('Return (%)')
plt.savefig("monthly_returns.png")
plt.close()

    # Calculate and plot the cumulative returns
cumulative_returns = (1 + equity_curve['Equity'].pct_change()).cumprod() - 1
plt.figure(figsize=(12, 6))
        # Calculate and plot the cumulative returns
cumulative_returns = (1 + equity_curve['Equity'].pct_change()).cumprod() - 1
plt.figure(figsize=(12, 6))
cumulative_returns.plot()
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.savefig("cumulative_returns.png")
plt.close()

    # Calculate and print risk metrics
print("\nRisk Metrics:")
returns = equity_curve['Equity'].pct_change()
annualized_return = (1 + returns.mean()) ** 252 - 1
annualized_volatility = returns.std() * (252 ** 0.5)
sharpe_ratio = (annualized_return - 0.02) / annualized_volatility  # Assuming 2% risk-free rate
    
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Plot return distribution
plt.figure(figsize=(12, 6))
returns.hist(bins=50)
plt.title('Return Distribution')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.savefig("return_distribution.png")
plt.close()

    # Analyze drawdowns
drawdowns = (equity_curve['Equity'] / equity_curve['Equity'].cummax() - 1) * 100
max_drawdown = drawdowns.min()
avg_drawdown = drawdowns[drawdowns < 0].mean()
    
print(f"\nMax Drawdown: {max_drawdown:.2f}%")
print(f"Average Drawdown: {avg_drawdown:.2f}%")

        # Plot drawdown distribution
plt.figure(figsize=(12, 6))
drawdowns[drawdowns < 0].hist(bins=50)
plt.title('Drawdown Distribution')
plt.xlabel('Drawdown (%)')
plt.ylabel('Frequency')
plt.savefig("drawdown_distribution.png")
plt.close()

# Analyze trade durations
if len(results._trades) > 0:
        trade_durations = [(trade.exit_time - trade.entry_time).total_seconds() / 3600 for trade in results._trades]
        avg_duration = sum(trade_durations) / len(trade_durations)
        max_duration = max(trade_durations)
        min_duration = min(trade_durations)

        print(f"\nAverage Trade Duration: {avg_duration:.2f} hours")
        print(f"Longest Trade Duration: {max_duration:.2f} hours")
        print(f"Shortest Trade Duration: {min_duration:.2f} hours")

        # Plot trade duration distribution
        plt.figure(figsize=(12, 6))
        plt.hist(trade_durations, bins=50)
        plt.title('Trade Duration Distribution')
        plt.xlabel('Duration (hours)')
        plt.ylabel('Frequency')
        plt.savefig("trade_duration_distribution.png")
        plt.close()

        # Analyze win/loss streak
if len(results._trades) > 0:
        trade_results = [1 if trade.pnl > 0 else 0 for trade in results.trades]
        current_streak = 1
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for i in range(1, len(trade_results)):
            if trade_results[i] == trade_results[i-1]:
                current_streak += 1
            else:
                if trade_results[i-1] == 1:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
                current_streak = 1

        if trade_results[-1] == 1:
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)

        print(f"\nLongest Winning Streak: {max_win_streak} trades")
        print(f"Longest Losing Streak: {max_loss_streak} trades")

    # Analyze trade size distribution
if len(results._trades) > 0:
        trade_sizes = [trade.size for trade in results.trades]
        avg_trade_size = sum(trade_sizes) / len(trade_sizes)
        max_trade_size = max(trade_sizes)
        min_trade_size = min(trade_sizes)

        print(f"\nAverage Trade Size: {avg_trade_size:.4f}")
        print(f"Largest Trade Size: {max_trade_size:.4f}")
        print(f"Smallest Trade Size: {min_trade_size:.4f}")

        # Plot trade size distribution
        plt.figure(figsize=(12, 6))
        plt.hist(trade_sizes, bins=50)
        plt.title('Trade Size Distribution')
        plt.xlabel('Trade Size')
        plt.ylabel('Frequency')
        plt.savefig("trade_size_distribution.png")
        plt.close()

    # Analyze profit factor
if len(results._trades) > 0:
        gross_profits = sum([trade.pnl for trade in results.trades if trade.pnl > 0])
        gross_losses = abs(sum([trade.pnl for trade in results.trades if trade.pnl < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')

        print(f"\nProfit Factor: {profit_factor:.2f}")

    # Analyze win rate by day of week
if len(results._trades) > 0:
        trades_by_day = {i: [] for i in range(7)}
        for trade in results.trades:
            day = trade.entry_time.weekday()
            trades_by_day[day].append(1 if trade.pnl > 0 else 0)

        win_rate_by_day = {day: sum(trades)/len(trades) if trades else 0 for day, trades in trades_by_day.items()}
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        win_rates = [win_rate_by_day[i] * 100 for i in range(7)]

        plt.figure(figsize=(12, 6))
        plt.bar(days, win_rates)
        plt.title('Win Rate by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Win Rate (%)')
        plt.savefig("win_rate_by_day.png")
        plt.close()

        print("\nWin Rate by Day of Week:")
        for day, win_rate in zip(days, win_rates):
            print(f"{day}: {win_rate:.2f}%")

    # Analyze performance by month
monthly_performance = equity_curve['Equity'].resample('ME').last().pct_change()
plt.figure(figsize=(12, 6))
monthly_performance.plot(kind='bar')
plt.title('Monthly Performance')
plt.xlabel('Month')
plt.ylabel('Return (%)')
plt.savefig("monthly_performance.png")
plt.close()

print("\nMonthly Performance Statistics:")
print(monthly_performance.describe())

    # Analyze rolling Sharpe ratio
rolling_returns = equity_curve['Equity'].pct_change()
rolling_sharpe = rolling_returns.rolling(window=252).apply(lambda x: (x.mean() - 0.02) / x.std() * (252**0.5))
    
plt.figure(figsize=(12, 6))
rolling_sharpe.plot()
plt.title('Rolling Sharpe Ratio (252-day window)')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.savefig("rolling_sharpe_ratio.png")
plt.close()

    # Analyze correlation with market benchmark
    # Note: You'll need to fetch benchmark data separately
    # Analyze correlation with market benchmark
    # Note: You'll need to fetch benchmark data separately
