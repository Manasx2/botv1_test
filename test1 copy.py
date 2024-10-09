import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import matplotlib.pyplot as plt

# Fetch Forex Data
def fetch_forex_data(symbol='EURUSD=X', start='2024-10-03', end='2024-10-07', interval='1m'):
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
    risk_factor = 1  # Risk factor for position sizing

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
            (self.macd[-1] > self.macd_signal[-1]) &
            (self.rsi[-1] < 70) &
            (self.data.Close[-1] > self.ema200[-1]) &
            (self.stoch_k[-1] > self.stoch_d[-1]) &
            (self.data.Close[-1] < self.bollinger_lband[-1]) &
            (not self.position)
        )

        short_condition = (
            (self.macd[-1] < self.macd_signal[-1]) &
            (self.rsi[-1] > 30) &
            (self.data.Close[-1] < self.ema200[-1]) &
            (self.stoch_k[-1] < self.stoch_d[-1]) &
            (self.data.Close[-1] > self.bollinger_hband[-1]) &
            (not self.position)
        )

        # Exit Conditions
        exit_long = (
            (self.macd[-1] < self.macd_signal[-1]) |
            (self.rsi[-1] > 70) |
            (self.data.Close[-1] < self.ema200[-1]) |
            (self.stoch_k[-1] < self.stoch_d[-1]) |
            (self.data.Close[-1] > self.bollinger_hband[-1])
        )

        exit_short = (
            (self.macd[-1] > self.macd_signal[-1]) |
            (self.rsi[-1] < 30) |
            (self.data.Close[-1] > self.ema200[-1]) |
            (self.stoch_k[-1] > self.stoch_d[-1]) |
            (self.data.Close[-1] < self.bollinger_lband[-1])
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
    df.to_csv('aapl_forex_data.csv')
    
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
    #optimized_params = bt.optimize(
      #  macd_fast=range(10, 20, 2),
      #  macd_slow=range(20, 40, 2),
      #  macd_signal=range(5, 15, 2),
      # rsi_period=range(10, 20, 2),
      #  ema_period=range(100, 300, 50),
       ## bollinger_std=[2, 2.5, 3],
       # stoch_k=range(10, 20, 5),
       # stoch_d=range(3, 6, 1),
       # atr_period=range(10, 20, 5),
       ##
    #print(optimized_params)

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
    results_df.to_csv('AAPL_backtest_results_advanced.csv', index=False)

    # Print results
    print(results)

    # Plot the backtest results with corrected resample frequency
    #bt.plot(plot_equity=True, plot_drawdown=True, resample='2h')
    #plt.savefig("bt.png")
else:
    print("Failed to download data. Please try again.")