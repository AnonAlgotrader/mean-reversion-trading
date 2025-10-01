"""
RSI-based mean reversion strategy implementation
RSI measures momentum exhaustion for mean reversion opportunities
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class RSIConfig:
    """RSI strategy configuration"""
    rsi_period: int = 14
    oversold_threshold: float = 30
    overbought_threshold: float = 70
    exit_threshold_long: float = 50
    exit_threshold_short: float = 50
    lookback_volatility: int = 20
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_holding_days: int = 10
    use_divergence: bool = True
    volatility_filter: bool = True

class RSIMeanReversion:
    """
    RSI-based mean reversion strategy
    Trades oversold/overbought conditions with momentum exhaustion
    """
    
    def __init__(self, config: RSIConfig = None):
        self.config = config or RSIConfig()
        self.positions = []
        self.trade_history = []
        
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index
        """
        period = period or self.config.rsi_period
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate exponential moving averages
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50)  # Neutral RSI for NaN values
        
        return rsi
    
    def detect_divergence(self, prices: pd.Series, rsi: pd.Series, 
                         lookback: int = 20) -> pd.Series:
        """
        Detect bullish and bearish divergences between price and RSI
        """
        divergence = pd.Series(0, index=prices.index)
        
        if len(prices) < lookback * 2:
            return divergence
        
        for i in range(lookback, len(prices)):
            # Get recent window
            price_window = prices.iloc[i-lookback:i]
            rsi_window = rsi.iloc[i-lookback:i]
            
            # Find local extremes
            price_min_idx = price_window.idxmin()
            price_max_idx = price_window.idxmax()
            
            # Bullish divergence: lower price low but higher RSI low
            if price_min_idx == price_window.index[-1]:  # Recent low
                prev_lows = price_window.iloc[:-5].nsmallest(3)
                if len(prev_lows) > 0:
                    for prev_low_idx in prev_lows.index:
                        if (prices.iloc[i] < prices.loc[prev_low_idx] and 
                            rsi.iloc[i] > rsi.loc[prev_low_idx]):
                            divergence.iloc[i] = 1  # Bullish divergence
                            break
            
            # Bearish divergence: higher price high but lower RSI high
            if price_max_idx == price_window.index[-1]:  # Recent high
                prev_highs = price_window.iloc[:-5].nlargest(3)
                if len(prev_highs) > 0:
                    for prev_high_idx in prev_highs.index:
                        if (prices.iloc[i] > prices.loc[prev_high_idx] and 
                            rsi.iloc[i] < rsi.loc[prev_high_idx]):
                            divergence.iloc[i] = -1  # Bearish divergence
                            break
        
        return divergence
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators for RSI strategy
        """
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['Close'])
        
        # Calculate RSI of different periods for confirmation
        df['rsi_fast'] = self.calculate_rsi(df['Close'], period=7)
        df['rsi_slow'] = self.calculate_rsi(df['Close'], period=21)
        
        # Divergence detection
        if self.config.use_divergence:
            df['divergence'] = self.detect_divergence(df['Close'], df['rsi'])
        else:
            df['divergence'] = 0
        
        # Volatility filter
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(
            self.config.lookback_volatility).std() * np.sqrt(252)
        df['avg_volatility'] = df['volatility'].rolling(60).mean()
        
        # Moving averages for trend filter
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        
        # RSI extremes counter (how long in extreme territory)
        df['oversold_count'] = 0
        df['overbought_count'] = 0
        
        oversold_count = 0
        overbought_count = 0
        
        for i in range(len(df)):
            if df['rsi'].iloc[i] < self.config.oversold_threshold:
                oversold_count += 1
                overbought_count = 0
            elif df['rsi'].iloc[i] > self.config.overbought_threshold:
                overbought_count += 1
                oversold_count = 0
            else:
                oversold_count = 0
                overbought_count = 0
            
            df.loc[df.index[i], 'oversold_count'] = oversold_count
            df.loc[df.index[i], 'overbought_count'] = overbought_count
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI
        """
        df = self.calculate_indicators(data)
        
        signals = pd.DataFrame(index=df.index)
        signals['rsi'] = df['rsi']
        signals['price'] = df['Close']
        
        # Basic RSI signals
        signals['long_entry'] = (
            (df['rsi'] < self.config.oversold_threshold) &
            (df['rsi'].shift(1) >= self.config.oversold_threshold)  # Just entered oversold
        )
        
        signals['short_entry'] = (
            (df['rsi'] > self.config.overbought_threshold) &
            (df['rsi'].shift(1) <= self.config.overbought_threshold)  # Just entered overbought
        )
        
        # Add divergence confirmation if enabled
        if self.config.use_divergence:
            signals['long_entry'] = signals['long_entry'] | (df['divergence'] == 1)
            signals['short_entry'] = signals['short_entry'] | (df['divergence'] == -1)
        
        # Volatility filter
        if self.config.volatility_filter:
            vol_condition = (
                (df['volatility'] > 0.5 * df['avg_volatility']) &
                (df['volatility'] < 2.0 * df['avg_volatility'])
            )
            signals['long_entry'] = signals['long_entry'] & vol_condition
            signals['short_entry'] = signals['short_entry'] & vol_condition
        
        # Exit signals
        signals['long_exit'] = df['rsi'] > self.config.exit_threshold_long
        signals['short_exit'] = df['rsi'] < self.config.exit_threshold_short
        
        # Combined signal
        signals['signal'] = 0
        signals.loc[signals['long_entry'], 'signal'] = 1
        signals.loc[signals['short_entry'], 'signal'] = -1
        signals.loc[signals['long_exit'] | signals['short_exit'], 'signal'] = 0
        
        # Add additional data for analysis
        signals['volatility'] = df['volatility']
        signals['divergence'] = df['divergence']
        
        return signals
    
    def calculate_position_size(self, capital: float, price: float, 
                               rsi: float, volatility: float) -> int:
        """
        Calculate position size based on RSI extremity and volatility
        """
        # Base size
        base_size = capital * 0.02  # 2% base
        
        # RSI extremity adjustment (more extreme = larger position)
        if rsi < self.config.oversold_threshold:
            rsi_multiplier = min((self.config.oversold_threshold - rsi) / 20, 2.0)
        elif rsi > self.config.overbought_threshold:
            rsi_multiplier = min((rsi - self.config.overbought_threshold) / 20, 2.0)
        else:
            rsi_multiplier = 0.5
        
        # Volatility adjustment
        target_vol = 0.15
        if volatility > 0:
            vol_adjustment = min(target_vol / volatility, 1.0)
        else:
            vol_adjustment = 1.0
        
        # Final position
        position_value = base_size * rsi_multiplier * vol_adjustment
        
        # Limits
        max_position = capital * 0.10  # 10% max
        position_value = min(position_value, max_position)
        
        return int(position_value / price)
    
    def check_exit_conditions(self, position: Dict, current_price: float,
                            current_rsi: float, current_date: pd.Timestamp) -> tuple:
        """
        Check exit conditions for RSI strategy
        """
        if not position:
            return False, None
        
        entry_price = position['entry_price']
        entry_date = position['entry_date']
        position_type = position['type']
        
        # Calculate P&L
        if position_type == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Exit conditions
        # 1. Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            return True, 'stop_loss'
        
        # 2. Take profit
        if pnl_pct >= self.config.take_profit_pct:
            return True, 'take_profit'
        
        # 3. RSI exit
        if position_type == 'long' and current_rsi > self.config.exit_threshold_long:
            return True, 'rsi_exit'
        elif position_type == 'short' and current_rsi < self.config.exit_threshold_short:
            return True, 'rsi_exit'
        
        # 4. Maximum holding period
        if hasattr(current_date - entry_date, 'days'):
            if (current_date - entry_date).days >= self.config.max_holding_days:
                return True, 'max_holding'
        
        return False, None
