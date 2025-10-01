"""
Bollinger Bands mean reversion strategy
Trades when price touches or breaches the bands
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class BollingerConfig:
    """Bollinger Bands configuration"""
    lookback: int = 20
    num_std: float = 2.0
    entry_threshold: float = 0.95  # Enter when price is 95% to band
    exit_threshold: float = 0.5   # Exit when price returns to middle
    squeeze_threshold: float = 0.01  # Min band width for volatility squeeze
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_holding_days: int = 10
    use_squeeze: bool = True
    volume_confirmation: bool = True

class BollingerBandsStrategy:
    """
    Bollinger Bands mean reversion strategy
    """
    
    def __init__(self, config: BollingerConfig = None):
        self.config = config or BollingerConfig()
        self.positions = []
        self.trade_history = []
        
    def calculate_bollinger_bands(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        """
        # Calculate moving average and standard deviation
        sma = prices.rolling(window=self.config.lookback).mean()
        std = prices.rolling(window=self.config.lookback).std()
        
        # Calculate bands
        upper_band = sma + (std * self.config.num_std)
        lower_band = sma - (std * self.config.num_std)
        
        # Calculate band width and %B
        band_width = (upper_band - lower_band) / sma
        percent_b = (prices - lower_band) / (upper_band - lower_band)
        
        # Create DataFrame
        bands = pd.DataFrame(index=prices.index)
        bands['price'] = prices
        bands['sma'] = sma
        bands['upper'] = upper_band
        bands['lower'] = lower_band
        bands['width'] = band_width
        bands['percent_b'] = percent_b
        
        # Detect squeeze (low volatility)
        bands['squeeze'] = band_width < self.config.squeeze_threshold
        
        return bands
    
    def detect_band_touches(self, bands: pd.DataFrame) -> pd.DataFrame:
        """
        Detect when price touches or breaches bands
        """
        signals = pd.DataFrame(index=bands.index)
        
        # Upper band touches
        signals['upper_touch'] = (
            (bands['percent_b'] >= self.config.entry_threshold) &
            (bands['percent_b'].shift(1) < self.config.entry_threshold)
        )
        
        # Lower band touches
        signals['lower_touch'] = (
            (bands['percent_b'] <= (1 - self.config.entry_threshold)) &
            (bands['percent_b'].shift(1) > (1 - self.config.entry_threshold))
        )
        
        # Band breaches
        signals['upper_breach'] = bands['price'] > bands['upper']
        signals['lower_breach'] = bands['price'] < bands['lower']
        
        return signals
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators for Bollinger strategy
        """
        df = data.copy()
        
        # Calculate Bollinger Bands
        bands = self.calculate_bollinger_bands(df['Close'])
        
        # Add band data to main DataFrame
        df['bb_upper'] = bands['upper']
        df['bb_lower'] = bands['lower']
        df['bb_middle'] = bands['sma']
        df['bb_width'] = bands['width']
        df['bb_percent'] = bands['percent_b']
        df['bb_squeeze'] = bands['squeeze']
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
        else:
            df['volume_ratio'] = 1
        
        # Additional indicators
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # RSI for confirmation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands
        """
        df = self.calculate_indicators(data)
        bands = self.calculate_bollinger_bands(df['Close'])
        touches = self.detect_band_touches(bands)
        
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['Close']
        signals['percent_b'] = bands['percent_b']
        
        # Entry signals
        # Long when price touches lower band
        signals['long_entry'] = (
            (touches['lower_touch'] | touches['lower_breach']) &
            (df['rsi'] < 35)  # RSI confirmation
        )
        
        # Short when price touches upper band
        signals['short_entry'] = (
            (touches['upper_touch'] | touches['upper_breach']) &
            (df['rsi'] > 65)  # RSI confirmation
        )
        
        # Apply squeeze filter if enabled
        if self.config.use_squeeze:
            # Trade breakouts from squeeze
            signals['long_entry'] = signals['long_entry'] & ~bands['squeeze'].shift(1)
            signals['short_entry'] = signals['short_entry'] & ~bands['squeeze'].shift(1)
        
        # Volume confirmation
        if self.config.volume_confirmation:
            signals['long_entry'] = signals['long_entry'] & (df['volume_ratio'] > 1.2)
            signals['short_entry'] = signals['short_entry'] & (df['volume_ratio'] > 1.2)
        
        # Exit signals
        signals['exit_long'] = bands['percent_b'] > self.config.exit_threshold
        signals['exit_short'] = bands['percent_b'] < (1 - self.config.exit_threshold)
        
        # Combined signal
        signals['signal'] = 0
        signals.loc[signals['long_entry'], 'signal'] = 1
        signals.loc[signals['short_entry'], 'signal'] = -1
        signals.loc[signals['exit_long'] | signals['exit_short'], 'signal'] = 0
        
        # Add additional data
        signals['volatility'] = df['volatility']
        signals['band_width'] = bands['width']
        
        return signals
    
    def calculate_position_size(self, capital: float, price: float,
                               band_width: float, volatility: float) -> int:
        """
        Calculate position size based on band width and volatility
        """
        # Base size
        base_size = capital * 0.02
        
        # Band width adjustment (wider bands = more volatile = smaller position)
        avg_width = 0.04  # Average band width assumption
        if band_width > 0:
            width_adjustment = min(avg_width / band_width, 1.5)
        else:
            width_adjustment = 1.0
        
        # Volatility adjustment
        target_vol = 0.15
        if volatility > 0:
            vol_adjustment = min(target_vol / volatility, 1.0)
        else:
            vol_adjustment = 1.0
        
        # Final position
        position_value = base_size * width_adjustment * vol_adjustment
        
        # Limits
        max_position = capital * 0.10
        position_value = min(position_value, max_position)
        
        return int(position_value / price)
    
    def check_exit_conditions(self, position: Dict, current_price: float,
                            current_percent_b: float, current_date) -> tuple:
        """
        Check exit conditions for Bollinger strategy
        """
        if not position:
            return False, None
        
        entry_price = position['entry_price']
        position_type = position['type']
        entry_date = position['entry_date']
        
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
        
        # 3. Return to middle band
        if position_type == 'long' and current_percent_b > self.config.exit_threshold:
            return True, 'mean_reversion'
        elif position_type == 'short' and current_percent_b < (1 - self.config.exit_threshold):
            return True, 'mean_reversion'
        
        # 4. Maximum holding
        if hasattr(current_date - entry_date, 'days'):
            if (current_date - entry_date).days >= self.config.max_holding_days:
                return True, 'max_holding'
        
        return False, None
