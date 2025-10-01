"""
Z-Score Mean Reversion Strategy - Complete Implementation
Includes all risk management and position sizing from the article
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

@dataclass
class SignalConfig:
    """Configuration for trading signals"""
    lookback: int = 20
    entry_z_threshold: float = 2.0
    exit_z_threshold: float = 0.5
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_holding_days: int = 10
    volatility_filter: bool = True
    volume_filter: bool = True
    regime_filter: bool = True
    
class ZScoreMeanReversion:
    """
    Complete Z-Score mean reversion strategy with risk management
    Production-ready implementation with all safety checks
    """
    
    def __init__(self, config: SignalConfig = None):
        """
        Initialize strategy with configuration
        
        Parameters:
        -----------
        config : SignalConfig
            Strategy configuration parameters
        """
        self.config = config or SignalConfig()
        self.positions = []
        self.current_position = None
        self.trade_history = []
        self.warnings = []
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators needed for the strategy
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume (optional)
            
        Returns:
        --------
        pd.DataFrame : Data with indicators added
        """
        df = data.copy()
        
        # Validate data
        if df.isnull().sum().sum() > 0:
            self.warnings.append("Data contains NaN values - filling forward")
            df = df.fillna(method='ffill')
        
        # Basic rolling statistics
        df['sma'] = df['Close'].rolling(self.config.lookback).mean()
        df['std'] = df['Close'].rolling(self.config.lookback).std()
        
        # Z-score calculation with safety check
        with np.errstate(divide='ignore', invalid='ignore'):
            df['z_score'] = (df['Close'] - df['sma']) / df['std']
            df['z_score'] = df['z_score'].fillna(0)  # Handle division by zero
        
        # Volatility metrics
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.config.lookback).std() * np.sqrt(252)
        df['avg_volatility'] = df['volatility'].rolling(60).mean()
        
        # Volume metrics (if available)
        if 'Volume' in df.columns:
            df['volume_sma'] = df['Volume'].rolling(self.config.lookback).mean()
            df['relative_volume'] = df['Volume'] / df['volume_sma']
            df['relative_volume'] = df['relative_volume'].fillna(1)
        else:
            df['relative_volume'] = 1  # Default if no volume data
        
        # Bollinger Bands for additional confirmation
        df['bb_upper'] = df['sma'] + (2 * df['std'])
        df['bb_lower'] = df['sma'] - (2 * df['std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma']
        df['bb_width'] = df['bb_width'].fillna(0)
        
        # Half-life calculation for dynamic parameters
        df['half_life'] = self._calculate_rolling_half_life(df['Close'])
        
        return df
    
    def _calculate_rolling_half_life(self, prices: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling half-life for dynamic parameter adjustment"""
        half_lives = []
        
        for i in range(len(prices)):
            if i < window:
                half_lives.append(np.nan)
            else:
                window_prices = prices.iloc[i-window:i]
                hl = self._calculate_half_life(window_prices)
                half_lives.append(hl if hl else np.nan)
        
        return pd.Series(half_lives, index=prices.index)
    
    def _calculate_half_life(self, prices: pd.Series) -> Optional[float]:
        """Calculate half-life using OLS regression"""
        from sklearn.linear_model import LinearRegression
        
        price_lag = prices.shift(1).dropna()
        price_diff = prices.diff().dropna()
        
        if len(price_diff) < 10:
            return None
        
        price_lag = price_lag[price_diff.index]
        
        model = LinearRegression()
        X = price_lag.values.reshape(-1, 1)
        y = price_diff.values
        
        model.fit(X, y)
        beta = model.coef_[0]
        
        if beta < 0:
            return -np.log(2) / beta
        return None
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on z-score with filters
        
        Returns:
        --------
        pd.DataFrame : Trading signals with entry/exit flags
        """
        df = self.calculate_indicators(data)
        
        signals = pd.DataFrame(index=df.index)
        signals['z_score'] = df['z_score']
        signals['volatility'] = df['volatility']
        
        # Basic entry signals
        signals['long_entry'] = (df['z_score'] < -self.config.entry_z_threshold)
        signals['short_entry'] = (df['z_score'] > self.config.entry_z_threshold)
        
        # Apply volatility filter
        if self.config.volatility_filter:
            vol_condition = (
                (df['volatility'] > 0.5 * df['avg_volatility']) & 
                (df['volatility'] < 2.0 * df['avg_volatility'])
            )
            vol_condition = vol_condition.fillna(False)
            signals['long_entry'] = signals['long_entry'] & vol_condition
            signals['short_entry'] = signals['short_entry'] & vol_condition
        
        # Apply volume filter
        if self.config.volume_filter and 'relative_volume' in df.columns:
            vol_condition = df['relative_volume'] > 0.8
            signals['long_entry'] = signals['long_entry'] & vol_condition
            signals['short_entry'] = signals['short_entry'] & vol_condition
        
        # Apply regime filter (avoid trending markets)
        if self.config.regime_filter:
            regime_condition = self._check_regime_filter(df)
            signals['long_entry'] = signals['long_entry'] & regime_condition
            signals['short_entry'] = signals['short_entry'] & regime_condition
        
        # Exit signals
        signals['exit'] = (np.abs(df['z_score']) < self.config.exit_z_threshold)
        
        # Position tracking
        signals['signal'] = 0
        signals.loc[signals['long_entry'], 'signal'] = 1
        signals.loc[signals['short_entry'], 'signal'] = -1
        signals.loc[signals['exit'], 'signal'] = 0
        
        # Add additional data for position sizing
        signals['price'] = df['Close']
        signals['sma'] = df['sma']
        
        return signals
    
    def _check_regime_filter(self, df: pd.DataFrame) -> pd.Series:
        """Check if market regime is suitable for mean reversion"""
        # Simple regime detection using SMA crossovers
        if len(df) < 50:
            return pd.Series(True, index=df.index)
        
        sma_short = df['Close'].rolling(20).mean()
        sma_long = df['Close'].rolling(50).mean()
        
        # Avoid strong trends
        trend_strength = abs((sma_short - sma_long) / sma_long)
        not_trending = trend_strength < 0.05  # Less than 5% difference
        
        return not_trending.fillna(False)
    
    def calculate_position_size(self, 
                               capital: float,
                               price: float,
                               volatility: float,
                               z_score: float,
                               half_life: Optional[float] = None) -> int:
        """
        Calculate position size using multiple methods
        
        Returns:
        --------
        int : Number of shares to trade
        """
        # Base position as percentage of capital
        base_position_pct = 0.02  # 2% base size
        
        # Method 1: Z-score magnitude adjustment
        z_magnitude = abs(z_score)
        if z_magnitude > self.config.entry_z_threshold:
            z_multiplier = min(z_magnitude / self.config.entry_z_threshold, 2.0)
        else:
            z_multiplier = 0.5  # Reduce size for weak signals
        
        # Method 2: Volatility adjustment
        target_vol = 0.15  # 15% target volatility
        if volatility > 0:
            vol_adjustment = min(target_vol / volatility, 1.0)
        else:
            vol_adjustment = 1.0
        
        # Method 3: Half-life adjustment (if available)
        if half_life and half_life > 0:
            # Shorter half-life = stronger mean reversion = larger position
            hl_adjustment = min(10 / half_life, 1.5)
        else:
            hl_adjustment = 1.0
        
        # Calculate final position size
        position_value = capital * base_position_pct * z_multiplier * vol_adjustment * hl_adjustment
        
        # Apply maximum position size limit
        max_position_value = capital * 0.10  # Max 10% per position
        position_value = min(position_value, max_position_value)
        
        # Apply minimum position size
        min_position_value = capital * 0.005  # Min 0.5% to avoid tiny positions
        if position_value < min_position_value:
            return 0
        
        # Convert to shares
        shares = int(position_value / price)
        
        return shares
    
    def check_exit_conditions(self, 
                            position: Dict,
                            current_price: float,
                            current_z_score: float,
                            current_date: pd.Timestamp) -> Tuple[bool, str]:
        """
        Check if any exit condition is met
        
        Returns:
        --------
        Tuple[bool, str] : (should_exit, exit_reason)
        """
        if position is None:
            return False, None
        
        entry_price = position['entry_price']
        entry_date = position['entry_date']
        position_type = position['type']
        
        # Calculate P&L
        if position_type == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # short
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Store P&L for analysis
        position['current_pnl'] = pnl_pct
        
        # 1. Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            return True, 'stop_loss'
        
        # 2. Take profit
        if pnl_pct >= self.config.take_profit_pct:
            return True, 'take_profit'
        
        # 3. Z-score mean reversion
        if position_type == 'long' and current_z_score > -self.config.exit_z_threshold:
            return True, 'z_score_exit'
        elif position_type == 'short' and current_z_score < self.config.exit_z_threshold:
            return True, 'z_score_exit'
        
        # 4. Maximum holding period
        holding_days = (current_date - entry_date).days if hasattr(current_date - entry_date, 'days') else 0
        if holding_days >= self.config.max_holding_days:
            return True, 'max_holding'
        
        # 5. Adverse z-score (position going wrong way)
        if position_type == 'long' and current_z_score > self.config.entry_z_threshold:
            return True, 'adverse_move'
        elif position_type == 'short' and current_z_score < -self.config.entry_z_threshold:
            return True, 'adverse_move'
        
        return False, None
    
    def execute_trade(self, 
                     date: pd.Timestamp,
                     action: str,
                     price: float,
                     shares: int,
                     reason: str = None) -> Dict:
        """
        Execute and record a trade
        """
        trade = {
            'date': date,
            'action': action,
            'price': price,
            'shares': shares,
            'value': price * shares,
            'reason': reason
        }
        
        self.trade_history.append(trade)
        return trade
    
    def get_performance_summary(self) -> Dict:
        """
        Calculate performance metrics for executed trades
        """
        if not self.trade_history:
            return {'error': 'No trades executed'}
        
        # Separate entries and exits
        entries = [t for t in self.trade_history if 'entry' in t['action']]
        exits = [t for t in self.trade_history if 'exit' in t['action']]
        
        if not exits:
            return {
                'total_trades': len(entries),
                'open_positions': len(entries),
                'completed_trades': 0
            }
        
        # Calculate P&L for completed trades
        completed_trades = min(len(entries), len(exits))
        pnl_list = []
        
        for i in range(completed_trades):
            entry = entries[i]
            exit = exits[i]
            
            if 'long' in entry['action']:
                pnl = (exit['price'] - entry['price']) * entry['shares']
            else:  # short
                pnl = (entry['price'] - exit['price']) * entry['shares']
            
            pnl_pct = pnl / (entry['price'] * entry['shares'])
            pnl_list.append({'pnl': pnl, 'pnl_pct': pnl_pct})
        
        # Calculate metrics
        total_pnl = sum(t['pnl'] for t in pnl_list)
        wins = [t for t in pnl_list if t['pnl'] > 0]
        losses = [t for t in pnl_list if t['pnl'] < 0]
        
        return {
            'total_trades': completed_trades,
            'total_pnl': total_pnl,
            'win_rate': len(wins) / completed_trades if completed_trades > 0 else 0,
            'avg_win': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl_pct'] for t in losses]) if losses else 0,
            'profit_factor': abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses else 0,
            'warnings': self.warnings
        }
