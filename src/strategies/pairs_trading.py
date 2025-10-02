"""
Pairs trading implementation with cointegration and dynamic hedge ratios
Market-neutral strategy trading relative value between correlated assets
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
import warnings

@dataclass 
class PairsConfig:
    """Pairs trading configuration"""
    lookback_coint: int = 60
    lookback_hedge: int = 30
    entry_z_threshold: float = 2.0
    exit_z_threshold: float = 0.5
    stop_loss_pct: float = 0.05
    max_holding_days: int = 15
    min_correlation: float = 0.7
    max_correlation: float = 0.95
    coint_significance: float = 0.05
    use_kalman_filter: bool = False
    rebalance_frequency: int = 5  # days

class PairsTrading:
    """
    Pairs trading strategy with cointegration testing and dynamic hedging
    """
    
    def __init__(self, config: PairsConfig = None):
        self.config = config or PairsConfig()
        self.pairs = []
        self.positions = {}
        self.trade_history = []
        
    def find_cointegrated_pairs(self, prices_df: pd.DataFrame, 
                               significance: float = None) -> List[Tuple]:
        """
        Find cointegrated pairs from a DataFrame of prices
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with asset prices in columns
        significance : float
            P-value threshold for cointegration
            
        Returns:
        --------
        List of tuples: (asset1, asset2, p_value, correlation)
        """
        significance = significance or self.config.coint_significance
        n = prices_df.shape[1]
        pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                asset1 = prices_df.columns[i]
                asset2 = prices_df.columns[j]
                
                # Check correlation first (computational efficiency)
                correlation = prices_df[asset1].corr(prices_df[asset2])
                
                if (self.config.min_correlation <= correlation <= self.config.max_correlation):
                    # Test for cointegration
                    try:
                        score, p_value, critical_values = coint(
                            prices_df[asset1].dropna(),
                            prices_df[asset2].dropna()
                        )
                        
                        if p_value < significance:
                            pairs.append((asset1, asset2, p_value, correlation))
                    except Exception as e:
                        warnings.warn(f"Cointegration test failed for {asset1}-{asset2}: {e}")
        
        # Sort by p-value (most cointegrated first)
        pairs.sort(key=lambda x: x[2])
        
        return pairs
    
    def calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series,
                            method: str = 'ols') -> float:
        """
        Calculate hedge ratio between two assets
        
        Parameters:
        -----------
        price1, price2 : pd.Series
            Price series for the pair
        method : str
            'ols' for ordinary least squares, 'tls' for total least squares
            
        Returns:
        --------
        float : Hedge ratio (units of asset2 per unit of asset1)
        """
        if method == 'ols':
            # Standard OLS regression
            model = LinearRegression()
            X = price2.values.reshape(-1, 1)
            y = price1.values
            model.fit(X, y)
            hedge_ratio = model.coef_[0]
            
        elif method == 'tls':
            # Total least squares (orthogonal regression)
            # Better when both variables have measurement error
            x = price2.values
            y = price1.values
            
            # Center the data
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            x_c = x - x_mean
            y_c = y - y_mean
            
            # Calculate covariance matrix
            cov_matrix = np.cov(x_c, y_c)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # Hedge ratio from eigenvector of smallest eigenvalue
            min_idx = np.argmin(eigenvalues)
            hedge_ratio = -eigenvectors[0, min_idx] / eigenvectors[1, min_idx]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return hedge_ratio
    
    def calculate_dynamic_hedge_ratio(self, price1: pd.Series, price2: pd.Series,
                                    window: int = None) -> pd.Series:
        """
        Calculate rolling hedge ratio over time
        """
        window = window or self.config.lookback_hedge
        hedge_ratios = []
        
        for i in range(len(price1)):
            if i < window:
                hedge_ratios.append(np.nan)
            else:
                window_p1 = price1.iloc[i-window:i]
                window_p2 = price2.iloc[i-window:i]
                
                try:
                    hr = self.calculate_hedge_ratio(window_p1, window_p2)
                    hedge_ratios.append(hr)
                except:
                    hedge_ratios.append(np.nan)
        
        return pd.Series(hedge_ratios, index=price1.index)
    
    def calculate_kalman_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> pd.Series:
        """
        Calculate hedge ratio using Kalman filter for dynamic adaptation
        """
        # Kalman filter parameters
        delta = 1e-5
        Vt = delta / (1 - delta)
        Wt = 1e-3
        theta = 0  # Initial hedge ratio
        C = Vt
        
        hedge_ratios = []
        
        for i in range(len(price1)):
            # Prediction
            theta_pred = theta
            C_pred = C + Wt
            
            # Update
            y = price1.iloc[i]
            x = price2.iloc[i]
            
            if x != 0:  # Avoid division by zero
                e = y - theta_pred * x  # Prediction error
                K = C_pred * x / (x**2 * C_pred + Vt)  # Kalman gain
                theta = theta_pred + K * e  # Updated hedge ratio
                C = (1 - K * x) * C_pred  # Updated covariance
            
            hedge_ratios.append(theta)
        
        return pd.Series(hedge_ratios, index=price1.index)
    
    def calculate_spread(self, price1: pd.Series, price2: pd.Series,
                       hedge_ratio: float = None) -> pd.Series:
        """
        Calculate spread between two assets
        """
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        
        spread = price1 - hedge_ratio * price2
        return spread
    
    def calculate_spread_zscore(self, spread: pd.Series, 
                               lookback: int = None) -> pd.Series:
        """
        Calculate z-score of the spread
        """
        lookback = lookback or self.config.lookback_coint
        
        rolling_mean = spread.rolling(lookback).mean()
        rolling_std = spread.rolling(lookback).std()
        
        # Avoid division by zero
        zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
        zscore = zscore.fillna(0)
        
        return zscore
    
    def generate_signals(self, price1: pd.Series, price2: pd.Series,
                       pair_name: str = None) -> pd.DataFrame:
        """
        Generate trading signals for a pair
        """
        # Calculate dynamic hedge ratio
        if self.config.use_kalman_filter:
            hedge_ratios = self.calculate_kalman_hedge_ratio(price1, price2)
        else:
            hedge_ratios = self.calculate_dynamic_hedge_ratio(price1, price2)
        
        # Calculate spread with dynamic hedge ratio
        spreads = []
        for i in range(len(price1)):
            if pd.notna(hedge_ratios.iloc[i]):
                spread_val = price1.iloc[i] - hedge_ratios.iloc[i] * price2.iloc[i]
            else:
                spread_val = np.nan
            spreads.append(spread_val)
        
        spread = pd.Series(spreads, index=price1.index)
        
        # Calculate z-score
        zscore = self.calculate_spread_zscore(spread)
        
        # Generate signals
        signals = pd.DataFrame(index=price1.index)
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['hedge_ratio'] = hedge_ratios
        
        # Entry signals
        signals['long_spread'] = zscore < -self.config.entry_z_threshold
        signals['short_spread'] = zscore > self.config.entry_z_threshold
        
        # Exit signals  
        signals['exit'] = np.abs(zscore) < self.config.exit_z_threshold
        
        # Combined signal
        signals['signal'] = 0
        signals.loc[signals['long_spread'], 'signal'] = 1
        signals.loc[signals['short_spread'], 'signal'] = -1
        signals.loc[signals['exit'], 'signal'] = 0
        
        # Add prices for position sizing
        signals['price1'] = price1
        signals['price2'] = price2
        
        return signals
    
    def calculate_position_sizes(self, capital: float, price1: float, 
                                price2: float, hedge_ratio: float,
                                zscore: float) -> Dict:
        """
        Calculate position sizes for both legs of the pair trade
        """
        # Base position size (risk 2% of capital)
        base_risk = capital * 0.02
        
        # Z-score adjustment (higher conviction for extreme z-scores)
        z_multiplier = min(abs(zscore) / self.config.entry_z_threshold, 2.0)
        
        # Total capital to allocate
        allocated_capital = base_risk * z_multiplier
        
        # Split between legs based on hedge ratio
        # If hedge_ratio = 0.8, we need 0.8 units of asset2 for each unit of asset1
        total_value = price1 + hedge_ratio * price2
        
        # Calculate shares
        asset1_value = allocated_capital * (price1 / total_value)
        asset2_value = allocated_capital * (hedge_ratio * price2 / total_value)
        
        shares1 = int(asset1_value / price1)
        shares2 = int(asset2_value / price2)
        
        return {
            'asset1_shares': shares1,
            'asset2_shares': shares2,
            'asset1_value': shares1 * price1,
            'asset2_value': shares2 * price2,
            'hedge_ratio': hedge_ratio
        }
    
    def check_exit_conditions(self, position: Dict, current_spread: float,
                            current_zscore: float, current_date: pd.Timestamp) -> Tuple:
        """
        Check exit conditions for pairs trade
        """
        if not position:
            return False, None
        
        entry_spread = position['entry_spread']
        entry_date = position['entry_date']
        position_type = position['type']
        
        # Calculate P&L
        if position_type == 'long_spread':
            pnl_pct = (current_spread - entry_spread) / abs(entry_spread) if entry_spread != 0 else 0
        else:  # short_spread
            pnl_pct = (entry_spread - current_spread) / abs(entry_spread) if entry_spread != 0 else 0
        
        # Exit conditions
        # 1. Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            return True, 'stop_loss'
        
        # 2. Z-score convergence
        if abs(current_zscore) < self.config.exit_z_threshold:
            return True, 'zscore_convergence'
        
        # 3. Z-score divergence (trade going wrong way)
        if position_type == 'long_spread' and current_zscore > self.config.entry_z_threshold:
            return True, 'spread_divergence'
        elif position_type == 'short_spread' and current_zscore < -self.config.entry_z_threshold:
            return True, 'spread_divergence'
        
        # 4. Maximum holding period
        if hasattr(current_date - entry_date, 'days'):
            if (current_date - entry_date).days >= self.config.max_holding_days:
                return True, 'max_holding'
        
        # 5. Cointegration breakdown (optional check)
        # This would require re-testing cointegration periodically
        
        return False, None
    
    def validate_pair(self, price1: pd.Series, price2: pd.Series) -> Dict:
        """
        Validate if a pair is suitable for trading
        """
        # Test cointegration
        score, p_value, critical_values = coint(price1.dropna(), price2.dropna())
        
        # Calculate correlation
        correlation = price1.corr(price2)
        
        # Calculate spread statistics
        hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        
        # Test spread for stationarity
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(spread.dropna())
        
        # Calculate half-life
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        spread_lag = spread_lag[spread_diff.index]
        
        model = LinearRegression()
        X = spread_lag.values.reshape(-1, 1)
        y = spread_diff.values
        
        model.fit(X, y)
        beta = model.coef_[0]
        
        if beta < 0:
            half_life = -np.log(2) / beta
        else:
            half_life = None
        
        return {
            'is_cointegrated': p_value < self.config.coint_significance,
            'coint_pvalue': p_value,
            'correlation': correlation,
            'hedge_ratio': hedge_ratio,
            'spread_stationary': adf_result[1] < 0.05,
            'spread_adf_pvalue': adf_result[1],
            'half_life': half_life,
            'suitable_for_trading': (
                p_value < self.config.coint_significance and
                self.config.min_correlation <= correlation <= self.config.max_correlation and
                adf_result[1] < 0.05 and
                half_life is not None and half_life < 30
            )
        }
