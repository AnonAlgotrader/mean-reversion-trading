"""
Market regime detection for mean reversion strategies
Critical for avoiding catastrophic losses during trending markets
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings

class RegimeDetector:
    """
    Detect market regimes to avoid trading during unfavorable conditions
    Multiple methods to ensure robustness
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.current_regime = 'unknown'
        self.regime_history = []
        
    def detect_regime(self, prices: pd.Series) -> Dict:
        """
        Comprehensive regime detection using multiple methods
        """
        results = {}
        
        # Method 1: Rolling Hurst Exponent
        hurst_regime = self._hurst_regime_detection(prices)
        results['hurst_regime'] = hurst_regime
        
        # Method 2: Volatility regime
        vol_regime = self._volatility_regime_detection(prices)
        results['volatility_regime'] = vol_regime
        
        # Method 3: Trend strength
        trend_regime = self._trend_regime_detection(prices)
        results['trend_regime'] = trend_regime
        
        # Method 4: Mean reversion speed
        reversion_regime = self._reversion_speed_detection(prices)
        results['reversion_speed'] = reversion_regime
        
        # Method 5: Structural breaks
        structural_regime = self._detect_structural_breaks(prices)
        results['structural_breaks'] = structural_regime
        
        # Combine all methods for final assessment
        final_regime = self._combine_regime_indicators(results)
        results['current_regime'] = final_regime
        results['is_favorable'] = final_regime in ['mean_reverting', 'ranging']
        results['should_trade'] = results['is_favorable']
        results['recommendation'] = self._get_recommendation(final_regime)
        
        # Store history
        self.current_regime = final_regime
        self.regime_history.append({
            'date': prices.index[-1] if len(prices) > 0 else None,
            'regime': final_regime
        })
        
        return results
    
    def _calculate_hurst(self, prices: pd.Series, max_lag: int = 50) -> Optional[float]:
        """Calculate Hurst exponent for a price series"""
        if len(prices) < max_lag * 2:
            return None
            
        lags = range(2, min(max_lag, len(prices)//2))
        log_prices = np.log(prices.dropna())
        
        variances = []
        for lag in lags:
            differences = log_prices[lag:].to_numpy() - log_prices[:-lag].to_numpy()
            variances.append(np.var(differences))
        
        if not variances or all(v == 0 for v in variances):
            return None
            
        log_lags = np.log(list(lags))
        log_variances = np.log(variances)
        
        coeffs = np.polyfit(log_lags, log_variances, 1)
        hurst = coeffs[0] / 2
        
        return hurst
    
    def _hurst_regime_detection(self, prices: pd.Series) -> Dict:
        """Detect regime using rolling Hurst exponent"""
        window = min(self.lookback, len(prices) // 2)
        
        if len(prices) < window * 2:
            return {'regime': 'insufficient_data', 'hurst': None}
        
        # Calculate rolling Hurst
        hurst_values = []
        for i in range(window, len(prices)):
            window_prices = prices.iloc[i-window:i]
            hurst = self._calculate_hurst(window_prices, max_lag=window//2)
            if hurst:
                hurst_values.append(hurst)
        
        if not hurst_values:
            return {'regime': 'insufficient_data', 'hurst': None}
        
        current_hurst = hurst_values[-1]
        avg_hurst = np.mean(hurst_values)
        
        # Classify regime based on Hurst
        if current_hurst < 0.45:
            regime = 'strong_mean_reverting'
        elif current_hurst < 0.50:
            regime = 'weak_mean_reverting'
        elif current_hurst < 0.55:
            regime = 'ranging'
        elif current_hurst < 0.65:
            regime = 'weak_trending'
        else:
            regime = 'strong_trending'
        
        return {
            'regime': regime,
            'current_hurst': current_hurst,
            'average_hurst': avg_hurst,
            'is_favorable': current_hurst < 0.55
        }
    
    def _volatility_regime_detection(self, prices: pd.Series) -> Dict:
        """Detect volatility regime"""
        returns = prices.pct_change().dropna()
        
        if len(returns) < self.lookback:
            return {'regime': 'insufficient_data', 'volatility': None}
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        historical_vol = returns.rolling(window=self.lookback).std() * np.sqrt(252)
        
        if len(rolling_vol.dropna()) == 0:
            return {'regime': 'insufficient_data', 'volatility': None}
        
        current_vol = rolling_vol.iloc[-1]
        avg_vol = historical_vol.mean()
        vol_percentile = (rolling_vol < current_vol).mean()
        
        # Classify volatility regime
        if current_vol < avg_vol * 0.5:
            regime = 'extremely_low_volatility'
            is_favorable = False  # Too quiet, may be before storm
        elif current_vol < avg_vol * 0.8:
            regime = 'low_volatility'
            is_favorable = True
        elif current_vol < avg_vol * 1.5:
            regime = 'normal_volatility'
            is_favorable = True
        elif current_vol < avg_vol * 2.0:
            regime = 'high_volatility'
            is_favorable = False
        else:
            regime = 'extreme_volatility'
            is_favorable = False
        
        return {
            'regime': regime,
            'current_volatility': current_vol,
            'average_volatility': avg_vol,
            'volatility_percentile': vol_percentile,
            'is_favorable': is_favorable
        }
    
    def _trend_regime_detection(self, prices: pd.Series) -> Dict:
        """Detect trend strength using multiple timeframes"""
        if len(prices) < 50:
            return {'regime': 'insufficient_data', 'trend_strength': None}
        
        # Calculate moving averages
        sma_10 = prices.rolling(window=10).mean()
        sma_20 = prices.rolling(window=20).mean()
        sma_50 = prices.rolling(window=50).mean()
        
        current_price = prices.iloc[-1]
        
        # Calculate trend alignment
        if len(sma_50.dropna()) == 0:
            return {'regime': 'insufficient_data', 'trend_strength': None}
        
        # Get current values
        ma_10 = sma_10.iloc[-1]
        ma_20 = sma_20.iloc[-1]
        ma_50 = sma_50.iloc[-1]
        
        # Check alignment
        bullish_alignment = (current_price > ma_10 > ma_20 > ma_50)
        bearish_alignment = (current_price < ma_10 < ma_20 < ma_50)
        
        # Calculate trend strength
        if bullish_alignment:
            regime = 'strong_uptrend'
            strength = ((current_price - ma_50) / ma_50) * 100
            is_favorable = False
        elif bearish_alignment:
            regime = 'strong_downtrend'
            strength = ((ma_50 - current_price) / ma_50) * 100
            is_favorable = False
        else:
            # Check for weak trend
            if current_price > ma_50 and ma_10 > ma_50:
                regime = 'weak_uptrend'
                strength = ((current_price - ma_50) / ma_50) * 100
            elif current_price < ma_50 and ma_10 < ma_50:
                regime = 'weak_downtrend'
                strength = ((ma_50 - current_price) / ma_50) * 100
            else:
                regime = 'no_clear_trend'
                strength = 0
            is_favorable = True
        
        return {
            'regime': regime,
            'trend_strength': strength,
            'is_favorable': is_favorable
        }
    
    def _reversion_speed_detection(self, prices: pd.Series) -> Dict:
        """Detect how quickly prices revert to mean"""
        if len(prices) < self.lookback:
            return {'regime': 'insufficient_data', 'half_life': None}
        
        # Calculate half-life
        from sklearn.linear_model import LinearRegression
        
        price_lag = prices.shift(1).dropna()
        price_diff = prices.diff().dropna()
        
        if len(price_diff) < 20:
            return {'regime': 'insufficient_data', 'half_life': None}
        
        price_lag = price_lag[price_diff.index]
        
        model = LinearRegression()
        X = price_lag.values.reshape(-1, 1)
        y = price_diff.values
        
        model.fit(X, y)
        beta = model.coef_[0]
        
        if beta < 0:
            half_life = -np.log(2) / beta
            
            if half_life < 5:
                regime = 'fast_mean_reversion'
                is_favorable = True
            elif half_life < 20:
                regime = 'normal_mean_reversion'
                is_favorable = True
            elif half_life < 50:
                regime = 'slow_mean_reversion'
                is_favorable = True
            else:
                regime = 'very_slow_mean_reversion'
                is_favorable = False
        else:
            half_life = None
            regime = 'no_mean_reversion'
            is_favorable = False
        
        return {
            'regime': regime,
            'half_life': half_life,
            'beta': beta,
            'is_favorable': is_favorable
        }
    
    def _detect_structural_breaks(self, prices: pd.Series) -> Dict:
        """Detect structural breaks in the time series"""
        if len(prices) < self.lookback * 2:
            return {'has_break': False, 'break_date': None}
        
        returns = prices.pct_change().dropna()
        
        # Calculate rolling statistics
        rolling_mean = returns.rolling(window=self.lookback).mean()
        rolling_std = returns.rolling(window=self.lookback).std()
        
        # Detect significant changes
        mean_change = abs(rolling_mean.diff()).fillna(0)
        std_change = abs(rolling_std.diff()).fillna(0)
        
        # Define thresholds for structural break
        mean_threshold = mean_change.quantile(0.95)
        std_threshold = std_change.quantile(0.95)
        
        # Check for recent breaks
        recent_window = 20
        recent_mean_break = (mean_change.tail(recent_window) > mean_threshold).any()
        recent_std_break = (std_change.tail(recent_window) > std_threshold).any()
        
        has_break = recent_mean_break or recent_std_break
        
        return {
            'has_break': has_break,
            'mean_stable': not recent_mean_break,
            'variance_stable': not recent_std_break,
            'is_favorable': not has_break
        }
    
    def _combine_regime_indicators(self, results: Dict) -> str:
        """Combine multiple regime indicators into final assessment"""
        votes = {
            'mean_reverting': 0,
            'ranging': 0,
            'trending': 0,
            'unstable': 0
        }
        
        weights = {
            'hurst': 3,      # Most important
            'volatility': 2,
            'trend': 2,
            'reversion': 2,
            'structural': 1
        }
        
        # Hurst regime votes
        hurst_regime = results.get('hurst_regime', {}).get('regime', '')
        if 'mean_reverting' in hurst_regime:
            votes['mean_reverting'] += weights['hurst']
        elif 'ranging' in hurst_regime:
            votes['ranging'] += weights['hurst']
        elif 'trending' in hurst_regime:
            votes['trending'] += weights['hurst']
        
        # Volatility regime votes
        vol_favorable = results.get('volatility_regime', {}).get('is_favorable', False)
        if vol_favorable:
            votes['ranging'] += weights['volatility']
        else:
            vol_regime = results.get('volatility_regime', {}).get('regime', '')
            if 'extreme' in vol_regime:
                votes['unstable'] += weights['volatility']
        
        # Trend regime votes
        trend_regime = results.get('trend_regime', {}).get('regime', '')
        if 'strong' in trend_regime and 'trend' in trend_regime:
            votes['trending'] += weights['trend']
        elif 'no_clear_trend' in trend_regime:
            votes['ranging'] += weights['trend']
        
        # Reversion speed votes
        reversion_favorable = results.get('reversion_speed', {}).get('is_favorable', False)
        if reversion_favorable:
            votes['mean_reverting'] += weights['reversion']
        else:
            reversion_regime = results.get('reversion_speed', {}).get('regime', '')
            if 'no_mean_reversion' in reversion_regime:
                votes['trending'] += weights['reversion']
        
        # Structural break votes
        has_break = results.get('structural_breaks', {}).get('has_break', False)
        if has_break:
            votes['unstable'] += weights['structural'] * 2
        
        # Determine final regime
        if votes['unstable'] >= 3:
            return 'unstable'
        
        max_votes = max(votes.values())
        if max_votes == 0:
            return 'unknown'
        
        for regime, count in votes.items():
            if count == max_votes:
                return regime
        
        return 'unknown'
    
    def _get_recommendation(self, regime: str) -> str:
        """Get trading recommendation based on regime"""
        recommendations = {
            'mean_reverting': "Excellent conditions for mean reversion. Trade with normal position sizes.",
            'ranging': "Good conditions for mean reversion. Consider slightly reduced position sizes.",
            'trending': "STOP TRADING. Strong trend detected. Mean reversion will likely fail.",
            'unstable': "STOP TRADING. Market regime unstable. Wait for conditions to stabilize.",
            'unknown': "Insufficient data to determine regime. Trade with extreme caution or wait."
        }
        
        return recommendations.get(regime, "Unable to determine recommendation.")
