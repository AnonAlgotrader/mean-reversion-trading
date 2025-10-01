"""
Statistical tests for mean reversion validation
Complete implementation with all tests from the article
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from typing import Dict, Tuple, Optional
import warnings
from sklearn.linear_model import LinearRegression

class MeanReversionTests:
    """
    Comprehensive statistical tests for mean reversion
    """
    
    @staticmethod
    def augmented_dickey_fuller(prices: pd.Series, 
                               significance_level: float = 0.05) -> Dict:
        """
        Test for stationarity using ADF test
        
        Parameters:
        -----------
        prices : pd.Series
            Price series to test (not returns!)
        significance_level : float
            Significance level for test (default 0.05)
            
        Returns:
        --------
        dict : Test results including statistic, p-value, and interpretation
        """
        if len(prices) < 20:
            warnings.warn("Insufficient data for reliable ADF test (n < 20)")
            
        result = adfuller(prices.dropna(), autolag='AIC')
        
        adf_statistic = result[0]
        p_value = result[1]
        used_lags = result[2]
        n_obs = result[3]
        critical_values = result[4]
        
        is_stationary = p_value < significance_level
        
        interpretation = f"Series is {'stationary (mean-reverting)' if is_stationary else 'non-stationary (not mean-reverting)'} at {significance_level:.0%} significance"
        
        return {
            'test_name': 'Augmented Dickey-Fuller',
            'is_stationary': is_stationary,
            'is_mean_reverting': is_stationary,
            'p_value': p_value,
            'adf_statistic': adf_statistic,
            'used_lags': used_lags,
            'n_observations': n_obs,
            'critical_values': critical_values,
            'significance_level': significance_level,
            'interpretation': interpretation,
            'warning': 'Stationarity is necessary but not sufficient for profitable trading'
        }
    
    @staticmethod
    def calculate_hurst_exponent(prices: pd.Series, 
                                 min_lag: int = 2,
                                 max_lag: int = 100) -> Dict:
        """
        Calculate Hurst Exponent to measure mean reversion tendency
        
        H < 0.5 : Mean reverting
        H = 0.5 : Random walk  
        H > 0.5 : Trending
        
        Warning: Financial markets rarely show consistent H < 0.5
        """
        if len(prices) < max_lag * 2:
            max_lag = len(prices) // 2
            
        lags = range(min_lag, max_lag)
        
        # Calculate log returns
        log_prices = np.log(prices.dropna())
        
        # Calculate variance of differences for each lag
        variances = []
        for lag in lags:
            differences = log_prices[lag:].to_numpy() - log_prices[:-lag].to_numpy()
            variances.append(np.var(differences))
        
        # Fit linear regression to log-log plot
        log_lags = np.log(list(lags))
        log_variances = np.log(variances)
        
        # Calculate slope (which is 2H)
        coeffs = np.polyfit(log_lags, log_variances, 1)
        hurst = coeffs[0] / 2
        
        # Determine behavior
        if hurst < 0.45:
            behavior = "Strong mean reversion"
        elif hurst < 0.5:
            behavior = "Weak mean reversion"
        elif hurst < 0.55:
            behavior = "Random walk"
        elif hurst < 0.65:
            behavior = "Weak trending"
        else:
            behavior = "Strong trending"
        
        # Calculate confidence
        n_observations = len(prices)
        std_error = 1 / np.sqrt(n_observations)
        
        if abs(hurst - 0.5) > 2 * std_error:
            confidence = "High confidence"
        elif abs(hurst - 0.5) > std_error:
            confidence = "Moderate confidence"
        else:
            confidence = "Low confidence"
        
        return {
            'hurst_exponent': hurst,
            'behavior': behavior,
            'is_mean_reverting': hurst < 0.5,
            'confidence': confidence,
            'min_lag_used': min_lag,
            'max_lag_used': max_lag,
            'interpretation': f"{behavior} with {confidence.lower()}"
        }
    
    @staticmethod
    def calculate_half_life(prices: pd.Series) -> Dict:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process
        Critical for determining holding periods and position sizing
        """
        price_lag = prices.shift(1).dropna()
        price_diff = prices.diff().dropna()
        
        # Align series
        price_lag = price_lag[price_diff.index]
        
        # Run linear regression: price_diff = beta * price_lag + const
        model = LinearRegression()
        X = price_lag.values.reshape(-1, 1)
        y = price_diff.values
        
        model.fit(X, y)
        beta = model.coef_[0]
        r_squared = model.score(X, y)
        
        if beta < 0:  # Necessary for mean reversion
            half_life = -np.log(2) / beta
            
            return {
                'half_life': half_life,
                'beta': beta,
                'r_squared': r_squared,
                'is_mean_reverting': True,
                'lookback_recommendation': int(np.ceil(half_life)),
                'max_holding_recommendation': int(half_life * 3),
                'interpretation': f"Prices revert halfway to mean in {half_life:.1f} periods"
            }
        else:
            return {
                'half_life': None,
                'beta': beta,
                'r_squared': r_squared,
                'is_mean_reverting': False,
                'lookback_recommendation': None,
                'max_holding_recommendation': None,
                'interpretation': "No mean reversion detected (beta >= 0)"
            }
    
    @staticmethod
    def test_cointegration(price1: pd.Series, 
                          price2: pd.Series,
                          significance_level: float = 0.05) -> Dict:
        """
        Test if two price series are cointegrated (for pairs trading)
        """
        # Engle-Granger test
        score, p_value, critical_values = coint(price1.dropna(), price2.dropna())
        
        is_cointegrated = p_value < significance_level
        
        # Calculate hedge ratio via OLS
        model = LinearRegression()
        X = price2.values.reshape(-1, 1)
        y = price1.values
        model.fit(X, y)
        hedge_ratio = model.coef_[0]
        
        # Calculate spread
        spread = price1 - hedge_ratio * price2
        
        # Test if spread is stationary
        tests = MeanReversionTests()
        spread_adf = tests.augmented_dickey_fuller(spread)
        spread_half_life = tests.calculate_half_life(spread)
        
        interpretation = f"Series are {'cointegrated' if is_cointegrated else 'not cointegrated'} (p={p_value:.4f}). "
        interpretation += "Suitable for pairs trading." if is_cointegrated else "Not suitable for pairs trading."
        
        return {
            'is_cointegrated': is_cointegrated,
            'p_value': p_value,
            'test_statistic': score,
            'critical_values': critical_values,
            'hedge_ratio': hedge_ratio,
            'spread_is_stationary': spread_adf['is_stationary'],
            'spread_half_life': spread_half_life['half_life'],
            'interpretation': interpretation
        }
    
    def run_all_tests(self, prices: pd.Series) -> Dict:
        """
        Run all mean reversion tests on a price series
        """
        results = {
            'adf_test': self.augmented_dickey_fuller(prices),
            'hurst_exponent': self.calculate_hurst_exponent(prices),
            'half_life': self.calculate_half_life(prices)
        }
        
        # Create summary
        adf = results['adf_test']
        hurst = results['hurst_exponent']
        half_life = results['half_life']
        
        # Count positive indicators
        indicators = [
            adf['is_mean_reverting'],
            hurst['is_mean_reverting'],
            half_life['is_mean_reverting']
        ]
        
        score = sum(indicators)
        
        if score == 3:
            conclusion = "Strong evidence of mean reversion"
            recommendation = "Suitable for mean reversion strategies"
        elif score == 2:
            conclusion = "Moderate evidence of mean reversion"
            recommendation = "Proceed with caution, monitor regime changes"
        elif score == 1:
            conclusion = "Weak evidence of mean reversion"
            recommendation = "Not recommended for mean reversion strategies"
        else:
            conclusion = "No evidence of mean reversion"
            recommendation = "Avoid mean reversion strategies for this asset"
        
        results['summary'] = {
            'mean_reversion_score': f"{score}/3",
            'conclusion': conclusion,
            'recommendation': recommendation,
            'tests_passed': {
                'ADF': adf['is_mean_reverting'],
                'Hurst': hurst['is_mean_reverting'],
                'Half-life': half_life['is_mean_reverting']
            }
        }
        
        return results
