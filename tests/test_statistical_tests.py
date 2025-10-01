"""
Unit tests for statistical tests module
"""
import pytest
import numpy as np
import pandas as pd
from src.statistical_tests import MeanReversionTests

class TestMeanReversionTests:
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.tests = MeanReversionTests()
        
        # Create mean-reverting series (Ornstein-Uhlenbeck process)
        n = 1000
        theta = 0.1  # mean reversion speed
        mu = 100     # long-term mean
        sigma = 2    # volatility
        
        prices = [mu]
        for i in range(1, n):
            drift = theta * (mu - prices[-1])
            diffusion = sigma * np.random.normal()
            prices.append(prices[-1] + drift + diffusion)
        
        self.mean_reverting_series = pd.Series(prices)
        
        # Create trending series
        trend = np.cumsum(np.random.normal(0.1, 1, n))
        self.trending_series = pd.Series(100 + trend)
        
    def test_adf_mean_reverting(self):
        """Test ADF on mean-reverting series"""
        result = self.tests.augmented_dickey_fuller(self.mean_reverting_series)
        assert result['is_stationary'] == True
        assert result['p_value'] < 0.05
        
    def test_adf_trending(self):
        """Test ADF on trending series"""
        result = self.tests.augmented_dickey_fuller(self.trending_series)
        assert result['is_stationary'] == False
        assert result['p_value'] > 0.05
        
    def test_hurst_exponent(self):
        """Test Hurst exponent calculation"""
        # Mean reverting should have H < 0.5
        hurst_mr = self.tests.calculate_hurst_exponent(self.mean_reverting_series)
        assert hurst_mr['hurst_exponent'] < 0.55
        
        # Trending should have H > 0.5
        hurst_trend = self.tests.calculate_hurst_exponent(self.trending_series)
        assert hurst_trend['hurst_exponent'] > 0.45
        
    def test_half_life(self):
        """Test half-life calculation"""
        result = self.tests.calculate_half_life(self.mean_reverting_series)
        assert result['half_life'] is not None
        assert result['half_life'] > 0
        assert result['is_mean_reverting'] == True
        
    def test_cointegration(self):
        """Test cointegration between two series"""
        # Create cointegrated pair
        series1 = self.mean_reverting_series
        series2 = series1 + np.random.normal(0, 0.5, len(series1))
        
        result = self.tests.test_cointegration(series1, series2)
        assert 'hedge_ratio' in result
        assert result['hedge_ratio'] > 0

if __name__ == "__main__":
    pytest.main([__file__])
