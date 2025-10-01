"""
Transaction cost analysis and optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class TransactionCostAnalyzer:
    """
    Analyze and optimize for transaction costs
    """
    
    @staticmethod
    def calculate_break_even_move(commission: float = 0.001,
                                 spread: float = 0.0005,
                                 slippage: float = 0.0005) -> Dict:
        """
        Calculate minimum profitable price move
        
        Parameters:
        -----------
        commission : float
            Commission per trade
        spread : float
            Bid-ask spread
        slippage : float
            Expected slippage
            
        Returns:
        --------
        dict : Break-even analysis
        """
        # Round trip costs
        total_cost = 2 * (commission + spread/2 + slippage)
        
        # Calculate for different holding periods
        holding_periods = [1, 5, 10, 20]  # days
        annual_trading_days = 252
        
        results = {}
        for days in holding_periods:
            trades_per_year = annual_trading_days / days
            annual_cost = total_cost * trades_per_year
            
            results[f'{days}_day'] = {
                'break_even_move': total_cost,
                'annual_cost': annual_cost,
                'required_annual_return': annual_cost
            }
        
        return {
            'total_round_trip_cost': total_cost,
            'break_even_by_period': results,
            'recommendation': self._get_cost_recommendation(total_cost)
        }
    
    @staticmethod
    def _get_cost_recommendation(total_cost: float) -> str:
        """Get recommendation based on cost levels"""
        if total_cost < 0.002:
            return "Low costs - suitable for frequent trading"
        elif total_cost < 0.005:
            return "Moderate costs - focus on larger moves"
        else:
            return "High costs - consider longer holding periods"
    
    @staticmethod
    def optimize_trade_frequency(strategy_returns: pd.Series,
                                current_frequency: int,
                                transaction_cost: float) -> Dict:
        """
        Find optimal trading frequency given costs
        
        Parameters:
        -----------
        strategy_returns : pd.Series
            Historical returns per trade
        current_frequency : int
            Current trades per year
        transaction_cost : float
            Cost per trade
            
        Returns:
        --------
        dict : Optimal frequency analysis
        """
        # Test different frequencies
        frequencies = [10, 25, 50, 100, 200, 252]  # trades per year
        
        results = []
        for freq in frequencies:
            # Estimate returns at this frequency
            # Assumption: fewer trades = larger moves
            scale_factor = np.sqrt(current_frequency / freq)
            scaled_returns = strategy_returns * scale_factor
            
            # Calculate net returns after costs
            avg_gross = scaled_returns.mean()
            net_return = avg_gross - transaction_cost
            
            # Annual return
            annual_return = net_return * freq
            
            # Sharpe ratio (simplified)
            if scaled_returns.std() > 0:
                sharpe = net_return / scaled_returns.std() * np.sqrt(freq)
            else:
                sharpe = 0
            
            results.append({
                'frequency': freq,
                'avg_gross_return': avg_gross,
                'avg_net_return': net_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe
            })
        
        # Find optimal frequency
        best_idx = np.argmax([r['sharpe_ratio'] for r in results])
        optimal = results[best_idx]
        
        return {
            'current_frequency': current_frequency,
            'optimal_frequency': optimal['frequency'],
            'frequency_analysis': results,
            'recommendation': f"Optimal frequency: {optimal['frequency']} trades/year"
        }
    
    @staticmethod
    def calculate_implementation_shortfall(ideal_prices: pd.Series,
                                          actual_fills: pd.Series) -> Dict:
        """
        Calculate implementation shortfall (slippage analysis)
        
        Parameters:
        -----------
        ideal_prices : pd.Series
            Theoretical/signal prices
        actual_fills : pd.Series
            Actual execution prices
            
        Returns:
        --------
        dict : Implementation shortfall metrics
        """
        # Calculate shortfall
        shortfall = (actual_fills - ideal_prices) / ideal_prices
        
        return {
            'mean_shortfall': shortfall.mean(),
            'median_shortfall': shortfall.median(),
            'worst_shortfall': shortfall.min(),
            'best_shortfall': shortfall.max(),
            'total_cost': shortfall.sum(),
            'shortfall_volatility': shortfall.std(),
            'recommendation': self._get_shortfall_recommendation(shortfall.mean())
        }
    
    @staticmethod
    def _get_shortfall_recommendation(mean_shortfall: float) -> str:
        """Get recommendation based on shortfall"""
        if abs(mean_shortfall) < 0.001:
            return "Excellent execution quality"
        elif abs(mean_shortfall) < 0.003:
            return "Good execution - consider limit orders for improvement"
        else:
            return "Poor execution - review order types and timing"
