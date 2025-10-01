"""
Position sizing algorithms for mean reversion strategies
Conservative implementations with safety factors
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

class PositionSizer:
    """
    Advanced position sizing methods with emphasis on capital preservation
    """
    
    @staticmethod
    def kelly_criterion(win_rate: float,
                       avg_win: float,
                       avg_loss: float,
                       safety_factor: float = 0.25) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        ALWAYS use safety factor to avoid ruin
        
        Parameters:
        -----------
        win_rate : float
            Historical win rate (0 to 1)
        avg_win : float
            Average winning trade return (positive)
        avg_loss : float
            Average losing trade return (negative)
        safety_factor : float
            Fraction of Kelly to use (default 0.25 for safety)
            
        Returns:
        --------
        float : Recommended position size as fraction of capital
        """
        # Validate inputs
        if avg_loss >= 0:
            print("Warning: avg_loss should be negative")
            return 0.0
        
        if avg_win <= 0:
            print("Warning: avg_win should be positive")
            return 0.0
        
        if win_rate <= 0 or win_rate >= 1:
            print("Warning: win_rate should be between 0 and 1")
            return 0.0
        
        # Kelly formula: f = (p*b - q)/b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_f = (p * b - q) / b
        
        # Kelly can suggest betting more than you have (>100%)
        # This is obviously impossible and dangerous
        if kelly_f > 1.0:
            print(f"Warning: Raw Kelly suggests {kelly_f:.1%} - capping at 100%")
            kelly_f = 1.0
        
        # Negative Kelly means don't trade at all
        if kelly_f < 0:
            print("Warning: Negative Kelly - strategy has negative edge")
            return 0.0
        
        # Apply safety factor (critical for survival)
        safe_f = kelly_f * safety_factor
        
        # Never risk more than 25% on a single position
        max_position = 0.25
        final_size = min(safe_f, max_position)
        
        return final_size
    
    @staticmethod
    def volatility_scaled_sizing(current_volatility: float,
                                target_volatility: float = 0.15,
                                base_size: float = 0.02,
                                max_leverage: float = 1.0) -> float:
        """
        Scale position size inversely to volatility
        Maintains consistent risk regardless of market conditions
        
        Parameters:
        -----------
        current_volatility : float
            Current market volatility (annualized)
        target_volatility : float
            Target portfolio volatility (default 15%)
        base_size : float
            Base position size (default 2%)
        max_leverage : float
            Maximum leverage allowed (default 1.0 = no leverage)
            
        Returns:
        --------
        float : Adjusted position size
        """
        if current_volatility <= 0:
            print("Warning: Invalid volatility")
            return 0.0
        
        # Scale position inversely to volatility
        vol_scalar = target_volatility / current_volatility
        
        # Apply base size
        position_size = base_size * vol_scalar
        
        # Apply leverage limit
        max_size = base_size * max_leverage
        
        # Also apply absolute maximum
        absolute_max = 0.10  # Never more than 10% in one position
        
        final_size = min(position_size, max_size, absolute_max)
        
        return final_size
    
    @staticmethod
    def fixed_fractional_position(capital: float,
                                 risk_per_trade: float,
                                 entry_price: float,
                                 stop_price: float) -> int:
        """
        Calculate position size based on fixed risk per trade
        Most conservative approach - risk fixed percentage
        
        Parameters:
        -----------
        capital : float
            Total capital available
        risk_per_trade : float
            Fraction of capital to risk per trade (e.g., 0.02 for 2%)
        entry_price : float
            Entry price for the trade
        stop_price : float
            Stop loss price
            
        Returns:
        --------
        int : Number of shares to trade
        """
        if entry_price <= 0 or stop_price <= 0:
            return 0
        
        if capital <= 0:
            return 0
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share == 0:
            print("Warning: No risk defined (entry = stop)")
            return 0
        
        # Calculate maximum dollar risk
        max_dollar_risk = capital * risk_per_trade
        
        # Calculate shares
        shares = max_dollar_risk / risk_per_share
        
        # Check if position size exceeds capital
        position_value = shares * entry_price
        if position_value > capital:
            # Adjust to maximum affordable
            shares = capital * 0.95 / entry_price  # Keep 5% buffer
            actual_risk = (shares * risk_per_share) / capital
            print(f"Warning: Position size limited by capital. Actual risk: {actual_risk:.2%}")
        
        return int(shares)
    
    @staticmethod
    def risk_parity_allocation(returns_matrix: pd.DataFrame,
                              target_risk: float = 0.10,
                              max_position: float = 0.25) -> Dict[str, float]:
        """
        Allocate capital using risk parity approach
        Equal risk contribution from each position
        
        Parameters:
        -----------
        returns_matrix : pd.DataFrame
            Returns for multiple assets (columns = assets)
        target_risk : float
            Target portfolio risk (default 10% annually)
        max_position : float
            Maximum weight per asset (default 25%)
            
        Returns:
        --------
        dict : Asset weights
        """
        if returns_matrix.empty:
            return {}
        
        # Calculate covariance matrix (annualized)
        cov_matrix = returns_matrix.cov() * 252
        
        # Get volatilities (diagonal of covariance matrix)
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        # Check for zero volatilities
        if any(vol <= 0 for vol in volatilities):
            print("Warning: Zero or negative volatility detected")
            return {}
        
        # Inverse volatility weighting (simplified risk parity)
        inv_vols = 1 / volatilities
        raw_weights = inv_vols / inv_vols.sum()
        
        # Scale to target risk
        portfolio_vol = np.sqrt(raw_weights @ cov_matrix @ raw_weights.T)
        
        if portfolio_vol > 0:
            scaling_factor = target_risk / portfolio_vol
            scaled_weights = raw_weights * scaling_factor
        else:
            scaled_weights = raw_weights
        
        # Apply position limits
        final_weights = np.minimum(scaled_weights, max_position)
        
        # Renormalize if we hit limits
        if final_weights.sum() > 1.0:
            final_weights = final_weights / final_weights.sum()
        
        # Create weight dictionary
        weights_dict = {}
        for i, col in enumerate(returns_matrix.columns):
            weights_dict[col] = final_weights[i]
        
        return weights_dict
    
    @staticmethod
    def calculate_optimal_f(trade_returns: pd.Series,
                           initial_capital: float = 10000) -> Dict:
        """
        Calculate optimal f (fraction) using Ralph Vince's method
        Warning: Optimal f is often too aggressive for real trading
        
        Parameters:
        -----------
        trade_returns : pd.Series
            Historical trade returns (as percentages)
        initial_capital : float
            Starting capital
            
        Returns:
        --------
        dict : Optimal f and related statistics
        """
        if len(trade_returns) == 0:
            return {
                'optimal_f': 0,
                'expected_growth': 0,
                'recommendation': 'No trade history available'
            }
        
        # Convert returns to profit/loss amounts
        pnl = trade_returns * initial_capital
        
        # Find the largest loss
        largest_loss = abs(min(pnl)) if min(pnl) < 0 else 0
        
        if largest_loss == 0:
            return {
                'optimal_f': 0,
                'expected_growth': 0,
                'recommendation': 'No losses in history - suspicious data'
            }
        
        # Test different f values
        f_values = np.linspace(0.01, 0.99, 99)
        twrs = []  # Terminal Wealth Relatives
        
        for f in f_values:
            twr = 1.0
            for profit in pnl:
                # Holding Period Return
                hpr = 1 + (f * profit / largest_loss)
                if hpr <= 0:
                    twr = 0
                    break
                twr *= hpr
            twrs.append(twr)
        
        # Find optimal f
        optimal_idx = np.argmax(twrs)
        optimal_f = f_values[optimal_idx]
        
        # Calculate expected growth rate
        if twrs[optimal_idx] > 0 and len(pnl) > 0:
            expected_growth = np.log(twrs[optimal_idx]) / len(pnl)
        else:
            expected_growth = 0
        
        # Safety recommendation
        safe_f = optimal_f * 0.25  # Use 25% of optimal f
        
        return {
            'optimal_f': optimal_f,
            'safe_f': safe_f,
            'expected_growth': expected_growth,
            'largest_loss': largest_loss,
            'terminal_wealth': twrs[optimal_idx],
            'recommendation': f'Use {safe_f:.1%} of capital (25% of optimal f)'
        }
    
    @staticmethod
    def dynamic_position_sizing(capital: float,
                              recent_performance: List[float],
                              base_size: float = 0.02,
                              increase_after_wins: int = 3,
                              decrease_after_losses: int = 2) -> float:
        """
        Adjust position size based on recent performance
        Anti-martingale approach: increase after wins, decrease after losses
        
        Parameters:
        -----------
        capital : float
            Current capital
        recent_performance : List[float]
            Recent trade returns (last N trades)
        base_size : float
            Base position size
        increase_after_wins : int
            Number of consecutive wins to increase size
        decrease_after_losses : int
            Number of consecutive losses to decrease size
            
        Returns:
        --------
        float : Adjusted position size
        """
        if not recent_performance:
            return base_size
        
        # Check recent streak
        recent_wins = 0
        recent_losses = 0
        
        for ret in reversed(recent_performance):
            if ret > 0:
                recent_wins += 1
                recent_losses = 0
            else:
                recent_losses += 1
                recent_wins = 0
            
            # Only look at current streak
            if recent_wins == 0 and recent_losses == 0:
                break
        
        # Adjust size based on streak
        if recent_wins >= increase_after_wins:
            # Increase size after consecutive wins (momentum)
            adjustment = 1.25
            print(f"Increasing position size after {recent_wins} wins")
        elif recent_losses >= decrease_after_losses:
            # Decrease size after consecutive losses (protection)
            adjustment = 0.5
            print(f"Decreasing position size after {recent_losses} losses")
        else:
            adjustment = 1.0
        
        adjusted_size = base_size * adjustment
        
        # Apply limits
        min_size = 0.005  # 0.5% minimum
        max_size = 0.05   # 5% maximum
        
        final_size = max(min(adjusted_size, max_size), min_size)
        
        return final_size
