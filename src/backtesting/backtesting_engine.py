"""
Comprehensive backtesting engine with realistic transaction costs and slippage
Production-ready implementation with all pitfalls addressed
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
import warnings
from copy import deepcopy

class BacktestEngine:
    """
    Realistic backtesting with transaction costs, slippage, and market impact
    Addresses all common backtesting pitfalls from the article
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 spread: float = 0.0005,
                 slippage: float = 0.0005,
                 market_impact: float = 0.0002,
                 interest_rate: float = 0.02,
                 max_positions: int = 5):
        """
        Initialize backtest engine with realistic costs
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        commission : float
            Commission per trade (as fraction)
        spread : float
            Bid-ask spread (as fraction)
        slippage : float
            Expected slippage (as fraction)
        market_impact : float
            Price impact of trades (as fraction)
        interest_rate : float
            Annual interest rate for cash (default 2%)
        max_positions : int
            Maximum concurrent positions
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.spread = spread
        self.slippage = slippage
        self.market_impact = market_impact
        self.interest_rate = interest_rate
        self.max_positions = max_positions
        
        # Calculate total transaction cost
        self.total_transaction_cost = commission + spread/2 + slippage + market_impact
        
        # State tracking
        self.reset()
        
    def reset(self):
        """Reset engine state for new backtest"""
        self.capital = self.initial_capital
        self.positions = {}  # Dict of open positions by entry date
        self.trades = []     # List of all executed trades
        self.equity_curve = []
        self.daily_returns = []
        self.warnings = []
        self.stats = {}
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for common issues
        
        Returns:
        --------
        bool : True if data is valid, False otherwise
        """
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required_columns if col not in data.columns]
        
        if missing:
            self.warnings.append(f"Missing required columns: {missing}")
            return False
        
        # Check for NaN values
        nan_count = data[required_columns].isnull().sum().sum()
        if nan_count > 0:
            self.warnings.append(f"Data contains {nan_count} NaN values")
            
        # Check for data consistency
        invalid_bars = (
            (data['High'] < data['Low']) |
            (data['High'] < data['Close']) |
            (data['High'] < data['Open']) |
            (data['Low'] > data['Close']) |
            (data['Low'] > data['Open'])
        ).sum()
        
        if invalid_bars > 0:
            self.warnings.append(f"Found {invalid_bars} invalid OHLC bars")
            
        # Check for sufficient data
        if len(data) < 100:
            self.warnings.append("Insufficient data (<100 bars) for reliable backtesting")
            
        # Check index
        if not data.index.is_monotonic_increasing:
            self.warnings.append("Data index is not monotonic increasing")
            
        return len(self.warnings) == 0
    
    def calculate_realistic_fill_price(self, 
                                      signal_price: float,
                                      action: str,
                                      shares: int) -> float:
        """
        Calculate realistic fill price including all market frictions
        
        Parameters:
        -----------
        signal_price : float
            Price at signal generation
        action : str
            'buy', 'sell', 'short', 'cover'
        shares : int
            Number of shares (for market impact)
            
        Returns:
        --------
        float : Realistic fill price
        """
        # Base slippage (always adverse)
        if action in ['buy', 'cover']:
            # Buying - price goes up
            base_slippage = signal_price * (1 + self.slippage)
        else:  # sell, short
            # Selling - price goes down
            base_slippage = signal_price * (1 - self.slippage)
        
        # Add spread cost (half spread on each side)
        spread_cost = signal_price * (self.spread / 2)
        if action in ['buy', 'cover']:
            with_spread = base_slippage + spread_cost
        else:
            with_spread = base_slippage - spread_cost
        
        # Add market impact (proportional to size)
        # Larger orders move the market more
        size_impact = min(shares / 10000, 0.01)  # Cap at 1% impact
        market_impact_cost = signal_price * self.market_impact * (1 + size_impact)
        
        if action in ['buy', 'cover']:
            final_price = with_spread + market_impact_cost
        else:
            final_price = with_spread - market_impact_cost
        
        return final_price
    
    def enter_position(self, 
                      date: pd.Timestamp,
                      signal_price: float,
                      position_type: str,
                      shares: int,
                      reason: str = None) -> Dict:
        """
        Enter a new position with realistic execution
        """
        if shares <= 0:
            return None
        
        # Check position limits
        if len(self.positions) >= self.max_positions:
            self.warnings.append(f"Max positions reached on {date}")
            return None
        
        # Calculate realistic fill
        if position_type == 'long':
            fill_price = self.calculate_realistic_fill_price(signal_price, 'buy', shares)
        else:  # short
            fill_price = self.calculate_realistic_fill_price(signal_price, 'short', shares)
        
        # Calculate costs
        trade_value = shares * fill_price
        commission_cost = trade_value * self.commission
        total_cost = trade_value + commission_cost
        
        # Check capital constraints
        if total_cost > self.capital:
            # Reduce position size to fit capital
            max_shares = int((self.capital * 0.95) / (fill_price * (1 + self.commission)))
            if max_shares <= 0:
                self.warnings.append(f"Insufficient capital for trade on {date}")
                return None
            shares = max_shares
            trade_value = shares * fill_price
            commission_cost = trade_value * self.commission
            total_cost = trade_value + commission_cost
        
        # Execute trade
        self.capital -= total_cost
        
        position = {
            'entry_date': date,
            'type': position_type,
            'entry_price': fill_price,
            'signal_price': signal_price,
            'shares': shares,
            'trade_value': trade_value,
            'commission_paid': commission_cost,
            'slippage': fill_price - signal_price,
            'reason': reason
        }
        
        self.positions[date] = position
        
        # Record trade
        self.trades.append({
            'date': date,
            'action': f'{position_type}_entry',
            'signal_price': signal_price,
            'fill_price': fill_price,
            'shares': shares,
            'value': trade_value,
            'commission': commission_cost,
            'slippage': fill_price - signal_price,
            'capital_after': self.capital
        })
        
        return position
    
    def exit_position(self,
                     exit_date: pd.Timestamp,
                     position: Dict,
                     signal_price: float,
                     reason: str) -> Dict:
        """
        Exit a position with realistic execution
        """
        position_type = position['type']
        shares = position['shares']
        
        # Calculate realistic fill
        if position_type == 'long':
            fill_price = self.calculate_realistic_fill_price(signal_price, 'sell', shares)
        else:  # short
            fill_price = self.calculate_realistic_fill_price(signal_price, 'cover', shares)
        
        # Calculate proceeds and costs
        trade_value = shares * fill_price
        commission_cost = trade_value * self.commission
        
        # Calculate P&L
        if position_type == 'long':
            gross_pnl = (fill_price - position['entry_price']) * shares
            net_proceeds = trade_value - commission_cost
            self.capital += net_proceeds
        else:  # short
            gross_pnl = (position['entry_price'] - fill_price) * shares
            # For short: return borrowed shares and keep profit/loss
            net_pnl = gross_pnl - commission_cost - position['commission_paid']
            self.capital += position['trade_value'] + net_pnl
        
        net_pnl = gross_pnl - commission_cost - position['commission_paid']
        
        # Record trade
        trade = {
            'date': exit_date,
            'action': f'{position_type}_exit',
            'signal_price': signal_price,
            'fill_price': fill_price,
            'shares': shares,
            'value': trade_value,
            'commission': commission_cost,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'exit_reason': reason,
            'holding_days': (exit_date - position['entry_date']).days,
            'capital_after': self.capital
        }
        
        self.trades.append(trade)
        return trade
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy: Any,
                    verbose: bool = False) -> Dict:
        """
        Run complete backtest with realistic assumptions
        """
        self.reset()
        
        # Validate data
        if not self.validate_data(data):
            return {'error': 'Data validation failed', 'warnings': self.warnings}
        
        # Generate signals
        try:
            signals = strategy.generate_signals(data)
        except Exception as e:
            return {'error': f'Signal generation failed: {str(e)}'}
        
        # Track daily values
        for i in range(len(data)):
            date = data.index[i]
            row = data.iloc[i]
            
            # Skip warm-up period
            if i < strategy.config.lookback:
                self.equity_curve.append({
                    'date': date,
                    'equity': self.capital,
                    'capital': self.capital,
                    'positions_value': 0
                })
                continue
            
            # Get current signal
            signal = signals.iloc[i]
            
            # Update cash with daily interest
            daily_interest = self.capital * (self.interest_rate / 252)
            self.capital += daily_interest
            
            # Check existing positions for exit
            positions_to_exit = []
            for entry_date, position in self.positions.items():
                should_exit, exit_reason = strategy.check_exit_conditions(
                    position, row['Close'], signal['z_score'], date
                )
                if should_exit:
                    positions_to_exit.append((entry_date, exit_reason))
            
            # Execute exits
            for entry_date, reason in positions_to_exit:
                position = self.positions.pop(entry_date)
                self.exit_position(date, position, row['Close'], reason)
            
            # Check for new entry signals (if we have capacity)
            if len(self.positions) < self.max_positions:
                if signal['signal'] == 1:  # Long entry
                    shares = strategy.calculate_position_size(
                        self.capital,
                        row['Close'],
                        signal.get('volatility', 0.20),
                        signal['z_score'],
                        None  # half_life if available
                    )
                    if shares > 0:
                        self.enter_position(date, row['Close'], 'long', shares, 'z_score_signal')
                        
                elif signal['signal'] == -1:  # Short entry
                    shares = strategy.calculate_position_size(
                        self.capital,
                        row['Close'],
                        signal.get('volatility', 0.20),
                        signal['z_score'],
                        None
                    )
                    if shares > 0:
                        self.enter_position(date, row['Close'], 'short', shares, 'z_score_signal')
            
            # Calculate current equity
            positions_value = 0
            for position in self.positions.values():
                current_price = row['Close']
                if position['type'] == 'long':
                    positions_value += position['shares'] * current_price
                else:  # short
                    # Short position value = initial value - (current value - initial value)
                    initial = position['shares'] * position['entry_price']
                    current = position['shares'] * current_price
                    positions_value += initial - (current - initial)
            
            total_equity = self.capital + positions_value
            
            self.equity_curve.append({
                'date': date,
                'equity': total_equity,
                'capital': self.capital,
                'positions_value': positions_value,
                'n_positions': len(self.positions)
            })
            
            if verbose and i % 50 == 0:
                print(f"{date}: Equity=${total_equity:,.0f}, Positions={len(self.positions)}")
        
        # Calculate final metrics
        results = self.calculate_metrics()
        results['warnings'] = self.warnings
        
        return results
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.equity_curve:
            return {'error': 'No equity data available'}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_series = equity_df['equity']
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Basic performance
        total_return = (equity_series.iloc[-1] / self.initial_capital) - 1
        
        # Annualization factor
        n_days = len(returns)
        years = n_days / 252
        
        # Sharpe ratio (with sample size adjustment)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            # Deflate for small sample size
            deflation_factor = 1 - (3 / (4 * len(returns)))
            adjusted_sharpe = sharpe * deflation_factor
        else:
            sharpe = adjusted_sharpe = 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        completed_trades = [t for t in self.trades if 'net_pnl' in t]
        
        if completed_trades:
            wins = [t for t in completed_trades if t['net_pnl'] > 0]
            losses = [t for t in completed_trades if t['net_pnl'] < 0]
            
            win_rate = len(wins) / len(completed_trades)
            avg_win = np.mean([t['net_pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['net_pnl'] for t in losses]) if losses else 0
            
            profit_factor = abs(sum(t['net_pnl'] for t in wins) / 
                              sum(t['net_pnl'] for t in losses)) if losses else 0
            
            # Exit reason analysis
            exit_reasons = {}
            for trade in completed_trades:
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            exit_reasons = {}
        
        # Transaction cost analysis
        total_commission = sum(t.get('commission', 0) for t in self.trades)
        total_slippage = sum(abs(t.get('slippage', 0) * t.get('shares', 0)) for t in self.trades)
        
        # Break-even analysis
        min_profitable_move = self.total_transaction_cost * 2  # Round trip
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': equity_series.iloc[-1],
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (1/years) - 1 if years > 0 else 0,
            'sharpe_ratio': sharpe,
            'adjusted_sharpe': adjusted_sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'completed_trades': len(completed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'exit_reasons': exit_reasons,
            'total_commission_paid': total_commission,
            'total_slippage_cost': total_slippage,
            'min_profitable_move': min_profitable_move,
            'equity_curve': equity_df,
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        }
