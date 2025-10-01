"""
Portfolio risk management module
Comprehensive risk controls and monitoring
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings

@dataclass
class RiskLimits:
    """Risk management limits and thresholds"""
    max_positions: int = 5
    max_correlation: float = 0.7
    max_position_size: float = 0.10  # 10% per position
    max_sector_exposure: float = 0.30  # 30% per sector
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_weekly_loss: float = 0.05  # 5% weekly loss limit
    max_drawdown: float = 0.10  # 10% maximum drawdown
    var_confidence: float = 0.95  # 95% VaR
    max_leverage: float = 1.0  # No leverage by default
    min_cash_reserve: float = 0.20  # Keep 20% in cash

class RiskManager:
    """
    Comprehensive risk management system
    """
    
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.positions = {}
        self.daily_pnl = []
        self.weekly_pnl = []
        self.peak_equity = 0
        self.current_equity = 0
        self.risk_metrics = {}
        self.violations = []
        
    def check_pre_trade_risk(self, new_position: Dict, portfolio: Dict) -> Dict:
        """
        Check risk before entering a new position
        
        Returns:
        --------
        dict : Approval status and any violations
        """
        violations = []
        warnings = []
        
        # 1. Position count check
        current_positions = len(portfolio.get('positions', {}))
        if current_positions >= self.limits.max_positions:
            violations.append(f"Maximum positions reached ({self.limits.max_positions})")
        
        # 2. Position size check
        position_size_pct = new_position['value'] / portfolio['total_equity']
        if position_size_pct > self.limits.max_position_size:
            violations.append(f"Position too large: {position_size_pct:.1%} > {self.limits.max_position_size:.1%}")
        
        # 3. Correlation check
        if 'returns' in new_position and 'returns' in portfolio:
            correlation = self._check_correlation(new_position['returns'], 
                                                 portfolio['returns'])
            if correlation > self.limits.max_correlation:
                warnings.append(f"High correlation with portfolio: {correlation:.2f}")
        
        # 4. Sector concentration check
        if 'sector' in new_position:
            sector_exposure = self._calculate_sector_exposure(new_position['sector'], 
                                                             portfolio)
            if sector_exposure > self.limits.max_sector_exposure:
                violations.append(f"Sector concentration too high: {sector_exposure:.1%}")
        
        # 5. Cash reserve check
        cash_after_trade = (portfolio['cash'] - new_position['value'])
        cash_ratio = cash_after_trade / portfolio['total_equity']
        if cash_ratio < self.limits.min_cash_reserve:
            warnings.append(f"Low cash reserve after trade: {cash_ratio:.1%}")
        
        # 6. Leverage check
        total_exposure = sum(p['value'] for p in portfolio.get('positions', {}).values())
        total_exposure += new_position['value']
        leverage = total_exposure / portfolio['total_equity']
        if leverage > self.limits.max_leverage:
            violations.append(f"Leverage exceeded: {leverage:.2f}x > {self.limits.max_leverage:.1f}x")
        
        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'metrics': {
                'position_size_pct': position_size_pct,
                'correlation': correlation if 'returns' in new_position else None,
                'leverage_after': leverage,
                'cash_ratio_after': cash_ratio
            }
        }
    
    def check_portfolio_risk(self, portfolio: Dict) -> Dict:
        """
        Comprehensive portfolio risk assessment
        """
        metrics = {}
        warnings = []
        
        # 1. Calculate current drawdown
        current_equity = portfolio.get('total_equity', 0)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown = (current_equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        metrics['drawdown'] = drawdown
        
        if drawdown < -self.limits.max_drawdown:
            warnings.append(f"Maximum drawdown exceeded: {drawdown:.1%}")
        
        # 2. Calculate Value at Risk (VaR)
        if 'returns_history' in portfolio and len(portfolio['returns_history']) > 20:
            var = self._calculate_var(portfolio['returns_history'], 
                                     self.limits.var_confidence)
            metrics['var_95'] = var
            
            if var < -self.limits.max_daily_loss:
                warnings.append(f"VaR exceeds daily loss limit: {var:.1%}")
        
        # 3. Calculate portfolio volatility
        if 'returns_history' in portfolio:
            returns = pd.Series(portfolio['returns_history'])
            volatility = returns.std() * np.sqrt(252)
            metrics['annual_volatility'] = volatility
            
            if volatility > 0.30:  # 30% annual volatility
                warnings.append(f"High portfolio volatility: {volatility:.1%}")
        
        # 4. Check correlation matrix
        if 'position_returns' in portfolio:
            corr_matrix = pd.DataFrame(portfolio['position_returns']).corr()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            metrics['avg_correlation'] = avg_correlation
            
            if avg_correlation > 0.6:
                warnings.append(f"High average correlation: {avg_correlation:.2f}")
        
        # 5. Concentration risk
        if 'positions' in portfolio:
            positions = portfolio['positions']
            total_value = sum(p['value'] for p in positions.values())
            
            if total_value > 0:
                concentrations = {k: v['value']/total_value for k, v in positions.items()}
                max_concentration = max(concentrations.values()) if concentrations else 0
                metrics['max_concentration'] = max_concentration
                
                if max_concentration > 0.30:
                    warnings.append(f"Position concentration risk: {max_concentration:.1%}")
        
        return {
            'metrics': metrics,
            'warnings': warnings,
            'risk_score': self._calculate_risk_score(metrics),
            'recommendation': self._get_risk_recommendation(metrics, warnings)
        }
    
    def update_daily_pnl(self, pnl: float, equity: float) -> str:
        """
        Update daily P&L and check loss limits
        """
        self.daily_pnl.append(pnl)
        pnl_pct = pnl / equity if equity > 0 else 0
        
        # Check daily loss limit
        if pnl_pct < -self.limits.max_daily_loss:
            self.violations.append(f"Daily loss limit breached: {pnl_pct:.2%}")
            return "STOP_TRADING_TODAY"
        
        # Check consecutive losses
        if len(self.daily_pnl) >= 3:
            last_three = self.daily_pnl[-3:]
            if all(p < 0 for p in last_three):
                return "REDUCE_POSITION_SIZES"
        
        # Check weekly loss (if we have enough data)
        if len(self.daily_pnl) >= 5:
            weekly_pnl = sum(self.daily_pnl[-5:])
            weekly_pnl_pct = weekly_pnl / equity if equity > 0 else 0
            
            if weekly_pnl_pct < -self.limits.max_weekly_loss:
                self.violations.append(f"Weekly loss limit breached: {weekly_pnl_pct:.2%}")
                return "STOP_TRADING_THIS_WEEK"
        
        return "CONTINUE"
    
    def _check_correlation(self, new_returns: pd.Series, 
                          portfolio_returns: pd.DataFrame) -> float:
        """Calculate correlation between new position and portfolio"""
        if portfolio_returns.empty:
            return 0
        
        correlations = portfolio_returns.corrwith(new_returns)
        return correlations.abs().max()
    
    def _calculate_sector_exposure(self, sector: str, portfolio: Dict) -> float:
        """Calculate exposure to a specific sector"""
        if 'positions' not in portfolio:
            return 0
        
        sector_value = sum(p['value'] for p in portfolio['positions'].values() 
                          if p.get('sector') == sector)
        
        total_value = portfolio['total_equity']
        return sector_value / total_value if total_value > 0 else 0
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """
        Calculate Value at Risk
        """
        if not returns:
            return 0
        
        returns_sorted = sorted(returns)
        index = int((1 - confidence) * len(returns_sorted))
        return returns_sorted[index] if index < len(returns_sorted) else returns_sorted[0]
    
    def _calculate_risk_score(self, metrics: Dict) -> float:
        """
        Calculate overall risk score (0-100, higher is riskier)
        """
        score = 0
        
        # Drawdown contribution (0-30 points)
        if 'drawdown' in metrics:
            drawdown_score = min(abs(metrics['drawdown']) * 300, 30)
            score += drawdown_score
        
        # Volatility contribution (0-25 points)
        if 'annual_volatility' in metrics:
            vol_score = min(metrics['annual_volatility'] * 83, 25)
            score += vol_score
        
        # Correlation contribution (0-20 points)
        if 'avg_correlation' in metrics:
            corr_score = metrics['avg_correlation'] * 20
            score += corr_score
        
        # Concentration contribution (0-25 points)
        if 'max_concentration' in metrics:
            conc_score = metrics['max_concentration'] * 83
            score += conc_score
        
        return min(score, 100)
    
    def _get_risk_recommendation(self, metrics: Dict, warnings: List[str]) -> str:
        """
        Generate risk-based recommendation
        """
        risk_score = self._calculate_risk_score(metrics)
        
        if risk_score < 30:
            base_rec = "Low risk - maintain current strategy"
        elif risk_score < 50:
            base_rec = "Moderate risk - monitor positions closely"
        elif risk_score < 70:
            base_rec = "Elevated risk - consider reducing positions"
        else:
            base_rec = "High risk - reduce exposure immediately"
        
        if warnings:
            base_rec += f". Warnings: {', '.join(warnings[:2])}"
        
        return base_rec
    
    def get_position_adjustment_recommendations(self, portfolio: Dict) -> Dict:
        """
        Generate specific position adjustment recommendations
        """
        recommendations = []
        
        # Check each position
        if 'positions' in portfolio:
            for symbol, position in portfolio['positions'].items():
                # Check position age
                if 'days_held' in position and position['days_held'] > 30:
                    recommendations.append(f"Consider closing {symbol} - held too long")
                
                # Check position P&L
                if 'unrealized_pnl_pct' in position:
                    if position['unrealized_pnl_pct'] < -0.10:
                        recommendations.append(f"Cut losses on {symbol} - down {position['unrealized_pnl_pct']:.1%}")
                    elif position['unrealized_pnl_pct'] > 0.20:
                        recommendations.append(f"Consider taking profits on {symbol} - up {position['unrealized_pnl_pct']:.1%}")
        
        # Portfolio level recommendations
        portfolio_risk = self.check_portfolio_risk(portfolio)
        
        if portfolio_risk['metrics'].get('drawdown', 0) < -0.08:
            recommendations.append("Reduce all positions by 50% - approaching max drawdown")
        
        if portfolio_risk['metrics'].get('avg_correlation', 0) > 0.7:
            recommendations.append("Diversify - positions too correlated")
        
        return {
            'recommendations': recommendations,
            'risk_score': portfolio_risk['risk_score'],
            'action_required': len(recommendations) > 0
        }
