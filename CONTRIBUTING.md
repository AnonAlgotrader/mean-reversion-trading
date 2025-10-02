# Contributing to Mean Reversion Trading Framework

## Statistical Validation Requirements

All contributions must adhere to the statistical principles outlined in "Mean Reversion Trading Strategies with Python":

### Required Statistical Tests
- **Augmented Dickey-Fuller Test** for stationarity (p < 0.05)
- **Hurst Exponent** calculation (H < 0.5 indicates mean reversion)
- **Half-life** calculation for reversion speed using Ornstein-Uhlenbeck process

### Minimum Sample Requirements
- **100+ trades** for statistical significance
- **3+ years of daily data** (756 periods)
- **Walk-forward validation** with consistent out-of-sample performance

### Transaction Cost Modeling
All strategies must include realistic transaction costs:
- **Commission**: 0.1% per trade
- **Bid-ask spread**: 0.05%
- **Market impact**: 0.02%
- **Minimum profitable move** must exceed total round-trip costs (0.3-0.5%)

### Performance Validation
- **Sharpe ratio** with small-sample deflation: `1 - (3/(4n))`
- **Statistical significance**: p-value < 0.05 for returns different from zero
- **Profit factor** and **win rate** analysis
- **Maximum drawdown** and **recovery factor**

### Risk Management Requirements
- **Position sizing** using fractional Kelly criterion (25% of full Kelly)
- **Regime detection** to disable trading during trending markets (Hurst > 0.55)
- **Maximum position limits** and **correlation controls**

## Strategy Development Standards

### Backtesting Requirements
- **Realistic fill prices** with slippage and market impact
- **Transaction costs** included in all performance calculations
- **Walk-forward analysis** to detect overfitting
- **Parameter sensitivity** testing to ensure robustness

### Statistical Significance
- Minimum required trades: `(z_score / expected_sharpe)² × 2`
- Returns must be statistically different from zero (p < 0.05)
- Consistent performance across market regimes

### Pitfall Prevention
All strategies must address:
- **Overfitting** through walk-forward validation
- **Regime changes** with adaptive detection
- **Transaction cost erosion** with break-even analysis
- **Insufficient sample sizes** with minimum trade requirements

## Pre-Deployment Checklist

Before submission, strategies must pass:
- [ ] Stationarity validation (ADF test)
- [ ] Mean reversion confirmation (Hurst < 0.5)
- [ ] Sufficient sample size (100+ trades, 3+ years)
- [ ] Positive returns after transaction costs
- [ ] Robustness to parameter changes
- [ ] Consistent walk-forward performance

## Contribution Process

1. **Statistical validation** of proposed strategy
2. **Backtesting** with realistic transaction costs
3. **Walk-forward analysis** to confirm robustness
4. **Documentation** of statistical foundations
5. **Pull request** with complete validation results

**Note**: Strategies that show overfitting, ignore transaction costs, or lack statistical foundations will not be accepted. The focus is on robust, statistically sound approaches rather than over-optimized backtests.
