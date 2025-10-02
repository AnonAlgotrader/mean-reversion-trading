# Mean Reversion Trading Strategies - Complete Python Implementation

A comprehensive, production-ready implementation of mean reversion trading strategies with emphasis on robustness, risk management, and avoiding common pitfalls.

## ⚠️ Critical Disclaimer

**This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. The code emphasizes what can go wrong rather than potential profits. Never trade with money you cannot afford to lose.**

## Overview

This repository contains the complete implementation of concepts from "Mean Reversion Trading Strategies with Python: A Statistical Approach". The code prioritizes:

- **Robustness over performance** - Conservative parameters throughout
- **Risk management** - Multiple safety checks and position limits
- **Realistic assumptions** - Transaction costs, slippage, and market impact
- **Pitfall detection** - Identifies overfitting, regime changes, and cost issues

## Key Features

### Statistical Validation
- Augmented Dickey-Fuller test for stationarity
- Hurst exponent calculation
- Half-life analysis for mean reversion speed
- Cointegration testing for pairs trading

### Strategy Implementations
- Z-score mean reversion with dynamic thresholds
- RSI-based mean reversion
- Bollinger Bands strategy
- Pairs trading with dynamic hedge ratios

### Risk Management
- Kelly Criterion with safety factors (25% fractional Kelly)
- Volatility-based position sizing
- Regime detection to avoid trending markets
- Maximum drawdown controls

### Realistic Backtesting
- Transaction costs (commission, spread, slippage, market impact)
- Realistic fill simulation
- Sample size validation
- Sharpe ratio deflation for small samples

### Overfitting Prevention
- Walk-forward analysis
- Parameter sensitivity testing
- Monte Carlo simulation
- Minimum sample size requirements

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mean-reversion-trading.git
cd mean-reversion-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.statistical_tests import MeanReversionTests
from src.strategies.zscore_strategy import ZScoreMeanReversion, SignalConfig
from src.backtesting.engine import BacktestEngine
import yfinance as yf

# Download data
data = yf.download('SPY', start='2020-01-01', end='2024-01-01')

# Test for mean reversion
tests = MeanReversionTests()
results = tests.run_all_tests(data['Close'])

if results['summary']['mean_reversion_score'] == '0/3':
    print("WARNING: No mean reversion detected. Do not trade this asset.")
else:
    # Configure strategy
    config = SignalConfig(
        lookback=20,
        entry_z_threshold=2.0,
        stop_loss_pct=0.05  # 5% stop loss
    )
    strategy = ZScoreMeanReversion(config)
    
    # Run backtest with realistic costs
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,      # 0.1% commission
        spread=0.0005,         # 0.05% spread
        slippage=0.0005,       # 0.05% slippage
        market_impact=0.0002   # 0.02% market impact
    )
    
    results = engine.run_backtest(data, strategy)
    
    print(f"Sharpe Ratio: {results['adjusted_sharpe']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Break-even move: {results['min_profitable_move']:.3%}")
```

## Complete Pipeline Example

Run the full pipeline with all safety checks:

```bash
python examples/complete_pipeline_example.py
```

This will:
1. Download and validate data
2. Run statistical tests for mean reversion
3. Check current market regime
4. Configure strategy based on half-life
5. Run realistic backtest with all costs
6. Test for overfitting
7. Generate performance visualizations
8. Provide go/no-go recommendation

## Repository Structure

```
├── src/
│   ├── statistical_tests.py           # ADF, Hurst, Half-life tests
│   ├── strategies/
│   │   ├── zscore_strategy.py        # Complete Z-score implementation
│   │   ├── rsi_strategy.py           # RSI mean reversion
│   │   └── pairs_trading.py          # Pairs trading with cointegration
    |   └── bollinger_strategy.py     # Complete Bollinger Bands Strategy
│   ├── risk_management/
│   │   ├── position_sizing.py        # Kelly, volatility sizing
│   │   ├── regime_detection.py       # Market regime identification
│   │   └── risk_manager.py           # Portfolio risk controls
│   ├── backtesting/
│   │   ├── backtesting_engine.py     # Realistic backtesting engine (contain execution, transaction costs, and performance metrics)
│   └── validation/
│       ├── overfitting_detection.py  # Walk-forward, sensitivity 
├── examples/
│   └── complete_pipeline_example.py  # Full implementation example
├── tests/
│   └── test_statistical_tests.py     # Unit tests
└── requirements.txt
```

## Critical Warnings

### Transaction Costs
- Minimum profitable move with all costs: ~0.44% round-trip
- Many "profitable" strategies fail after realistic costs
- Always include commission, spread, slippage, AND market impact

### Overfitting
- Require minimum 100 trades for validation
- Use walk-forward analysis on out-of-sample data
- If Sharpe drops >30% out-of-sample, strategy is overfit

### Regime Changes
- Mean reversion fails catastrophically during trends
- Always check regime before trading
- Stop trading immediately when trend detected

### Position Sizing
- Never use full Kelly - use 25% maximum
- Never risk more than 2% per trade
- Account for correlation between positions

## Expected Performance (Realistic)

After all costs and realistic assumptions:
- **Annual Return**: 8-15% (if everything goes well)
- **Sharpe Ratio**: 0.8-1.2
- **Max Drawdown**: 15-25%
- **Win Rate**: 55-65%
- **Break-even per trade**: 0.44%

If backtests show better results, they're probably wrong.

## Common Failure Modes

1. **Ignoring costs**: Strategy profitable gross, loses money net
2. **Overfitting**: Great backtest, immediate failure live
3. **Wrong regime**: Mean reversion during trend = large losses
4. **Excessive leverage**: One bad trade wipes out account
5. **Insufficient data**: Drawing conclusions from <100 trades

## Testing

Run comprehensive tests:

```bash
pytest tests/ -v --cov=src
```

## Production Deployment Checklist

Before trading real money, ensure:

- [ ] Statistical tests confirm mean reversion (3/3 pass)
- [ ] Current regime favorable (not trending)
- [ ] Backtest includes ALL transaction costs
- [ ] Out-of-sample Sharpe within 30% of in-sample
- [ ] Minimum 100 trades in backtest
- [ ] Maximum drawdown acceptable (<20%)
- [ ] Break-even move achievable
- [ ] Paper traded successfully for 30+ days
- [ ] Risk per trade ≤ 2% of capital
- [ ] Stop losses implemented and tested

## Support and Contributing

- **Issues**: Use GitHub issues for bugs/questions
- **Contributing**: Read CONTRIBUTING.md first
- **License**: MIT (see LICENSE file)

## References

Based on the article: "Mean Reversion Trading Strategies with Python: A Statistical Approach"

Key concepts:
- Ornstein-Uhlenbeck process for mean reversion
- Augmented Dickey-Fuller test for stationarity  
- Kelly Criterion with safety factors
- Transaction cost impact on high-frequency strategies

## Final Warning

If a strategy looks too good in backtesting, it probably is. The market has a way of humbling overconfident traders. Start small, monitor carefully, and always prioritize capital preservation over profits.

**Remember: Most mean reversion strategies that work in backtesting fail in live trading due to ignored implementation details.**
