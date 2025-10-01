#!/usr/bin/env python3
"""
Complete mean reversion trading pipeline example
Demonstrates all concepts from the article with realistic assumptions
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import all modules (adjust path as needed)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statistical_tests import MeanReversionTests
from src.strategies.zscore_strategy import ZScoreMeanReversion, SignalConfig
from src.risk_management.regime_detection import RegimeDetector
from src.risk_management.position_sizing import PositionSizer
from src.backtesting.engine import BacktestEngine

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def download_and_validate_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download and validate historical data"""
    print_section("1. DATA DOWNLOAD AND VALIDATION")
    
    print(f"Downloading {symbol} from {start_date} to {end_date}...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        raise ValueError(f"No data available for {symbol}")
    
    # Data quality checks
    print(f"✓ Downloaded {len(data)} days of data")
    
    # Check for missing values
    missing = data.isnull().sum().sum()
    if missing > 0:
        print(f"⚠ Warning: {missing} missing values found - forward filling")
        data = data.fillna(method='ffill')
    
    # Check for data consistency
    invalid_bars = (
        (data['High'] < data['Low']) |
        (data['High'] < data['Close']) |
        (data['Low'] > data['Close'])
    ).sum()
    
    if invalid_bars > 0:
        print(f"⚠ Warning: {invalid_bars} invalid OHLC bars detected")
    
    return data

def run_statistical_tests(prices: pd.Series) -> Dict:
    """Run comprehensive statistical tests"""
    print_section("2. STATISTICAL VALIDATION")
    
    tests = MeanReversionTests()
    results = tests.run_all_tests(prices)
    
    # Display results
    print("\nADF Test:")
    print(f"  P-value: {results['adf_test']['p_value']:.4f}")
    print(f"  Stationary: {results['adf_test']['is_stationary']}")
    
    print("\nHurst Exponent:")
    print(f"  Value: {results['hurst_exponent']['hurst_exponent']:.4f}")
    print(f"  Behavior: {results['hurst_exponent']['behavior']}")
    
    print("\nHalf-Life:")
    if results['half_life']['half_life']:
        print(f"  Value: {results['half_life']['half_life']:.1f} days")
        print(f"  Interpretation: {results['half_life']['interpretation']}")
    else:
        print(f"  Not detected (beta = {results['half_life']['beta']:.4f})")
    
    print("\n" + "-" * 40)
    print(f"Overall Score: {results['summary']['mean_reversion_score']}")
    print(f"Conclusion: {results['summary']['conclusion']}")
    print(f"Recommendation: {results['summary']['recommendation']}")
    
    return results

def check_regime(prices: pd.Series) -> Dict:
    """Check current market regime"""
    print_section("3. REGIME DETECTION")
    
    detector = RegimeDetector(lookback=60)
    regime = detector.detect_regime(prices)
    
    print(f"\nCurrent Regime: {regime['current_regime'].upper()}")
    print(f"Is Favorable: {'✓ Yes' if regime['is_favorable'] else '✗ No'}")
    print(f"Should Trade: {'✓ Yes' if regime['should_trade'] else '✗ No'}")
    
    # Display component regimes
    print("\nRegime Components:")
    if 'hurst_regime' in regime:
        print(f"  Hurst: {regime['hurst_regime'].get('regime', 'N/A')}")
    if 'volatility_regime' in regime:
        print(f"  Volatility: {regime['volatility_regime'].get('regime', 'N/A')}")
    if 'trend_regime' in regime:
        print(f"  Trend: {regime['trend_regime'].get('regime', 'N/A')}")
    
    print(f"\nRecommendation: {regime['recommendation']}")
    
    return regime

def configure_strategy(test_results: Dict) -> ZScoreMeanReversion:
    """Configure strategy based on statistical tests"""
    print_section("4. STRATEGY CONFIGURATION")
    
    # Use half-life for parameters if available
    half_life = test_results['half_life'].get('half_life')
    
    if half_life and half_life > 0:
        lookback = min(int(half_life * 1.5), 60)  # 1.5x half-life, max 60
        max_holding = int(half_life * 3)  # 3x half-life
        print(f"Using half-life based parameters:")
    else:
        lookback = 20
        max_holding = 10
        print(f"Using default parameters (no half-life detected):")
    
    print(f"  Lookback period: {lookback} days")
    print(f"  Max holding period: {max_holding} days")
    
    config = SignalConfig(
        lookback=lookback,
        entry_z_threshold=2.0,
        exit_z_threshold=0.5,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        max_holding_days=max_holding,
        volatility_filter=True,
        volume_filter=True,
        regime_filter=True
    )
    
    strategy = ZScoreMeanReversion(config)
    
    print("\nRisk Parameters:")
    print(f"  Entry Z-score: ±{config.entry_z_threshold}")
    print(f"  Exit Z-score: ±{config.exit_z_threshold}")
    print(f"  Stop Loss: {config.stop_loss_pct:.1%}")
    print(f"  Take Profit: {config.take_profit_pct:.1%}")
    print(f"  Filters: Volatility={'✓' if config.volatility_filter else '✗'}, "
          f"Volume={'✓' if config.volume_filter else '✗'}, "
          f"Regime={'✓' if config.regime_filter else '✗'}")
    
    return strategy

def run_backtest(data: pd.DataFrame, strategy: ZScoreMeanReversion) -> Dict:
    """Run realistic backtest with transaction costs"""
    print_section("5. BACKTESTING WITH REALISTIC COSTS")
    
    print("Transaction Cost Assumptions:")
    print("  Commission: 0.1%")
    print("  Spread: 0.05%")
    print("  Slippage: 0.05%")
    print("  Market Impact: 0.02%")
    print("  Total Round-Trip Cost: ~0.44%")
    
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,
        spread=0.0005,
        slippage=0.0005,
        market_impact=0.0002
    )
    
    results = engine.run_backtest(data, strategy, verbose=False)
    
    # Display results
    print("\n" + "-" * 40)
    print("BACKTEST RESULTS:")
    print(f"  Initial Capital: ${results['initial_capital']:,.0f}")
    print(f"  Final Equity: ${results['final_equity']:,.0f}")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Annualized Return: {results['annualized_return']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Adjusted Sharpe: {results['adjusted_sharpe']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Completed Trades: {results['completed_trades']}")
    print(f"  Win Rate: {results['win_rate']:.1%}")
    print(f"  Avg Win: {results['avg_win']:.2%}")
    print(f"  Avg Loss: {results['avg_loss']:.2%}")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    
    print(f"\nCost Analysis:")
    print(f"  Total Commission: ${results['total_commission_paid']:.2f}")
    print(f"  Total Slippage: ${results['total_slippage_cost']:.2f}")
    print(f"  Min Profitable Move: {results['min_profitable_move']:.3%}")
    
    if results['exit_reasons']:
        print(f"\nExit Reasons:")
        for reason, count in results['exit_reasons'].items():
            print(f"  {reason}: {count} trades")
    
    if results.get('warnings'):
        print(f"\nWarnings:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    
    return results

def check_for_overfitting(strategy: ZScoreMeanReversion, data: pd.DataFrame):
    """Check for signs of overfitting"""
    print_section("6. OVERFITTING DETECTION")
    
    print("Testing parameter sensitivity...")
    
    # Test different parameter values
    lookback_values = [15, 20, 25, 30]
    threshold_values = [1.5, 2.0, 2.5, 3.0]
    
    results_matrix = []
    
    for lb in lookback_values:
        for thresh in threshold_values:
            # Create test strategy
            test_config = SignalConfig(
                lookback=lb,
                entry_z_threshold=thresh,
                exit_z_threshold=0.5
            )
            test_strategy = ZScoreMeanReversion(test_config)
            
            # Run quick backtest
            test_engine = BacktestEngine(initial_capital=10000)
            test_results = test_engine.run_backtest(data, test_strategy)
            
            results_matrix.append({
                'lookback': lb,
                'threshold': thresh,
                'sharpe': test_results['sharpe_ratio'],
                'return': test_results['total_return']
            })
    
    # Analyze sensitivity
    sharpe_values = [r['sharpe'] for r in results_matrix]
    return_values = [r['return'] for r in results_matrix]
    
    sharpe_std = np.std(sharpe_values)
    return_std = np.std(return_values)
    
    print(f"\nParameter Sensitivity Analysis:")
    print(f"  Sharpe Std Dev: {sharpe_std:.3f}")
    print(f"  Return Std Dev: {return_std:.3%}")
    
    # Determine if overfit
    if sharpe_std > 0.5:
        print("  ⚠ HIGH SENSITIVITY - Strategy may be overfit")
        print("  Recommendation: Use more conservative parameters")
    elif sharpe_std > 0.3:
        print("  ⚠ MODERATE SENSITIVITY - Some overfitting risk")
        print("  Recommendation: Monitor out-of-sample performance carefully")
    else:
        print("  ✓ LOW SENSITIVITY - Parameters appear robust")
        print("  Recommendation: Strategy shows good parameter stability")
    
    # Show parameter performance grid
    print("\nParameter Performance Grid (Sharpe Ratio):")
    print("         ", end="")
    for thresh in threshold_values:
        print(f"Z={thresh:.1f}  ", end="")
    print()
    
    for lb in lookback_values:
        print(f"  LB={lb:2d}: ", end="")
        for thresh in threshold_values:
            sharpe = next(r['sharpe'] for r in results_matrix 
                         if r['lookback'] == lb and r['threshold'] == thresh)
            print(f"{sharpe:5.2f} ", end="")
        print()

def create_visualizations(results: Dict):
    """Create performance visualizations"""
    print_section("7. GENERATING VISUALIZATIONS")
    
    if 'equity_curve' not in results or results['equity_curve'].empty:
        print("No equity curve data available for visualization")
        return
    
    equity_df = results['equity_curve']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Equity Curve
    axes[0].plot(equity_df['date'], equity_df['equity'], 
                 label='Portfolio Value', color='blue', linewidth=2)
    axes[0].axhline(y=10000, color='gray', linestyle='--', 
                    label='Initial Capital', alpha=0.5)
    axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Drawdown
    returns = equity_df['equity'].pct_change().fillna(0)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = ((cumulative - running_max) / running_max) * 100
    
    axes[1].fill_between(equity_df['date'], drawdown, 0, 
                         color='red', alpha=0.3, label='Drawdown')
    axes[1].set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Position Count
    if 'n_positions' in equity_df.columns:
        axes[2].plot(equity_df['date'], equity_df['n_positions'], 
                    label='Open Positions', color='green', linewidth=1)
        axes[2].fill_between(equity_df['date'], 0, equity_df['n_positions'],
                            color='green', alpha=0.2)
        axes[2].set_title('Position Count Over Time', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Number of Positions')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'mean_reversion_backtest_results.png'
    plt.savefig(filename, dpi=100)
    print(f"✓ Saved visualization to {filename}")
    
    # Show plot
    plt.show()

def generate_final_report(test_results: Dict, regime: Dict, backtest_results: Dict):
    """Generate final trading recommendation"""
    print_section("8. FINAL ASSESSMENT AND RECOMMENDATIONS")
    
    # Calculate overall score
    score = 0
    max_score = 5
    
    # Statistical tests (0-2 points)
    mr_score = test_results['summary']['mean_reversion_score']
    if '3/3' in mr_score:
        score += 2
        stat_assessment = "Strong mean reversion"
    elif '2/3' in mr_score:
        score += 1
        stat_assessment = "Moderate mean reversion"
    else:
        stat_assessment = "Weak/No mean reversion"
    
    # Regime favorability (0-1 point)
    if regime['is_favorable']:
        score += 1
        regime_assessment = "Favorable"
    else:
        regime_assessment = "Unfavorable"
    
    # Backtest performance (0-2 points)
    sharpe = backtest_results['adjusted_sharpe']
    drawdown = backtest_results['max_drawdown']
    
    if sharpe > 1.0 and drawdown > -0.20:
        score += 2
        perf_assessment = "Good risk-adjusted returns"
    elif sharpe > 0.5 and drawdown > -0.30:
        score += 1
        perf_assessment = "Moderate risk-adjusted returns"
    else:
        perf_assessment = "Poor risk-adjusted returns"
    
    # Generate recommendation
    print(f"Assessment Summary:")
    print(f"  Statistical Tests: {stat_assessment}")
    print(f"  Market Regime: {regime_assessment}")
    print(f"  Backtest Performance: {perf_assessment}")
    print(f"  Overall Score: {score}/{max_score}")
    
    print("\n" + "=" * 40)
    
    if score >= 4:
        print("✓ RECOMMENDATION: PROCEED TO PAPER TRADING")
        print("\nNext Steps:")
        print("1. Run strategy on paper trading account for 30 days")
        print("2. Monitor daily performance and regime changes")
        print("3. Compare paper results to backtest expectations")
        print("4. If paper trading successful, start with minimum capital")
    elif score >= 2:
        print("⚠ RECOMMENDATION: REQUIRES IMPROVEMENT")
        print("\nIssues to Address:")
        if '0/3' in mr_score or '1/3' in mr_score:
            print("- Asset does not show sufficient mean reversion")
        if not regime['is_favorable']:
            print("- Current market regime unsuitable for mean reversion")
        if sharpe < 0.5:
            print("- Risk-adjusted returns too low")
        if drawdown < -0.30:
            print("- Excessive drawdown risk")
        print("\nSuggestions:")
        print("- Try different assets or timeframes")
        print("- Adjust strategy parameters")
        print("- Wait for more favorable market conditions")
    else:
        print("✗ RECOMMENDATION: DO NOT TRADE")
        print("\nStrategy shows insufficient edge for profitable trading.")
        print("Consider alternative approaches or asset classes.")
    
    print("\n" + "=" * 40)
    print("\nRISK DISCLAIMER:")
    print("Past performance does not guarantee future results.")
    print("All trading involves risk of loss.")
    print("Never trade with money you cannot afford to lose.")

def main():
    """
    Execute complete mean reversion trading pipeline
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " MEAN REVERSION TRADING STRATEGY PIPELINE ".center(78) + "*")
    print("*" + " Production-Ready Implementation with All Safety Checks ".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    # Configuration
    SYMBOL = 'SPY'
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    
    try:
        # Step 1: Download and validate data
        data = download_and_validate_data(SYMBOL, START_DATE, END_DATE)
        
        # Step 2: Run statistical tests
        test_results = run_statistical_tests(data['Close'])
        
        # Early exit if no mean reversion
        if test_results['summary']['mean_reversion_score'] == '0/3':
            print("\n" + "!" * 60)
            print("! STOPPING: No evidence of mean reversion detected !")
            print("!" * 60)
            print("\nThis asset is not suitable for mean reversion strategies.")
            print("Consider trend-following or other approaches.")
            return
        
        # Step 3: Check market regime
        regime = check_regime(data['Close'])
        
        # Warning if regime unfavorable
        if not regime['is_favorable']:
            print("\n" + "!" * 60)
            print("! WARNING: Current regime unfavorable for mean reversion !")
            print("!" * 60)
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting. Wait for favorable conditions.")
                return
        
        # Step 4: Configure strategy
        strategy = configure_strategy(test_results)
        
        # Step 5: Run backtest
        backtest_results = run_backtest(data, strategy)
        
        # Step 6: Check for overfitting
        check_for_overfitting(strategy, data)
        
        # Step 7: Generate visualizations
        create_visualizations(backtest_results)
        
        # Step 8: Generate final report
        generate_final_report(test_results, regime, backtest_results)
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print("Pipeline failed. Check data and parameters.")
        raise
    
    print("\n" + "*" * 80)
    print("Pipeline completed successfully.")
    print("Review all results carefully before proceeding to live trading.")
    print("*" * 80 + "\n")

if __name__ == "__main__":
    main()
