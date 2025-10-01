### src/validation/overfitting_detection.py
```python
"""
Overfitting detection and prevention tools
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Callable
from sklearn.model_selection import TimeSeriesSplit

class OverfittingDetector:
    """
    Detect overfitting in trading strategies
    """
    
    @staticmethod
    def walk_forward_analysis(strategy: Callable,
                             data: pd.DataFrame,
                             n_splits: int = 5,
                             train_ratio: float = 0.6) -> Dict:
        """
        Perform walk-forward analysis to detect overfitting
        
        Parameters:
        -----------
        strategy : Callable
            Strategy class with fit and backtest methods
        data : pd.DataFrame
            Historical data
        n_splits : int
            Number of forward walks
        train_ratio : float
            Ratio of data for training (0.6 = 60% train, 40% test)
            
        Returns:
        --------
        dict : Analysis results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        in_sample_results = []
        out_sample_results = []
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Optimize on training data
            from src.backtesting.engine import BacktestEngine
            
            # In-sample performance
            engine_train = BacktestEngine()
            train_results = engine_train.run_backtest(train_data, strategy)
            in_sample_results.append(train_results['sharpe_ratio'])
            
            # Out-of-sample performance
            engine_test = BacktestEngine()
            test_results = engine_test.run_backtest(test_data, strategy)
            out_sample_results.append(test_results['sharpe_ratio'])
        
        # Calculate degradation
        avg_in_sample = np.mean(in_sample_results)
        avg_out_sample = np.mean(out_sample_results)
        degradation = 1 - (avg_out_sample / avg_in_sample) if avg_in_sample != 0 else 1
        
        # Determine if overfit
        is_overfit = degradation > 0.3  # 30% degradation threshold
        
        return {
            'in_sample_sharpe': in_sample_results,
            'out_sample_sharpe': out_sample_results,
            'avg_in_sample': avg_in_sample,
            'avg_out_sample': avg_out_sample,
            'performance_degradation': degradation,
            'is_overfit': is_overfit,
            'recommendation': 'Strategy appears overfit' if is_overfit else 'Strategy appears robust'
        }
    
    @staticmethod
    def parameter_sensitivity_analysis(strategy_class,
                                      data: pd.DataFrame,
                                      base_params: Dict,
                                      param_ranges: Dict) -> Dict:
        """
        Test strategy sensitivity to parameter changes
        
        Parameters:
        -----------
        strategy_class : class
            Strategy class to test
        data : pd.DataFrame
            Historical data
        base_params : dict
            Base parameters for strategy
        param_ranges : dict
            Ranges to test for each parameter
            
        Returns:
        --------
        dict : Sensitivity analysis results
        """
        from src.backtesting.engine import BacktestEngine
        
        results = {}
        
        for param_name, param_range in param_ranges.items():
            param_results = []
            
            for value in param_range:
                # Create strategy with modified parameter
                test_params = base_params.copy()
                test_params[param_name] = value
                
                strategy = strategy_class(**test_params)
                engine = BacktestEngine()
                backtest = engine.run_backtest(data, strategy)
                
                param_results.append({
                    'value': value,
                    'sharpe': backtest['sharpe_ratio'],
                    'return': backtest['total_return']
                })
            
            # Calculate sensitivity metrics
            sharpes = [r['sharpe'] for r in param_results]
            sensitivity = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else float('inf')
            
            results[param_name] = {
                'results': param_results,
                'sensitivity': sensitivity,
                'is_stable': sensitivity < 0.3,  # Coefficient of variation < 30%
                'optimal_value': param_results[np.argmax(sharpes)]['value']
            }
        
        # Overall assessment
        unstable_params = [p for p, r in results.items() if not r['is_stable']]
        
        return {
            'parameter_results': results,
            'unstable_parameters': unstable_params,
            'is_robust': len(unstable_params) == 0,
            'recommendation': 'Parameters stable' if len(unstable_params) == 0 
                            else f'Unstable parameters: {unstable_params}'
        }
    
    @staticmethod
    def monte_carlo_analysis(strategy: Callable,
                            data: pd.DataFrame,
                            n_simulations: int = 1000,
                            noise_level: float = 0.02) -> Dict:
        """
        Test strategy robustness using Monte Carlo simulation
        
        Parameters:
        -----------
        strategy : Callable
            Strategy to test
        data : pd.DataFrame
            Historical data
        n_simulations : int
            Number of Monte Carlo runs
        noise_level : float
            Standard deviation of noise to add
            
        Returns:
        --------
        dict : Monte Carlo analysis results
        """
        from src.backtesting.engine import BacktestEngine
        
        # Baseline performance
        engine = BacktestEngine()
        baseline = engine.run_backtest(data, strategy)
        baseline_sharpe = baseline['sharpe_ratio']
        
        # Run simulations with noise
        simulation_results = []
        
        for i in range(n_simulations):
            # Add random noise to prices
            noisy_data = data.copy()
            noise = np.random.normal(0, noise_level, len(data))
            noisy_data['Close'] = data['Close'] * (1 + noise)
            
            # Ensure price constraints
            noisy_data['High'] = np.maximum(noisy_data['High'], noisy_data['Close'])
            noisy_data['Low'] = np.minimum(noisy_data['Low'], noisy_data['Close'])
            
            # Run backtest on noisy data
            engine = BacktestEngine()
            result = engine.run_backtest(noisy_data, strategy)
            simulation_results.append(result['sharpe_ratio'])
        
        # Calculate statistics
        sim_mean = np.mean(simulation_results)
        sim_std = np.std(simulation_results)
        sim_min = np.min(simulation_results)
        sim_max = np.max(simulation_results)
        
        # Calculate probability of positive Sharpe
        prob_positive = np.mean([s > 0 for s in simulation_results])
        
        # Calculate Value at Risk (5th percentile)
        var_5 = np.percentile(simulation_results, 5)
        
        return {
            'baseline_sharpe': baseline_sharpe,
            'simulation_mean': sim_mean,
            'simulation_std': sim_std,
            'simulation_min': sim_min,
            'simulation_max': sim_max,
            'probability_positive_sharpe': prob_positive,
            'value_at_risk_5pct': var_5,
            'is_robust': prob_positive > 0.8 and var_5 > 0,
            'confidence_interval_95': (
                np.percentile(simulation_results, 2.5),
                np.percentile(simulation_results, 97.5)
            )
        }
