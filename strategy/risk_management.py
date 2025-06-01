"""
Risk Management Module for Crypto Trading
- Position sizing based on feature importance
- Dynamic risk checks
- Portfolio management
- Drawdown monitoring
- Correlation analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_volatility: float = 0.5,
                 max_drawdown: float = 0.2,
                 max_correlation: float = 0.7,
                 min_confidence: float = 0.7,
                 feature_importance_threshold: float = 0.01):
        """
        Initialize risk management parameters.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_volatility: Maximum allowed volatility
            max_drawdown: Maximum allowed drawdown
            max_correlation: Maximum allowed correlation with market
            min_confidence: Minimum confidence threshold for signals
            feature_importance_threshold: Minimum importance for features to be considered
        """
        self.max_position_size = max_position_size
        self.max_volatility = max_volatility
        self.max_drawdown = max_drawdown
        self.max_correlation = max_correlation
        self.min_confidence = min_confidence
        self.feature_importance_threshold = feature_importance_threshold
        self.positions = {}
        self.portfolio_value = 0.0
        self.peak_value = 0.0
        self.feature_importance = None
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def update_feature_importance(self, feature_importance: Dict[str, float]):
        """Update feature importance scores."""
        self.feature_importance = feature_importance
        logger.info("Updated feature importance scores")

    def calculate_feature_based_position_size(self, 
                                           signal: Dict,
                                           available_capital: float) -> Tuple[float, Dict]:
        """
        Calculate position size based on feature importance and signal strength.
        
        Args:
            signal: Dictionary containing signal information
            available_capital: Available capital for the position
            
        Returns:
            Tuple of (position_size, position_metrics)
        """
        if not self.feature_importance:
            logger.warning("No feature importance scores available")
            return 0.0, {}

        # Calculate weighted signal strength based on feature importance
        weighted_strength = 0.0
        total_importance = 0.0
        feature_contributions = {}

        for feature, importance in self.feature_importance.items():
            if importance >= self.feature_importance_threshold:
                if feature in signal['feature_values']:
                    contribution = importance * signal['feature_values'][feature]
                    weighted_strength += contribution
                    total_importance += importance
                    feature_contributions[feature] = contribution

        if total_importance > 0:
            weighted_strength /= total_importance

        # Calculate base position size
        base_size = abs(weighted_strength) * self.max_position_size

        # Adjust for volatility
        vol_adjustment = 1 / (1 + signal['risk_metrics']['volatility'])
        position_size = base_size * vol_adjustment

        # Calculate position metrics
        position_metrics = {
            'weighted_strength': weighted_strength,
            'feature_contributions': feature_contributions,
            'volatility_adjustment': vol_adjustment,
            'base_size': base_size,
            'final_size': position_size
        }

        logger.info(f"Position size calculation: base={base_size:.2f}, "
                   f"vol_adjustment={vol_adjustment:.2f}, final={position_size:.2f}")

        return position_size, position_metrics

    def calculate_dynamic_risk_limits(self, market_conditions: Dict) -> Dict:
        """
        Calculate dynamic risk limits based on market conditions.
        
        Args:
            market_conditions: Dictionary containing market condition metrics
            
        Returns:
            Dictionary of adjusted risk limits
        """
        # Adjust position size based on market volatility
        vol_factor = 1 / (1 + market_conditions.get('market_volatility', 0))
        adjusted_position_size = self.max_position_size * vol_factor

        # Adjust drawdown limit based on market trend
        trend_factor = 1 + market_conditions.get('market_trend', 0)
        adjusted_drawdown = self.max_drawdown * trend_factor

        # Adjust correlation limit based on market regime
        regime_factor = 1 - market_conditions.get('regime_volatility', 0)
        adjusted_correlation = self.max_correlation * regime_factor

        return {
            'position_size': adjusted_position_size,
            'drawdown': adjusted_drawdown,
            'correlation': adjusted_correlation,
            'volatility': self.max_volatility * vol_factor
        }

    def risk_check(self, signal: Dict, market_conditions: Dict) -> Tuple[bool, Dict]:
        """
        Perform enhanced risk checks before executing a trade.
        
        Args:
            signal: Dictionary containing signal information
            market_conditions: Dictionary containing market condition metrics
            
        Returns:
            Tuple of (passes_checks, risk_metrics)
        """
        # Get dynamic risk limits
        risk_limits = self.calculate_dynamic_risk_limits(market_conditions)
        
        # Initialize risk metrics
        risk_metrics = {
            'checks_passed': 0,
            'total_checks': 5,
            'failed_checks': []
        }

        # Volatility check
        if signal['risk_metrics']['volatility'] > risk_limits['volatility']:
            risk_metrics['failed_checks'].append('volatility')
            logger.warning(f"Trade rejected: Volatility {signal['risk_metrics']['volatility']:.2f} "
                         f"exceeds maximum {risk_limits['volatility']:.2f}")
        else:
            risk_metrics['checks_passed'] += 1

        # Drawdown check
        if signal['risk_metrics']['drawdown'] > risk_limits['drawdown']:
            risk_metrics['failed_checks'].append('drawdown')
            logger.warning(f"Trade rejected: Drawdown {signal['risk_metrics']['drawdown']:.2f} "
                         f"exceeds maximum {risk_limits['drawdown']:.2f}")
        else:
            risk_metrics['checks_passed'] += 1

        # Correlation check
        if signal['risk_metrics']['correlation'] > risk_limits['correlation']:
            risk_metrics['failed_checks'].append('correlation')
            logger.warning(f"Trade rejected: Correlation {signal['risk_metrics']['correlation']:.2f} "
                         f"exceeds maximum {risk_limits['correlation']:.2f}")
        else:
            risk_metrics['checks_passed'] += 1

        # Position size check
        if signal['position_size'] > risk_limits['position_size']:
            risk_metrics['failed_checks'].append('position_size')
            logger.warning(f"Trade rejected: Position size {signal['position_size']:.2f} "
                         f"exceeds maximum {risk_limits['position_size']:.2f}")
        else:
            risk_metrics['checks_passed'] += 1

        # Confidence check
        if signal['confidence'] < self.min_confidence:
            risk_metrics['failed_checks'].append('confidence')
            logger.warning(f"Trade rejected: Confidence {signal['confidence']:.2f} "
                         f"below minimum {self.min_confidence:.2f}")
        else:
            risk_metrics['checks_passed'] += 1

        # Calculate risk score
        risk_metrics['risk_score'] = risk_metrics['checks_passed'] / risk_metrics['total_checks']
        
        # Trade passes if all checks pass
        passes_checks = risk_metrics['checks_passed'] == risk_metrics['total_checks']
        
        return passes_checks, risk_metrics

    def update_portfolio(self, 
                        position_id: str, 
                        size: float, 
                        price: float, 
                        timestamp: pd.Timestamp,
                        risk_metrics: Dict):
        """
        Update portfolio with new position and risk metrics.
        
        Args:
            position_id: Unique identifier for the position
            size: Position size in base currency
            price: Current price
            timestamp: Timestamp of the update
            risk_metrics: Dictionary containing risk metrics
        """
        self.positions[position_id] = {
            'size': size,
            'entry_price': price,
            'current_price': price,
            'timestamp': timestamp,
            'pnl': 0.0,
            'risk_metrics': risk_metrics
        }
        
        self.portfolio_value = sum(pos['size'] * pos['current_price'] 
                                 for pos in self.positions.values())
        self.peak_value = max(self.peak_value, self.portfolio_value)

    def calculate_portfolio_metrics(self) -> Dict:
        """
        Calculate enhanced portfolio performance metrics.
        
        Returns:
            Dictionary containing portfolio metrics
        """
        if not self.positions:
            return {
                'total_value': 0.0,
                'drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'position_count': 0,
                'avg_risk_score': 0.0
            }
            
        # Calculate returns
        returns = pd.Series([pos['pnl'] for pos in self.positions.values()])
        
        # Calculate metrics
        metrics = {
            'total_value': self.portfolio_value,
            'drawdown': (self.peak_value - self.portfolio_value) / self.peak_value,
            'sharpe_ratio': returns.mean() / returns.std() if len(returns) > 1 else 0.0,
            'sortino_ratio': returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0.0,
            'max_drawdown': abs(returns.min()) if len(returns) > 0 else 0.0,
            'win_rate': len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0.0,
            'position_count': len(self.positions),
            'avg_risk_score': np.mean([pos['risk_metrics'].get('risk_score', 0) for pos in self.positions.values()])
        }
        
        return metrics

    def generate_risk_report(self) -> str:
        """
        Generate a detailed risk report.
        
        Returns:
            Formatted risk report string
        """
        metrics = self.calculate_portfolio_metrics()
        
        report = f"""
Risk Management Report
---------------------
Portfolio Value: ${metrics['total_value']:,.2f}
Current Drawdown: {metrics['drawdown']*100:.2f}%
Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Win Rate: {metrics['win_rate']*100:.2f}%
Active Positions: {metrics['position_count']}
Average Risk Score: {metrics['avg_risk_score']:.2f}

Risk Limits:
- Max Position Size: {self.max_position_size*100:.1f}%
- Max Volatility: {self.max_volatility*100:.1f}%
- Max Drawdown: {self.max_drawdown*100:.1f}%
- Max Correlation: {self.max_correlation*100:.1f}%
- Min Confidence: {self.min_confidence*100:.1f}%
- Feature Importance Threshold: {self.feature_importance_threshold:.3f}

Top Feature Contributions:
{self._format_feature_contributions()}
"""
        return report

    def _format_feature_contributions(self) -> str:
        """Format feature contributions for the risk report."""
        if not self.feature_importance:
            return "No feature importance data available"
            
        top_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return "\n".join([f"- {feature}: {importance:.3f}" 
                         for feature, importance in top_features]) 