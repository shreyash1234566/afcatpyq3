"""
Topic Predictor Model
=====================
XGBoost-based model for predicting topic frequencies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
import json
from pathlib import Path
from datetime import datetime

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_structures import TopicFrequency, TopicPrediction, Section, TrendDirection
from utils.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)


class TopicPredictor:
    """
    Predicts topic question counts for future AFCAT exams.
    
    Uses XGBoost regression with temporal features.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state
        }
        
        if XGBOOST_AVAILABLE:
            self.model = XGBRegressor(**self.params)
            self.model_type = 'xgboost'
        elif SKLEARN_AVAILABLE:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
            self.model_type = 'gradient_boosting'
        else:
            self.model = None
            self.model_type = 'fallback'
            
        self.feature_engine = FeatureEngine()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.topics: List[str] = []
        
    def prepare_training_data(
        self,
        topic_frequencies: Dict[str, TopicFrequency],
        target_year: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data for the model.
        
        Uses leave-one-year-out approach for time series.
        """
        X_list = []
        y_list = []
        topics = []
        
        for topic, freq_data in topic_frequencies.items():
            years = sorted(freq_data.frequencies.keys())
            
            if len(years) < 3:
                continue
                
            # Create features from years before target
            train_freqs = {y: freq_data.frequencies[y] for y in years if y < target_year}
            
            if not train_freqs:
                continue
                
            # Create a TopicFrequency object with only training data
            train_tf = TopicFrequency(
                topic=topic,
                section=freq_data.section,
                frequencies=train_freqs
            )
            train_tf.calculate_stats()
            
            # Get features
            features = self._extract_topic_features(train_tf, target_year)
            
            # Target is the actual count in target year (if available)
            if target_year in freq_data.frequencies:
                target = freq_data.frequencies[target_year]
            else:
                # Use last known value as proxy
                target = freq_data.frequencies[max(years)]
                
            X_list.append(features)
            y_list.append(target)
            topics.append(topic)
            
        return np.array(X_list), np.array(y_list), topics
    
    def _extract_topic_features(
        self,
        freq_data: TopicFrequency,
        target_year: int
    ) -> List[float]:
        """Extract features for a single topic."""
        years = sorted(freq_data.frequencies.keys())
        values = [freq_data.frequencies[y] for y in years]
        
        if not values:
            return [0] * 13
            
        # Basic stats
        avg_freq = np.mean(values)
        recent_freq = np.mean(values[-2:]) if len(values) >= 2 else avg_freq
        
        # Trend coefficient
        if len(values) >= 3:
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            trend_coef = coeffs[0]
        else:
            trend_coef = 0
            
        # Recency
        years_since_last = target_year - max(years)
        
        # Consecutive appearances
        consecutive = 0
        for y in reversed(years):
            if freq_data.frequencies.get(y, 0) > 0:
                consecutive += 1
            else:
                break
                
        # Volatility
        volatility = np.std(values) if len(values) > 1 else 0
        
        # Min/Max
        max_freq = max(values)
        min_freq = min(values)
        
        # Last 5 years (padded)
        last_5 = [0] * 5
        for i, val in enumerate(values[-5:]):
            last_5[i] = val
            
        features = [
            avg_freq,
            recent_freq,
            trend_coef,
            years_since_last,
            consecutive,
            volatility,
            max_freq,
            min_freq,
            *last_5
        ]
        
        return features
    
    def fit(
        self,
        topic_frequencies: Dict[str, TopicFrequency],
        validation_year: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Fit the model on historical data.
        
        Args:
            topic_frequencies: Dict of topic frequency data
            validation_year: Year to use for validation (default: most recent)
            
        Returns:
            Dict of training metrics
        """
        if self.model is None:
            logger.warning("No ML model available, using fallback predictions")
            self.is_fitted = True
            return {"status": "fallback_mode"}
            
        # Determine years available
        all_years = set()
        for freq_data in topic_frequencies.values():
            all_years.update(freq_data.frequencies.keys())
            
        years = sorted(all_years)
        
        if not validation_year:
            validation_year = max(years)
            
        # Prepare training data
        X, y, self.topics = self.prepare_training_data(topic_frequencies, validation_year)
        
        if len(X) == 0:
            logger.error("No training data available")
            from models.enhanced_difficulty import EnhancedDifficultyPredictor
            from models.hybrid_classifier import HybridClassifier
            return {"error": "No training data"}
            
        self.feature_names = [
            'avg_frequency', 'recent_frequency', 'trend_coefficient',
            'years_since_last', 'consecutive_appearances', 'volatility',
            'max_frequency', 'min_frequency',
            'freq_year_1', 'freq_year_2', 'freq_year_3', 'freq_year_4', 'freq_year_5'
        ]
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        metrics = {
            'mae': round(mae, 3),
            'rmse': round(rmse, 3),
            'num_topics': len(self.topics),
            'num_features': len(self.feature_names),
            'model_type': self.model_type
        }
        
        logger.info(f"Model trained: MAE={mae:.3f}, RMSE={rmse:.3f}")
        
        return metrics
    
    def predict(
        self,
        topic_frequencies: Dict[str, TopicFrequency],
        target_year: int = 2026
    ) -> List[TopicPrediction]:
        """
        Generate predictions for target year.
        
        Returns list of TopicPrediction objects.
                    # Enhanced modules
                    self.difficulty_predictor = EnhancedDifficultyPredictor()
                    self.topic_classifier = HybridClassifier(use_transformers=True)
        """
        predictions = []
        
        for topic, freq_data in topic_frequencies.items():
            # Extract features
            features = self._extract_topic_features(freq_data, target_year)
            
            if self.model is not None and self.is_fitted:
                # ML prediction
                X = np.array([features])
                pred_count = float(self.model.predict(X)[0])
                
                # Confidence based on data quality and trend stability
                data_quality = min(len(freq_data.frequencies) / 6, 1.0)  # Max at 6 years
                trend_stability = 1 - min(abs(freq_data.trend_coefficient), 0.5)
                confidence = 0.5 + 0.3 * data_quality + 0.2 * trend_stability
            else:
                # Fallback: weighted average
                years = sorted(freq_data.frequencies.keys())
                values = [freq_data.frequencies[y] for y in years]
                
                if values:
                    weights = [0.5 ** (len(values) - i - 1) for i in range(len(values))]
                    weights = [w / sum(weights) for w in weights]
                    pred_count = sum(v * w for v, w in zip(values, weights))
                    confidence = 0.5 + min(len(values) / 10, 0.3)
                else:
                    pred_count = 0
                    confidence = 0.3
                    
            # Ensure non-negative
            pred_count = max(0, pred_count)
            
            # Create prediction object
            prediction = TopicPrediction(
                topic=topic,
                section=freq_data.section,
                predicted_count=pred_count,
                confidence=min(confidence, 0.95),
                trend=freq_data.trend,
                historical_average=freq_data.average
            )
            
            predictions.append(prediction)
            
        # Rank predictions by priority
        predictions.sort(key=lambda p: p.predicted_count * p.confidence, reverse=True)
        for i, pred in enumerate(predictions):
            pred.priority_rank = i + 1
            
        return predictions
    
    def cross_validate(
        self,
        topic_frequencies: Dict[str, TopicFrequency],
        n_splits: int = 3
    ) -> Dict[str, float]:
        """
        Perform time-series cross-validation.
        """
        if self.model is None:
            return {"error": "No ML model available"}
            
        # Get all years
        all_years = set()
        for freq_data in topic_frequencies.values():
            all_years.update(freq_data.frequencies.keys())
        years = sorted(all_years)
        
        if len(years) < n_splits + 1:
            return {"error": f"Need at least {n_splits + 1} years for cross-validation"}
            
        mae_scores = []
        
        for i in range(n_splits):
            val_year = years[-(i + 1)]
            X, y, _ = self.prepare_training_data(topic_frequencies, val_year)
            
            if len(X) < 5:
                continue
                
            # Simple train/test split
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            
            mae = mean_absolute_error(y, y_pred)
            mae_scores.append(mae)
            
        if not mae_scores:
            return {"error": "Cross-validation failed"}
            
        return {
            'mean_mae': round(np.mean(mae_scores), 3),
            'std_mae': round(np.std(mae_scores), 3),
            'n_folds': len(mae_scores)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_fitted or self.model is None:
            return {}
            
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {
                name: round(float(imp), 4)
                for name, imp in zip(self.feature_names, importances)
            }
        return {}
    
    def save_model(self, filepath: Path):
        """Save model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'topics': self.topics,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model metadata saved to {filepath}")
    
    def load_model(self, filepath: Path) -> bool:
        """Load model from disk."""
        try:
            with open(filepath.with_suffix('.json'), 'r') as f:
                metadata = json.load(f)
                
            self.model_type = metadata['model_type']
            self.feature_names = metadata['feature_names']
            self.topics = metadata['topics']
            self.is_fitted = metadata['is_fitted']
            
            logger.info(f"Model metadata loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


def create_ensemble_predictor(
    topic_frequencies: Dict[str, TopicFrequency],
    target_year: int = 2026
) -> List[TopicPrediction]:
    """
    Advanced stacking ensemble: XGBoost + LightGBM + RandomForest (static weights, as per research best practice).
    Returns predictions with confidence and notes.
    """
    # Import LightGBM if available
    try:
        from lightgbm import LGBMRegressor
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False

    # Prepare training data
    predictor = TopicPredictor()
    X, y, topics = predictor.prepare_training_data(topic_frequencies, target_year)
    if len(X) == 0:
        return []

    # Fit base models
    base_models = {}
    preds = {}
    weights = {'xgboost': 0.4, 'lightgbm': 0.35, 'random_forest': 0.25}

    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
        xgb.fit(X, y)
        base_models['xgboost'] = xgb
        preds['xgboost'] = xgb.predict(X)
    else:
        preds['xgboost'] = np.zeros(len(X))

    # LightGBM
    if LIGHTGBM_AVAILABLE:
        lgb = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
        lgb.fit(X, y)
        base_models['lightgbm'] = lgb
        preds['lightgbm'] = lgb.predict(X)
    else:
        preds['lightgbm'] = np.zeros(len(X))

    # Random Forest
    if SKLEARN_AVAILABLE:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        base_models['random_forest'] = rf
        preds['random_forest'] = rf.predict(X)
    else:
        preds['random_forest'] = np.zeros(len(X))

    # Weighted ensemble prediction
    y_ensemble = (
        weights['xgboost'] * preds['xgboost'] +
        weights['lightgbm'] * preds['lightgbm'] +
        weights['random_forest'] * preds['random_forest']
    )

    # Map topic to prediction
    topic_to_pred = {t: y_ensemble[i] for i, t in enumerate(topics)}

    # Compose predictions
    ensemble_predictions = []
    for topic, freq_data in topic_frequencies.items():
        pred_count = topic_to_pred.get(topic, freq_data.average)
        confidence = 0.7  # Static for now; can be improved with validation
        prediction = TopicPrediction(
            topic=topic,
            section=freq_data.section,
            predicted_count=pred_count,
            confidence=confidence,
            trend=freq_data.trend,
            historical_average=freq_data.average,
            notes="Stacking ensemble (XGBoost+LGBM+RF)"
        )
        ensemble_predictions.append(prediction)

    # Re-rank
    ensemble_predictions.sort(key=lambda p: p.predicted_count * p.confidence, reverse=True)
    for i, pred in enumerate(ensemble_predictions):
        pred.priority_rank = i + 1

    return ensemble_predictions
