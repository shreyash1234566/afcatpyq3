"""
Trend Detection Module
======================
Detects temporal trends in topic frequencies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy import stats
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_structures import TopicFrequency, TrendDirection, Section

logger = logging.getLogger(__name__)


class TrendDetector:
    """
    Detects and analyzes temporal trends in AFCAT topic frequencies.
    
    Uses statistical methods to identify:
    - Linear trends (rising/falling)
    - Cyclical patterns (AFCAT-1 vs AFCAT-2)
    - Structural breaks (pattern changes)
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        min_data_points: int = 4
    ):
        self.significance_level = significance_level
        self.min_data_points = min_data_points
        
    def detect_linear_trend(
        self,
        frequencies: Dict[int, int]
    ) -> Tuple[TrendDirection, float, float]:
        """
        Detect linear trend using linear regression.
        
        Returns:
            (trend_direction, slope, p_value)
        """
        if len(frequencies) < self.min_data_points:
            return TrendDirection.STABLE, 0.0, 1.0
            
        years = sorted(frequencies.keys())
        values = [frequencies[y] for y in years]
        
        # Perform linear regression
        x = np.arange(len(years))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if p_value < self.significance_level:
            if slope > 0.1:  # Meaningful positive slope
                trend = TrendDirection.RISING
            elif slope < -0.1:  # Meaningful negative slope
                trend = TrendDirection.DECLINING
            else:
                trend = TrendDirection.STABLE
        else:
            trend = TrendDirection.STABLE
            
        return trend, slope, p_value
    
    def detect_seasonality(
        self,
        frequencies: Dict[int, int],
        period: int = 2
    ) -> Tuple[bool, float]:
        """
        Detect seasonal/cyclical patterns.
        
        For AFCAT: Check if AFCAT-1 (odd cycle) differs from AFCAT-2 (even cycle).
        
        Returns:
            (has_seasonality, seasonality_strength)
        """
        years = sorted(frequencies.keys())
        values = [frequencies[y] for y in years]
        
        if len(values) < period * 2:
            return False, 0.0
            
        # Split into odd and even cycles
        odd_values = values[::2]
        even_values = values[1::2]
        
        if not odd_values or not even_values:
            return False, 0.0
            
        # Perform t-test to check if means differ significantly
        try:
            t_stat, p_value = stats.ttest_ind(odd_values, even_values)
            
            has_seasonality = p_value < self.significance_level
            strength = abs(np.mean(odd_values) - np.mean(even_values)) / (np.mean(values) + 0.01)
            
            return has_seasonality, strength
        except:
            return False, 0.0
    
    def detect_structural_break(
        self,
        frequencies: Dict[int, int],
        break_year: Optional[int] = None
    ) -> Tuple[bool, Optional[int], float]:
        """
        Detect if there's a structural break (significant pattern change).
        
        Uses Chow test concept - compares variance before and after potential break.
        
        Returns:
            (has_break, break_year, break_magnitude)
        """
        years = sorted(frequencies.keys())
        values = [frequencies[y] for y in years]
        
        if len(values) < 6:
            return False, None, 0.0
            
        if break_year:
            # Test specific break year
            break_idx = years.index(break_year) if break_year in years else None
            if break_idx is None or break_idx < 2 or break_idx >= len(years) - 2:
                return False, None, 0.0
                
            break_indices = [break_idx]
        else:
            # Test all possible break points
            break_indices = range(2, len(years) - 2)
            
        best_break = None
        best_magnitude = 0
        
        for idx in break_indices:
            before = values[:idx]
            after = values[idx:]
            
            mean_before = np.mean(before)
            mean_after = np.mean(after)
            
            # Magnitude of change relative to overall mean
            magnitude = abs(mean_after - mean_before) / (np.mean(values) + 0.01)
            
            # Check significance with t-test
            try:
                _, p_value = stats.ttest_ind(before, after)
                
                if p_value < self.significance_level and magnitude > best_magnitude:
                    best_magnitude = magnitude
                    best_break = years[idx]
            except:
                continue
                
        has_break = best_break is not None and best_magnitude > 0.3
        
        return has_break, best_break, best_magnitude
    
    def forecast_next_value(
        self,
        frequencies: Dict[int, int],
        method: str = 'exponential_smoothing'
    ) -> Tuple[float, float]:
        """
        Forecast next year's value.
        
        Methods:
        - 'simple_average': Mean of all values
        - 'moving_average': Mean of last N values
        - 'exponential_smoothing': EMA with alpha decay
        - 'linear_extrapolation': Extend linear trend
        
        Returns:
            (predicted_value, confidence_interval)
        """
        years = sorted(frequencies.keys())
        values = [frequencies[y] for y in years]
        
        if not values:
            return 0.0, 0.0
            
        if method == 'simple_average':
            prediction = np.mean(values)
            ci = np.std(values) * 1.96
            
        elif method == 'moving_average':
            window = min(3, len(values))
            prediction = np.mean(values[-window:])
            ci = np.std(values[-window:]) * 1.96 if window > 1 else np.std(values)
            
        elif method == 'exponential_smoothing':
            alpha = 0.3
            ema = values[0]
            for val in values[1:]:
                ema = alpha * val + (1 - alpha) * ema
            prediction = ema
            
            # Confidence based on recent volatility
            recent_std = np.std(values[-3:]) if len(values) >= 3 else np.std(values)
            ci = recent_std * 1.96
            
        elif method == 'linear_extrapolation':
            x = np.arange(len(years))
            slope, intercept, _, _, std_err = stats.linregress(x, values)
            prediction = slope * len(years) + intercept
            ci = std_err * 1.96 * np.sqrt(len(years))
            
        else:
            prediction = values[-1] if values else 0
            ci = 0
            
        return max(0, prediction), ci
    
    def analyze_all_trends(
        self,
        topic_frequencies: Dict[str, TopicFrequency]
    ) -> Dict[str, Dict]:
        """
        Analyze trends for all topics.
        
        Returns comprehensive trend analysis for each topic.
        """
        results = {}
        
        for topic, freq_data in topic_frequencies.items():
            # Linear trend
            trend, slope, p_value = self.detect_linear_trend(freq_data.frequencies)
            
            # Seasonality
            has_seasonal, seasonal_strength = self.detect_seasonality(freq_data.frequencies)
            
            # Structural break
            has_break, break_year, break_magnitude = self.detect_structural_break(freq_data.frequencies)
            
            # Forecast
            prediction, ci = self.forecast_next_value(
                freq_data.frequencies,
                method='exponential_smoothing'
            )
            
            results[topic] = {
                'section': freq_data.section.value,
                'historical_average': freq_data.average,
                'trend': {
                    'direction': trend.value,
                    'slope': round(slope, 3),
                    'p_value': round(p_value, 4),
                    'significant': p_value < self.significance_level
                },
                'seasonality': {
                    'detected': has_seasonal,
                    'strength': round(seasonal_strength, 3)
                },
                'structural_break': {
                    'detected': has_break,
                    'year': break_year,
                    'magnitude': round(break_magnitude, 3)
                },
                'forecast': {
                    'predicted_count': round(prediction, 1),
                    'confidence_interval': round(ci, 1),
                    'range': (round(max(0, prediction - ci), 1), round(prediction + ci, 1))
                }
            }
            
        return results


def identify_afcat_2024_break(topic_frequencies: Dict[str, TopicFrequency]) -> Dict[str, bool]:
    """
    Specifically check for the 2024 structural break (Math: 18→20 questions).
    
    Returns dict of topics affected by the 2024 pattern change.
    """
    detector = TrendDetector()
    affected = {}
    
    math_topics = [
        t for t, f in topic_frequencies.items()
        if f.section == Section.NUMERICAL_ABILITY
    ]
    
    for topic in math_topics:
        freq_data = topic_frequencies[topic]
        has_break, break_year, magnitude = detector.detect_structural_break(
            freq_data.frequencies,
            break_year=2024
        )
        
        affected[topic] = has_break and break_year == 2024
        
    return affected


def get_hot_topics(
    trend_analysis: Dict[str, Dict],
    top_n: int = 10
) -> List[Dict]:
    """
    Get 'hot' topics - rising trends with high predicted counts.
    """
    hot_topics = []
    
    for topic, analysis in trend_analysis.items():
        if analysis['trend']['direction'] == 'rising':
            hot_topics.append({
                'topic': topic,
                'section': analysis['section'],
                'predicted': analysis['forecast']['predicted_count'],
                'trend_strength': abs(analysis['trend']['slope']),
                'historical_avg': analysis['historical_average']
            })
            
    # Sort by combination of prediction and trend strength
    hot_topics.sort(
        key=lambda x: x['predicted'] * (1 + x['trend_strength']),
        reverse=True
    )
    
    return hot_topics[:top_n]


def get_cold_topics(
    trend_analysis: Dict[str, Dict],
    top_n: int = 10
) -> List[Dict]:
    """
    Get 'cold' topics - declining trends to deprioritize.
    """
    cold_topics = []
    
    for topic, analysis in trend_analysis.items():
        if analysis['trend']['direction'] == 'declining':
            cold_topics.append({
                'topic': topic,
                'section': analysis['section'],
                'predicted': analysis['forecast']['predicted_count'],
                'decline_rate': abs(analysis['trend']['slope']),
                'historical_avg': analysis['historical_average']
            })
            
    # Sort by steepest decline
    cold_topics.sort(key=lambda x: x['decline_rate'], reverse=True)
    
    return cold_topics[:top_n]
