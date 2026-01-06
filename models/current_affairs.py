import requests
"""
Current Affairs Classifier
===========================
NLP-based classifier for AFCAT-relevant news articles.
"""

import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_structures import NewsArticle

logger = logging.getLogger(__name__)


class CurrentAffairsClassifier:
    """
    Classifies news articles by AFCAT relevance.
    Uses zero-shot classification or keyword-based fallback.
    """

    # AFCAT-relevant categories
    CATEGORIES = [
        "Defense & Military",
        "International Relations",
        "Sports",
        "Awards & Honors",
        "Science & Technology",
        "Government Schemes",
        "Economy & Budget",
        "Environment",
        "Art & Culture"
    ]
    # High-priority keywords for AFCAT
    PRIORITY_KEYWORDS = {
        "defense": [
            "indian air force", "iaf", "rafale", "tejas", "sukhoi", "mig",
            "missile", "brahmos", "agni", "prithvi", "akash",
            "drdo", "hal", "exercise", "operation", "bilateral",
            "army", "navy", "coast guard", "paramilitary",
            "defense budget", "chief of staff", "air marshal"
        ],
        "international": [
            "summit", "treaty", "agreement", "bilateral", "multilateral",
            "g20", "brics", "asean", "saarc", "un", "unesco",
            "prime minister visit", "foreign minister", "mou",
            "diplomatic", "ambassador", "high commissioner"
        ],
        "sports": [
            "olympic", "asian games", "commonwealth games", "world cup",
            "champion", "medal", "trophy", "tournament",
            "cricket", "hockey", "badminton", "tennis", "chess",
            "arjuna award", "dronacharya", "khel ratna", "padma"
        ],
        "science": [
            "isro", "space", "satellite", "launch", "chandrayaan", "mangalyaan",
            "gaganyaan", "pslv", "gslv", "rocket",
            "nuclear", "atomic", "research", "discovery", "innovation"
        ],
        "economy": [
            "budget", "gdp", "rbi", "repo rate", "inflation",
            "fiscal", "monetary", "niti aayog", "economic survey",
            "fdi", "export", "import", "trade"
        ],
        "awards": [
            "bharat ratna", "padma vibhushan", "padma bhushan", "padma shri",
            "nobel", "booker", "oscar", "grammy", "bafta",
            "gallantry", "param vir chakra", "ashoka chakra", "shaurya chakra"
        ]
    }

    def __init__(self, use_transformer: bool = True):
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.classifier = None

    def fetch_newsapi_articles(self, api_key: str, query: str = "India", from_date: str = None, to_date: str = None, page_size: int = 50) -> list:
        """Fetch news articles from NewsAPI.org for AFCAT relevance."""
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": api_key
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for art in data.get("articles", []):
                article = NewsArticle(
                    title=art.get("title", ""),
                    content=art.get("content", ""),
                    published_date=datetime.strptime(art.get("publishedAt", "1970-01-01T00:00:00Z")[:10], "%Y-%m-%d"),
                    url=art.get("url", "")
                )
                articles.append(article)
            return articles
        except Exception as e:
            logger.warning(f"Failed to fetch news: {e}")
            return []
        
        if self.use_transformer:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                logger.info("Loaded transformer model for classification")
            except Exception as e:
                logger.warning(f"Failed to load transformer: {e}. Using keyword-based.")
                self.use_transformer = False
                
    def classify(self, article: NewsArticle) -> NewsArticle:
        """
        Classify a news article and update its relevance scores.
        """
        text = f"{article.title} {article.content[:500]}"
        
        if self.use_transformer and self.classifier:
            result = self._classify_transformer(text)
        else:
            result = self._classify_keywords(text)
            
        article.category = result['category']
        article.relevance_score = result['relevance_score']
        article.afcat_probability = result['afcat_probability']
        article.key_facts = self._extract_key_facts(text)
        
        return article
    
    def _classify_transformer(self, text: str) -> Dict:
        """Use transformer model for classification."""
        result = self.classifier(
            text,
            self.CATEGORIES,
            multi_label=False
        )
        
        category = result['labels'][0]
        score = result['scores'][0]
        
        # Calculate AFCAT probability based on category importance
        category_weights = {
            "Defense & Military": 1.0,
            "Sports": 0.9,
            "Awards & Honors": 0.85,
            "International Relations": 0.8,
            "Science & Technology": 0.75,
            "Government Schemes": 0.7,
            "Economy & Budget": 0.65,
            "Environment": 0.5,
            "Art & Culture": 0.4
        }
        
        weight = category_weights.get(category, 0.5)
        afcat_prob = score * weight
        
        return {
            'category': category,
            'relevance_score': round(score, 3),
            'afcat_probability': round(afcat_prob, 3)
        }
    
    def _classify_keywords(self, text: str) -> Dict:
        """Keyword-based classification fallback."""
        text_lower = text.lower()
        
        # Count keyword matches per category
        category_scores = {}
        
        for category, keywords in self.PRIORITY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            category_scores[category] = score
            
        # Find best category
        if not category_scores or max(category_scores.values()) == 0:
            return {
                'category': 'General',
                'relevance_score': 0.1,
                'afcat_probability': 0.1
            }
            
        best_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[best_category]
        
        # Normalize score (max possible ~ 10 keywords)
        relevance = min(max_score / 5, 1.0)
        
        # Map internal category to display category
        category_map = {
            'defense': 'Defense & Military',
            'international': 'International Relations',
            'sports': 'Sports',
            'science': 'Science & Technology',
            'economy': 'Economy & Budget',
            'awards': 'Awards & Honors'
        }
        
        display_category = category_map.get(best_category, best_category.title())
        
        # AFCAT probability
        priority_categories = {'defense', 'sports', 'awards'}
        afcat_prob = relevance * (1.0 if best_category in priority_categories else 0.7)
        
        return {
            'category': display_category,
            'relevance_score': round(relevance, 3),
            'afcat_probability': round(afcat_prob, 3)
        }
    
    def _extract_key_facts(self, text: str) -> List[str]:
        """Extract key facts/entities from text."""
        facts = []
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            facts.extend([f"Date: {m}" for m in matches[:2]])
            
        # Extract numbers with context
        number_pattern = r'(\d+(?:\.\d+)?)\s*(crore|lakh|million|billion|km|meters|years)'
        matches = re.findall(number_pattern, text, re.IGNORECASE)
        facts.extend([f"{m[0]} {m[1]}" for m in matches[:3]])
        
        # Extract names (capitalized sequences)
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        matches = re.findall(name_pattern, text)
        unique_names = list(set(matches))[:3]
        facts.extend([f"Name: {n}" for n in unique_names])
        
        return facts[:5]  # Limit to 5 key facts
    
    def batch_classify(
        self,
        articles: List[NewsArticle],
        min_probability: float = 0.5
    ) -> List[NewsArticle]:
        """
        Classify multiple articles and filter by probability.
        """
        classified = [self.classify(article) for article in articles]
        
        # Filter by minimum probability
        filtered = [a for a in classified if a.afcat_probability >= min_probability]
        
        # Sort by probability
        filtered.sort(key=lambda a: a.afcat_probability, reverse=True)
        
        return filtered
    
    def get_weekly_digest(
        self,
        articles: List[NewsArticle],
        top_n: int = 20
    ) -> Dict[str, List[Dict]]:
        """
        Generate weekly digest of AFCAT-relevant news.
        """
        classified = self.batch_classify(articles, min_probability=0.4)
        
        # Group by category
        by_category = {}
        for article in classified[:top_n]:
            category = article.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(article.to_dict())
            
        # Sort categories by importance
        category_order = [
            "Defense & Military",
            "Sports",
            "Awards & Honors",
            "International Relations",
            "Science & Technology",
            "Government Schemes",
            "Economy & Budget"
        ]
        
        ordered_digest = {}
        for cat in category_order:
            if cat in by_category:
                ordered_digest[cat] = by_category[cat]
                
        # Add remaining categories
        for cat, articles in by_category.items():
            if cat not in ordered_digest:
                ordered_digest[cat] = articles
                
        return ordered_digest


def generate_current_affairs_summary(
    articles: List[NewsArticle],
    months_lookback: int = 6
) -> Dict:
    """
    Generate summary of current affairs for AFCAT preparation.
    """
    classifier = CurrentAffairsClassifier(use_transformer=False)  # Use fast keyword method
    
    # Filter by date
    cutoff_date = datetime.now() - timedelta(days=months_lookback * 30)
    recent_articles = [a for a in articles if a.published_date >= cutoff_date]
    
    # Classify all
    classified = classifier.batch_classify(recent_articles, min_probability=0.3)
    
    # Generate statistics
    category_counts = {}
    for article in classified:
        cat = article.category
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
    # Top articles per category
    top_articles = classifier.get_weekly_digest(classified, top_n=30)
    
    return {
        'period': f"Last {months_lookback} months",
        'total_articles': len(classified),
        'category_distribution': category_counts,
        'top_articles_by_category': top_articles,
        'high_priority_count': len([a for a in classified if a.afcat_probability >= 0.7]),
        'recommendation': "Focus on Defense & Military and Sports categories for highest AFCAT probability."
    }


def create_mock_news_data() -> List[NewsArticle]:
    """
    Create sample news articles for testing.
    """
    sample_articles = [
        NewsArticle(
            title="IAF Conducts Exercise Vayu Shakti 2026",
            content="The Indian Air Force conducted its firepower demonstration Exercise Vayu Shakti at Pokhran range. The exercise showcased the IAF's combat capabilities with participation of Rafale, Tejas, and Sukhoi fighters.",
            source="The Hindu",
            published_date=datetime(2025, 12, 15),
            url="https://example.com/vayu-shakti"
        ),
        NewsArticle(
            title="India Wins Hockey Asia Cup 2025",
            content="Indian Men's Hockey team won the Asia Cup 2025 defeating Pakistan 3-1 in the final. Captain Harmanpreet Singh scored two goals. This is India's 5th Asia Cup title.",
            source="Times of India",
            published_date=datetime(2025, 11, 20),
            url="https://example.com/hockey-asia-cup"
        ),
        NewsArticle(
            title="ISRO Successfully Launches Chandrayaan-4",
            content="ISRO launched Chandrayaan-4 mission from Sriharikota. The mission aims to bring back lunar samples to Earth. The GSLV Mk-III rocket carried the spacecraft.",
            source="Indian Express",
            published_date=datetime(2025, 12, 1),
            url="https://example.com/chandrayaan-4"
        ),
        NewsArticle(
            title="Padma Awards 2026 Announced",
            content="President announced Padma Awards 2026. Former ISRO Chief received Bharat Ratna. 15 Padma Vibhushan, 25 Padma Bhushan, and 100 Padma Shri awards were conferred.",
            source="PIB",
            published_date=datetime(2026, 1, 1),
            url="https://example.com/padma-2026"
        ),
        NewsArticle(
            title="India-France Sign Defense Agreement",
            content="India and France signed a new defense cooperation agreement during PM's visit to Paris. The agreement includes joint development of next-generation fighter aircraft.",
            source="Economic Times",
            published_date=datetime(2025, 10, 15),
            url="https://example.com/india-france"
        )
    ]
    
    return sample_articles
