"""
Query Classifier - Categorize user queries to determine processing approach
Helps route queries to appropriate data sources (database, web search, historical docs, etc.)
"""

import re
from typing import Dict, List
from datetime import datetime


class QueryClassifier:
    """
    Classifies sponsorship queries into categories to determine optimal processing approach.
    """

    # Query type constants
    SPONSOR_CHECK = "sponsor_check"        # Checking if sponsor exists/can be proposed
    TEMPORAL = "temporal"                   # Time-sensitive, needs current/recent info
    HISTORICAL = "historical"               # About past events/partnerships
    ANALYSIS = "analysis"                   # Deep analysis, comparisons, recommendations
    GENERAL = "general"                     # General questions about sponsorships
    LIST_REQUEST = "list_request"          # Asking for lists of sponsors/categories
    OFF_TOPIC = "off_topic"                # Query not related to sponsorships

    def __init__(self):
        # Keywords for each category
        self.sponsor_check_keywords = [
            'propose', 'suggest', 'recommend', 'can we', 'should we',
            'conflict', 'compete', 'already have', 'existing sponsor'
        ]

        self.temporal_keywords = [
            'recent', 'current', 'latest', 'today', 'now', 'this year',
            '2024', '2025', 'new', 'upcoming', 'just signed', 'recently'
        ]

        self.historical_keywords = [
            'history', 'past', 'previous', 'former', 'used to', 'before',
            'originated', 'began', 'started', '1990s', '2000s', 'decade ago'
        ]

        self.analysis_keywords = [
            'compare', 'analyze', 'evaluate', 'why', 'how does', 'what if',
            'pros and cons', 'advantages', 'disadvantages', 'strategy',
            'recommend', 'best', 'better', 'versus', 'vs'
        ]

        self.list_keywords = [
            'list', 'all sponsors', 'what sponsors', 'who are', 'show me',
            'tell me about our', 'current partners', 'charter partners'
        ]

        # Sponsorship-related keywords to determine if query is on-topic
        self.sponsorship_keywords = [
            'sponsor', 'sponsorship', 'partner', 'partnership', 'brand', 'deal',
            'contract', 'agreement', 'nike', 'adidas', 'under armour', 'charter',
            'apparel', 'footwear', 'athletic', 'athletics', 'sports', 'college',
            'university', 'oregon', 'uo', 'ducks', 'marketing', 'endorsement'
        ]

    def classify(self, query: str) -> Dict:
        """
        Classify a query and return classification details.

        Args:
            query: User's question

        Returns:
            {
                'type': str,           # Primary classification
                'confidence': float,   # 0.0 to 1.0
                'keywords': List[str], # Matched keywords
                'entities': List[str], # Detected company/brand names
                'needs_web': bool,     # Whether web search is recommended
                'needs_db': bool,      # Whether sponsor DB check needed
                'time_sensitive': bool # Whether recency matters
            }
        """
        query_lower = query.lower()

        # Extract potential company/brand names (capitalized words)
        entities = self._extract_entities(query)

        # Check if query is related to sponsorships at all
        is_on_topic = self._is_sponsorship_related(query_lower)

        # If off-topic, return immediately
        if not is_on_topic:
            return {
                'type': self.OFF_TOPIC,
                'confidence': 0.9,
                'keywords': [],
                'entities': [],
                'needs_web': False,
                'needs_db': False,
                'time_sensitive': False,
                'all_scores': {}
            }

        # Score each category
        scores = {
            self.SPONSOR_CHECK: self._score_keywords(query_lower, self.sponsor_check_keywords),
            self.TEMPORAL: self._score_keywords(query_lower, self.temporal_keywords),
            self.HISTORICAL: self._score_keywords(query_lower, self.historical_keywords),
            self.ANALYSIS: self._score_keywords(query_lower, self.analysis_keywords),
            self.LIST_REQUEST: self._score_keywords(query_lower, self.list_keywords),
        }

        # Determine primary type (highest score, or GENERAL if all low)
        max_score = max(scores.values())

        if max_score < 0.3:
            primary_type = self.GENERAL
            confidence = 0.5
        else:
            primary_type = max(scores, key=scores.get)
            confidence = max_score

        # Determine what resources are needed
        needs_web = scores[self.TEMPORAL] > 0.4 or self._has_year_reference(query, current_year=True)
        needs_db = scores[self.SPONSOR_CHECK] > 0.3 or len(entities) > 0
        time_sensitive = scores[self.TEMPORAL] > 0.3

        # Get matched keywords
        matched_keywords = self._get_matched_keywords(query_lower)

        return {
            'type': primary_type,
            'confidence': round(confidence, 2),
            'keywords': matched_keywords,
            'entities': entities,
            'needs_web': needs_web,
            'needs_db': needs_db,
            'time_sensitive': time_sensitive,
            'all_scores': scores
        }

    def _is_sponsorship_related(self, query: str) -> bool:
        """Check if query is related to sponsorships at all."""
        # Check if any sponsorship keywords are present
        return any(keyword in query for keyword in self.sponsorship_keywords)

    def _score_keywords(self, query: str, keywords: List[str]) -> float:
        """Score how many keywords match in the query."""
        matches = sum(1 for kw in keywords if kw in query)
        if matches == 0:
            return 0.0
        # Normalize by number of keywords and query length
        return min(1.0, matches / max(1, len(keywords) / 5))

    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential company/brand names (capitalized words)."""
        # Remove common words that are capitalized
        stop_words = {'I', 'Can', 'We', 'Should', 'What', 'Who', 'When', 'Where',
                     'Why', 'How', 'Tell', 'Show', 'Give', 'University', 'Oregon',
                     'UO', 'Athletics', 'Sports'}

        words = query.split()
        entities = []

        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            # Check if capitalized and not a stop word
            if clean_word and clean_word[0].isupper() and clean_word not in stop_words:
                entities.append(clean_word)

        return entities

    def _has_year_reference(self, query: str, current_year: bool = False) -> bool:
        """Check if query references specific years, especially recent ones."""
        current = datetime.now().year
        recent_years = [str(current), str(current - 1), str(current + 1)]

        if current_year:
            return any(year in query for year in recent_years)
        else:
            # Check for any 4-digit year
            return bool(re.search(r'\b(19|20)\d{2}\b', query))

    def _get_matched_keywords(self, query: str) -> List[str]:
        """Get list of all matched keywords from query."""
        all_keywords = (
            self.sponsor_check_keywords +
            self.temporal_keywords +
            self.historical_keywords +
            self.analysis_keywords +
            self.list_keywords
        )

        matched = [kw for kw in all_keywords if kw in query]
        return matched[:5]  # Return top 5

    def get_classification_summary(self, classification: Dict) -> str:
        """Get human-readable summary of classification."""
        type_map = {
            self.SPONSOR_CHECK: "Sponsor Eligibility Check",
            self.TEMPORAL: "Current/Recent Information Request",
            self.HISTORICAL: "Historical Information Request",
            self.ANALYSIS: "Analysis/Comparison Request",
            self.LIST_REQUEST: "List/Overview Request",
            self.GENERAL: "General Question"
        }

        summary = f"Query Type: {type_map.get(classification['type'], 'Unknown')}\n"
        summary += f"Confidence: {classification['confidence']:.0%}\n"

        if classification['entities']:
            summary += f"Detected Brands: {', '.join(classification['entities'])}\n"

        recommendations = []
        if classification['needs_db']:
            recommendations.append("Check sponsor database")
        if classification['needs_web']:
            recommendations.append("Use web search for current info")
        if classification['time_sensitive']:
            recommendations.append("Prioritize recent sources")

        if recommendations:
            summary += f"Recommended: {', '.join(recommendations)}"

        return summary


# Helper function for quick classification
def classify_query(query: str) -> Dict:
    """Quick helper to classify a query."""
    classifier = QueryClassifier()
    return classifier.classify(query)


if __name__ == "__main__":
    # Test the classifier
    classifier = QueryClassifier()

    test_queries = [
        "Can we propose Nike as a sponsor?",
        "What are recent sponsorship trends in college athletics?",
        "Tell me about our Nike partnership history",
        "Who are our current Charter Partners?",
        "Should we approach Adidas or Under Armour for a deal?",
        "What sponsors signed in 2024?",
        "Compare our sponsorship portfolio to other Pac-12 schools",
        "How did Phil Knight get involved with UO?"
    ]

    print("🧪 Query Classification Test\n")
    print("=" * 70)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 70)

        result = classifier.classify(query)
        print(classifier.get_classification_summary(result))

        print()
