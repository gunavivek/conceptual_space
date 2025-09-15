"""
Q2.1: Enhanced Intent Layer Module
Critical intelligence module for structured data intent classification
Focuses on table intersection, temporal lookup, and analytical operations
"""

import json
import os
import re
from typing import Dict, List, Set, Optional
from datetime import datetime
import numpy as np


class TableIntersectionDetector:
    """
    CRITICAL: Detects table navigation intent for geometric matching
    """

    def __init__(self):
        self.row_indicators = [
            "current", "state", "provision", "revenue", "cost", "expense",
            "assets", "liabilities", "equity", "income", "loss", "total",
            "net", "gross", "operating", "non-operating", "cash", "debt"
        ]
        self.column_indicators = [
            "2018", "2019", "2020", "2021", "2022", "2023", "2024",
            "year", "period", "quarter", "q1", "q2", "q3", "q4",
            "annual", "fiscal", "month", "jan", "feb", "mar", "apr",
            "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ]
        self.intersection_patterns = [
            r"what (?:was|is) (?:the )?(.+?) in (\d{4})",  # "What was X in YEAR"
            r"(.+?) for (?:the )?(\d{4})",                 # "X for YEAR"
            r"(\d{4}) (.+?) (?:was|is)",                   # "YEAR X was"
            r"(?:the )?(.+?) (?:in|for|during) (\d{4})",   # "X in YEAR"
        ]

    def detect_table_intersection(self, question_text: str) -> Dict:
        """
        Detect if question seeks table intersection (row × column lookup)
        """
        text_lower = question_text.lower()

        # Row reference detection
        row_matches = [indicator for indicator in self.row_indicators
                      if indicator in text_lower]

        # Column reference detection
        column_matches = [indicator for indicator in self.column_indicators
                         if indicator in text_lower]

        # Intersection pattern matching
        pattern_matches = []
        for pattern in self.intersection_patterns:
            matches = re.findall(pattern, text_lower)
            pattern_matches.extend(matches)

        # Calculate table intersection score
        intersection_score = self._calculate_intersection_score(
            row_matches, column_matches, pattern_matches
        )

        return {
            'score': intersection_score,
            'row_indicators': row_matches,
            'column_indicators': column_matches,
            'pattern_matches': pattern_matches,
            'intersection_likely': intersection_score > 0.6
        }

    def _calculate_intersection_score(self, row_matches: List[str],
                                    column_matches: List[str],
                                    pattern_matches: List) -> float:
        """
        Calculate table intersection likelihood score
        """
        score = 0.0

        # Row indicators contribute
        score += min(len(row_matches) * 0.3, 0.4)

        # Column indicators contribute
        score += min(len(column_matches) * 0.3, 0.4)

        # Pattern matches contribute heavily
        score += min(len(pattern_matches) * 0.4, 0.5)

        # Both row and column present = high table intersection likelihood
        if row_matches and column_matches:
            score += 0.3

        return min(score, 1.0)


class TemporalAnalysisEngine:
    """
    Analyzes temporal intent for time-based queries
    """

    def __init__(self):
        self.year_patterns = [r'\b(19|20)\d{2}\b', r'\b\d{4}\b']
        self.period_patterns = [
            r'quarter', r'q[1-4]', r'fiscal year', r'annual', r'yearly',
            r'month', r'weekly', r'daily', r'period'
        ]
        self.comparison_patterns = [
            r'from .+ to', r'between .+ and', r'year.over.year', r'yoy',
            r'compared to', r'versus', r'change', r'growth', r'increase',
            r'decrease', r'percentage change'
        ]

    def analyze_temporal_intent(self, question_text: str) -> Dict:
        """
        Analyze temporal lookup and comparison intent
        """
        text_lower = question_text.lower()

        # Year detection
        years = []
        for pattern in self.year_patterns:
            years.extend(re.findall(pattern, question_text))

        # Period detection
        periods = []
        for pattern in self.period_patterns:
            if re.search(pattern, text_lower):
                periods.append(pattern)

        # Comparison detection
        comparisons = []
        for pattern in self.comparison_patterns:
            if re.search(pattern, text_lower):
                comparisons.append(pattern)

        return {
            'temporal_lookup_score': self._calculate_temporal_score(years, periods),
            'comparison_temporal_score': self._calculate_comparison_score(comparisons, years),
            'detected_years': years,
            'detected_periods': periods,
            'comparison_indicators': comparisons
        }

    def _calculate_temporal_score(self, years: List[str], periods: List[str]) -> float:
        """Calculate temporal lookup score"""
        score = 0.0

        # Years contribute
        score += min(len(years) * 0.25, 0.5)

        # Periods contribute
        score += min(len(periods) * 0.2, 0.4)

        # Both present = temporal query
        if years and periods:
            score += 0.3

        return min(score, 1.0)

    def _calculate_comparison_score(self, comparisons: List[str], years: List[str]) -> float:
        """Calculate temporal comparison score"""
        score = 0.0

        # Comparison indicators
        score += min(len(comparisons) * 0.4, 0.6)

        # Multiple years suggest comparison
        if len(years) >= 2:
            score += 0.4

        return min(score, 1.0)


class AnalyticalOperationDetector:
    """
    Detects analytical operations like calculations, percentages, ratios
    """

    def __init__(self):
        self.calculation_keywords = [
            'percentage', 'percent', '%', 'ratio', 'rate', 'change', 'growth',
            'increase', 'decrease', 'difference', 'total', 'sum', 'average',
            'mean', 'calculate', 'compute', 'derive'
        ]
        self.aggregation_keywords = [
            'total', 'sum', 'aggregate', 'combined', 'overall', 'net',
            'gross', 'average', 'mean', 'maximum', 'minimum', 'count'
        ]
        self.trend_keywords = [
            'trend', 'pattern', 'growth', 'decline', 'trajectory', 'direction',
            'progression', 'evolution', 'development'
        ]
        self.numerical_patterns = [
            r'\$[\d,]+', r'[\d,]+\s*(?:million|billion|thousand)',
            r'\d+\.?\d*%', r'\d+\.?\d*\s*percent'
        ]

    def detect_analytical_operations(self, question_text: str) -> Dict:
        """
        Detect analytical operation requirements
        """
        text_lower = question_text.lower()

        # Operation type detection
        calc_matches = [kw for kw in self.calculation_keywords if kw in text_lower]
        agg_matches = [kw for kw in self.aggregation_keywords if kw in text_lower]
        trend_matches = [kw for kw in self.trend_keywords if kw in text_lower]

        # Numerical pattern detection
        numerical_matches = []
        for pattern in self.numerical_patterns:
            numerical_matches.extend(re.findall(pattern, question_text))

        return {
            'analytical_operation_score': min(len(calc_matches) * 0.3, 0.8),
            'aggregation_score': min(len(agg_matches) * 0.25, 0.6),
            'trend_analysis_score': min(len(trend_matches) * 0.2, 0.5),
            'numerical_extraction_score': min(len(numerical_matches) * 0.2, 0.4),
            'calculation_keywords': calc_matches,
            'aggregation_keywords': agg_matches,
            'trend_keywords': trend_matches,
            'numerical_patterns': numerical_matches
        }


class Q21_EnhancedIntentLayer:
    """
    Enhanced Intent Layer for Q-Pipeline
    Provides sophisticated intent classification for structured data navigation
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Q2.1 Enhanced Intent Layer

        Args:
            config: Configuration parameters
        """
        self.config = config or {
            "confidence_threshold": 0.7,
            "table_intersection_weight": 0.4,
            "temporal_analysis_weight": 0.3,
            "analytical_operation_weight": 0.3,
            "processing_timeout": 100
        }

        # Initialize detection engines
        self.table_detector = TableIntersectionDetector()
        self.temporal_engine = TemporalAnalysisEngine()
        self.analytical_detector = AnalyticalOperationDetector()

    def load_question_from_q1(self, question_id: str = None) -> Dict:
        """
        Load question from Q1 output file

        Args:
            question_id: Optional specific question ID

        Returns:
            Question data from Q1
        """
        q1_path = "../outputs/Q1_Question_ingestion.json"

        if not os.path.exists(q1_path):
            raise FileNotFoundError(f"Q1 output file not found: {q1_path}")

        with open(q1_path, 'r') as f:
            q1_data = json.load(f)

        # Handle different Q1 output formats
        if isinstance(q1_data, dict):
            if question_id:
                # Multi-question format
                if question_id in q1_data:
                    return q1_data[question_id]
                else:
                    raise ValueError(f"Question {question_id} not found in Q1 output")
            else:
                # Single question format or return first available
                if 'question_id' in q1_data:
                    return q1_data
                else:
                    # Get first question from multi-question format
                    first_key = next(iter(q1_data.keys()))
                    return q1_data[first_key]

        raise ValueError("Invalid Q1 output format")

    def analyze_traditional_intents(self, question_text: str) -> Dict:
        """
        Analyze traditional intent categories
        """
        text_lower = question_text.lower()

        # Comparison intent
        comparison_indicators = ['compare', 'versus', 'vs', 'difference', 'higher', 'lower',
                               'greater', 'less', 'more', 'than']
        comparison_score = sum(0.2 for ind in comparison_indicators if ind in text_lower)

        # Calculation intent
        calc_indicators = ['calculate', 'compute', 'determine', 'find', 'what is']
        calculation_score = sum(0.25 for ind in calc_indicators if ind in text_lower)

        # Definition intent
        def_indicators = ['what is', 'define', 'meaning', 'definition', 'explain']
        definition_score = sum(0.3 for ind in def_indicators if ind in text_lower)

        # Identification intent
        id_indicators = ['who', 'which', 'identify', 'name', 'list']
        identification_score = sum(0.25 for ind in id_indicators if ind in text_lower)

        # Factual intent (baseline)
        factual_score = 0.3  # Default factual component

        return {
            'comparison': min(comparison_score, 1.0),
            'calculation': min(calculation_score, 1.0),
            'definition': min(definition_score, 1.0),
            'identification': min(identification_score, 1.0),
            'factual': min(factual_score, 1.0)
        }

    def classify_enhanced_intent(self, question_text: str, doc_id: str) -> Dict:
        """
        Perform 13-dimensional enhanced intent classification

        Args:
            question_text: Raw question text
            doc_id: Document identifier for context

        Returns:
            Complete intent classification results
        """
        start_time = datetime.now()

        # Initialize all intent scores
        intent_scores = {
            # Traditional intents
            "comparison": 0.0,
            "calculation": 0.0,
            "definition": 0.0,
            "identification": 0.0,
            "factual": 0.0,

            # Enhanced structured data intents
            "table_intersection": 0.0,
            "temporal_lookup": 0.0,
            "numerical_extraction": 0.0,
            "analytical_operation": 0.0,
            "contextual_integration": 0.0,
            "aggregation": 0.0,
            "comparison_temporal": 0.0,
            "trend_analysis": 0.0
        }

        # Enhanced pattern analysis
        table_analysis = self.table_detector.detect_table_intersection(question_text)
        temporal_analysis = self.temporal_engine.analyze_temporal_intent(question_text)
        analytical_analysis = self.analytical_detector.detect_analytical_operations(question_text)

        # Update enhanced intent scores
        intent_scores.update({
            "table_intersection": table_analysis['score'],
            "temporal_lookup": temporal_analysis['temporal_lookup_score'],
            "comparison_temporal": temporal_analysis['comparison_temporal_score'],
            "analytical_operation": analytical_analysis['analytical_operation_score'],
            "aggregation": analytical_analysis['aggregation_score'],
            "trend_analysis": analytical_analysis['trend_analysis_score'],
            "numerical_extraction": analytical_analysis['numerical_extraction_score']
        })

        # Traditional intent analysis
        traditional_scores = self.analyze_traditional_intents(question_text)
        intent_scores.update(traditional_scores)

        # Contextual integration heuristic
        if (intent_scores["table_intersection"] > 0.5 and
            intent_scores["temporal_lookup"] > 0.3):
            intent_scores["contextual_integration"] = 0.6

        # Normalize scores to ensure proper distribution
        normalized_scores = self._normalize_intent_scores(intent_scores)

        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000

        # Compile structured data indicators
        structured_indicators = {
            "table_references": table_analysis['row_indicators'] + table_analysis['column_indicators'],
            "temporal_markers": temporal_analysis['detected_years'] + temporal_analysis['detected_periods'],
            "numerical_patterns": analytical_analysis['numerical_patterns'],
            "calculation_keywords": analytical_analysis['calculation_keywords'],
            "comparison_indicators": temporal_analysis['comparison_indicators']
        }

        return {
            'intent_classification': normalized_scores,
            'intent_vector': list(normalized_scores.values()),
            'primary_intent': max(normalized_scores, key=normalized_scores.get),
            'intent_confidence': self._calculate_confidence(normalized_scores),
            'structured_data_indicators': structured_indicators,
            'processing_metadata': {
                'classification_timestamp': end_time.isoformat(),
                'processing_time_ms': processing_time,
                'pattern_matches': (len(table_analysis['pattern_matches']) +
                                  len(temporal_analysis['comparison_indicators']) +
                                  len(analytical_analysis['calculation_keywords'])),
                'confidence_factors': self._identify_confidence_factors(normalized_scores)
            }
        }

    def _normalize_intent_scores(self, scores: Dict) -> Dict:
        """
        Normalize intent scores to ensure proper probability distribution
        """
        # Apply softmax-like normalization
        max_score = max(scores.values()) if scores.values() else 1.0

        # Prevent division by zero
        if max_score == 0:
            return scores

        # Scale scores to [0,1] range with emphasis on highest scores
        normalized = {}
        for intent, score in scores.items():
            if score > 0:
                normalized[intent] = min(score / max_score, 1.0)
            else:
                normalized[intent] = 0.0

        return normalized

    def _calculate_confidence(self, intent_scores: Dict) -> float:
        """
        Calculate confidence in intent classification
        """
        scores = [score for score in intent_scores.values() if score > 0]

        if len(scores) < 2:
            return 0.5  # Low confidence if only one score

        scores.sort(reverse=True)
        max_score = scores[0]
        second_max = scores[1]

        # High confidence if clear winner
        separation = max_score - second_max

        # Confidence based on score separation and magnitude
        confidence = min(1.0, (separation * 1.5) + (max_score * 0.3))

        return confidence

    def _identify_confidence_factors(self, scores: Dict) -> List[str]:
        """
        Identify factors contributing to classification confidence
        """
        factors = []

        # High-confidence indicators
        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]

        if max_score > 0.7:
            factors.append(f"strong_{max_intent}_signals")

        # Multiple intent detection
        high_scores = [intent for intent, score in scores.items() if score > 0.5]
        if len(high_scores) > 1:
            factors.append("multi_intent_detected")

        # Structured data confidence
        structured_intents = ["table_intersection", "temporal_lookup", "analytical_operation"]
        if any(scores[intent] > 0.6 for intent in structured_intents):
            factors.append("structured_data_strong")

        return factors

    def process_question(self, question_id: str = None) -> Dict:
        """
        Complete Q2.1 processing for a question

        Args:
            question_id: Optional specific question ID

        Returns:
            Complete Q2.1 output with enhanced intent classification
        """
        # Load question from Q1
        question_data = self.load_question_from_q1(question_id)

        # Extract required fields
        qid = question_data['question_id']
        question_text = question_data['question_text']
        doc_id = question_data['doc_id']

        print(f"Processing Q2.1 for question: {qid}")
        print(f"Question: {question_text[:80]}...")

        # Perform enhanced intent classification
        intent_results = self.classify_enhanced_intent(question_text, doc_id)

        # Compile final output
        output = {
            'question_id': qid,
            'doc_id': doc_id,
            'question_text': question_text,
            **intent_results
        }

        return output

    def save_output(self, output_data: Dict, output_path: str = None):
        """
        Save Q2.1 output for downstream modules

        Args:
            output_data: Q2.1 processing results
            output_path: Output file path
        """
        if output_path is None:
            output_path = "../outputs/Q2.1_enhanced_intent_classification.json"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load existing data if file exists
        existing_data = {}
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_data = json.load(f)

        # Add/update this question's data
        existing_data[output_data['question_id']] = output_data

        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=2)

        print(f"Q2.1 output saved to {output_path}")


if __name__ == "__main__":
    # Test Q2.1 module
    print("="*60)
    print("Q2.1: Enhanced Intent Layer Test")
    print("="*60)

    q21 = Q21_EnhancedIntentLayer()

    try:
        # Process question from Q1 output
        result = q21.process_question()

        print(f"\n{'='*40}")
        print("Q2.1 OUTPUT - Enhanced Intent Classification:")
        print(f"{'='*40}")

        print(f"Question ID: {result['question_id']}")
        print(f"Primary Intent: {result['primary_intent']}")
        print(f"Confidence: {result['intent_confidence']:.3f}")

        print(f"\nTop Intent Scores:")
        intent_scores = result['intent_classification']
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)

        for i, (intent, score) in enumerate(sorted_intents[:5]):
            if score > 0:
                print(f"  {i+1}. {intent}: {score:.3f}")

        print(f"\nStructured Data Indicators:")
        indicators = result['structured_data_indicators']
        for category, items in indicators.items():
            if items:
                print(f"  {category}: {items}")

        print(f"\nProcessing Metadata:")
        metadata = result['processing_metadata']
        print(f"  Processing time: {metadata['processing_time_ms']:.1f}ms")
        print(f"  Pattern matches: {metadata['pattern_matches']}")
        print(f"  Confidence factors: {metadata['confidence_factors']}")

        # Save output
        q21.save_output(result)

    except Exception as e:
        print(f"Error in Q2.1 testing: {e}")
        import traceback
        traceback.print_exc()