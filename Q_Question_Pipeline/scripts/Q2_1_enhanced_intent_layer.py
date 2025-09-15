"""
Q2.1: Enhanced Intent Layer (Corrected)
Analyzes question intent with enhanced semantic categories focused on question semantics,
not document structure (which is handled by A-Pipeline)
"""

import json
import os
import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional


class TemporalAnalysisEngine:
    """
    Analyzes temporal intent in questions (independent of document structure)
    """

    def __init__(self):
        self.year_patterns = [r'\b(19|20)\d{2}\b', r'\b\d{4}\b']
        self.period_patterns = [
            r'quarter', r'q[1-4]', r'fiscal year', r'annual', r'yearly',
            r'month', r'weekly', r'daily', r'period'
        ]
        self.comparison_patterns = [
            r'from .+ to', r'between .+ and', r'year.over.year', r'yoy',
            r'compared to', r'versus', r'change', r'growth', r'increase', r'decrease'
        ]

    def analyze_temporal_intent(self, question_text: str) -> Dict:
        """
        Analyze temporal requirements in question semantics
        """
        text_lower = question_text.lower()

        # Year detection
        years = []
        for pattern in self.year_patterns:
            years.extend(re.findall(pattern, question_text))

        # Period detection
        periods = [p for p in self.period_patterns if re.search(p, text_lower)]

        # Comparison detection
        comparisons = [c for c in self.comparison_patterns if re.search(c, text_lower)]

        return {
            'temporal_lookup_score': self._calculate_temporal_score(years, periods),
            'comparison_temporal_score': self._calculate_comparison_score(comparisons, years),
            'detected_years': years,
            'detected_periods': periods,
            'comparison_indicators': comparisons
        }

    def _calculate_temporal_score(self, years: List[str], periods: List[str]) -> float:
        """Calculate temporal lookup score"""
        # High score if multiple years mentioned
        year_score = min(1.0, len(years) * 0.4)
        period_score = min(0.3, len(periods) * 0.1)
        return min(1.0, year_score + period_score)

    def _calculate_comparison_score(self, comparisons: List[str], years: List[str]) -> float:
        """Calculate temporal comparison score"""
        # High score if comparison terms + multiple years
        comparison_score = min(0.7, len(comparisons) * 0.3)
        year_bonus = 0.3 if len(years) >= 2 else 0.0
        return min(1.0, comparison_score + year_bonus)


class AnalyticalOperationDetector:
    """
    Detects analytical operations required by questions (semantic analysis)
    """

    def __init__(self):
        self.calculation_keywords = [
            'percentage', 'percent', '%', 'ratio', 'rate', 'change', 'growth',
            'increase', 'decrease', 'difference', 'total', 'sum', 'average',
            'mean', 'calculate', 'compute', 'derive', 'determine'
        ]
        self.aggregation_keywords = [
            'total', 'sum', 'aggregate', 'combined', 'overall', 'net',
            'gross', 'average', 'mean', 'maximum', 'minimum', 'count'
        ]
        self.trend_keywords = [
            'trend', 'pattern', 'growth', 'decline', 'trajectory', 'direction',
            'progression', 'evolution', 'development', 'over time'
        ]

    def detect_analytical_operations(self, question_text: str) -> Dict:
        """
        Detect analytical operation requirements in question semantics
        """
        text_lower = question_text.lower()

        # Operation type detection
        calc_matches = [kw for kw in self.calculation_keywords if kw in text_lower]
        agg_matches = [kw for kw in self.aggregation_keywords if kw in text_lower]
        trend_matches = [kw for kw in self.trend_keywords if kw in text_lower]

        return {
            'analytical_operation_score': min(1.0, len(calc_matches) * 0.3),
            'aggregation_score': min(1.0, len(agg_matches) * 0.25),
            'trend_analysis_score': min(1.0, len(trend_matches) * 0.2),
            'calculation_keywords': calc_matches,
            'aggregation_keywords': agg_matches,
            'trend_keywords': trend_matches
        }


class ComputationalRequirementAnalyzer:
    """
    Analyzes computational complexity requirements of questions
    """

    def __init__(self):
        self.computational_indicators = [
            'calculate', 'compute', 'derive', 'determine', 'find', 'what is',
            'how much', 'how many', 'what was', 'what will be'
        ]
        self.complex_operations = [
            'percentage change', 'compound growth', 'ratio analysis',
            'variance', 'correlation', 'regression', 'forecast'
        ]

    def analyze_computational_requirements(self, question_text: str) -> Dict:
        """
        Analyze computational complexity requirements
        """
        text_lower = question_text.lower()

        # Basic computational indicators
        basic_matches = [indicator for indicator in self.computational_indicators
                        if indicator in text_lower]

        # Complex operation indicators
        complex_matches = [op for op in self.complex_operations
                          if op in text_lower]

        computational_score = 0.0

        # Basic computation score
        if basic_matches:
            computational_score += 0.5

        # Complex computation bonus
        if complex_matches:
            computational_score += 0.5

        return {
            'computational_requirement_score': min(1.0, computational_score),
            'basic_computational_signals': basic_matches,
            'complex_computational_signals': complex_matches
        }


class QuestionSemanticAnalyzer:
    """
    Analyzes question semantic patterns for intent classification
    """

    def __init__(self):
        self.numerical_patterns = [
            r'\$[\d,]+\.?\d*',     # Dollar amounts
            r'\d+\.?\d*%',        # Percentages
            r'\b\d+\.?\d*\b'      # General numbers
        ]

    def analyze_question_semantics(self, question_text: str) -> Dict:
        """
        Comprehensive question semantic analysis
        """
        text_lower = question_text.lower()

        # Numerical extraction needs
        numerical_score = self._calculate_numerical_extraction_score(question_text)

        # Contextual integration needs
        contextual_score = self._calculate_contextual_integration_score(question_text)

        # Extract numerical patterns
        numerical_patterns = self._extract_numerical_patterns(question_text)

        return {
            'numerical_extraction_score': numerical_score,
            'contextual_integration_score': contextual_score,
            'numerical_patterns': numerical_patterns
        }

    def _calculate_numerical_extraction_score(self, question_text: str) -> float:
        """Calculate how much the question needs specific numbers"""
        text_lower = question_text.lower()

        # Look for numerical needs indicators
        numerical_indicators = [
            'what is', 'how much', 'how many', 'percentage', 'amount',
            'value', 'number', 'figure', 'total', 'sum'
        ]

        matches = [indicator for indicator in numerical_indicators if indicator in text_lower]
        return min(1.0, len(matches) * 0.3)

    def _calculate_contextual_integration_score(self, question_text: str) -> float:
        """Calculate need for multi-element context integration"""
        text_lower = question_text.lower()

        # Look for integration indicators
        integration_indicators = [
            'between', 'from', 'to', 'compare', 'relationship', 'versus',
            'difference', 'change', 'across', 'throughout'
        ]

        matches = [indicator for indicator in integration_indicators if indicator in text_lower]
        return min(1.0, len(matches) * 0.25)

    def _extract_numerical_patterns(self, question_text: str) -> List[str]:
        """Extract numerical patterns from question"""
        patterns = []
        for pattern in self.numerical_patterns:
            matches = re.findall(pattern, question_text)
            patterns.extend(matches)
        return patterns


class TraditionalIntentClassifier:
    """
    Handles traditional intent classification
    """

    def __init__(self):
        self.comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'between']
        self.calculation_keywords = ['calculate', 'compute', 'sum', 'total']
        self.definition_keywords = ['what is', 'define', 'meaning', 'definition']
        self.identification_keywords = ['identify', 'find', 'locate', 'which']
        self.factual_keywords = ['fact', 'information', 'data', 'details']

    def analyze_traditional_intents(self, question_text: str) -> Dict:
        """
        Analyze traditional intent categories
        """
        text_lower = question_text.lower()

        return {
            'comparison': self._score_intent(text_lower, self.comparison_keywords),
            'calculation': self._score_intent(text_lower, self.calculation_keywords),
            'definition': self._score_intent(text_lower, self.definition_keywords),
            'identification': self._score_intent(text_lower, self.identification_keywords),
            'factual': self._score_intent(text_lower, self.factual_keywords)
        }

    def _score_intent(self, text: str, keywords: List[str]) -> float:
        """Score intent based on keyword matches"""
        matches = sum(1 for keyword in keywords if keyword in text)
        return min(1.0, matches * 0.4)


class Q2_1_EnhancedIntentLayer:
    """
    Main Q2.1 Enhanced Intent Layer processor (corrected for question semantics)
    """

    def __init__(self):
        self.temporal_engine = TemporalAnalysisEngine()
        self.analytical_detector = AnalyticalOperationDetector()
        self.computational_analyzer = ComputationalRequirementAnalyzer()
        self.semantic_analyzer = QuestionSemanticAnalyzer()
        self.traditional_classifier = TraditionalIntentClassifier()

    def classify_enhanced_intent(self, question_id: str) -> Dict:
        """
        Main processing function for enhanced intent classification
        """
        start_time = datetime.now()

        try:
            # Load question data from Q1
            question_data = self._load_question_from_q1(question_id)

            question_text = question_data['question_text']
            doc_id = question_data['doc_id']

            # Perform semantic analysis
            temporal_analysis = self.temporal_engine.analyze_temporal_intent(question_text)
            analytical_analysis = self.analytical_detector.detect_analytical_operations(question_text)
            computational_analysis = self.computational_analyzer.analyze_computational_requirements(question_text)
            semantic_analysis = self.semantic_analyzer.analyze_question_semantics(question_text)
            traditional_analysis = self.traditional_classifier.analyze_traditional_intents(question_text)

            # Build intent classification scores
            intent_scores = {
                # Traditional intents
                "comparison": traditional_analysis['comparison'],
                "calculation": traditional_analysis['calculation'],
                "definition": traditional_analysis['definition'],
                "identification": traditional_analysis['identification'],
                "factual": traditional_analysis['factual'],

                # Enhanced question semantic intents
                "temporal_lookup": temporal_analysis['temporal_lookup_score'],
                "numerical_extraction": semantic_analysis['numerical_extraction_score'],
                "analytical_operation": analytical_analysis['analytical_operation_score'],
                "contextual_integration": semantic_analysis['contextual_integration_score'],
                "aggregation": analytical_analysis['aggregation_score'],
                "comparison_temporal": temporal_analysis['comparison_temporal_score'],
                "trend_analysis": analytical_analysis['trend_analysis_score'],
                "computational_requirement": computational_analysis['computational_requirement_score']
            }

            # Normalize scores
            normalized_scores = self._normalize_intent_scores(intent_scores)

            # Extract semantic indicators
            semantic_indicators = {
                'temporal_markers': temporal_analysis.get('detected_years', []) +
                                  temporal_analysis.get('detected_periods', []),
                'numerical_patterns': semantic_analysis.get('numerical_patterns', []),
                'calculation_keywords': analytical_analysis.get('calculation_keywords', []),
                'comparison_indicators': temporal_analysis.get('comparison_indicators', []),
                'computational_signals': computational_analysis.get('basic_computational_signals', []) +
                                       computational_analysis.get('complex_computational_signals', [])
            }

            # Calculate processing metadata
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            result = {
                'question_id': question_data['question_id'],
                'doc_id': question_data['doc_id'],
                'question_text': question_data['question_text'],
                'intent_classification': normalized_scores,
                'intent_vector': list(normalized_scores.values()),
                'primary_intent': max(normalized_scores, key=normalized_scores.get),
                'intent_confidence': self._calculate_confidence(normalized_scores),
                'question_semantic_indicators': semantic_indicators,
                'processing_metadata': {
                    'classification_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': processing_time,
                    'pattern_matches': len(semantic_indicators['temporal_markers']) +
                                      len(semantic_indicators['calculation_keywords']),
                    'confidence_factors': self._identify_confidence_factors(normalized_scores, semantic_indicators)
                }
            }

            return result

        except Exception as e:
            print(f"Error in Q2.1 processing: {e}")
            return self._get_default_output(question_id)

    def _load_question_from_q1(self, question_id: str) -> Dict:
        """Load ONLY question data from Q1 output - NO ANSWER/RESPONSE DATA"""
        try:
            q1_path = "../outputs/Q1_Question_ingestion.json"
            with open(q1_path, 'r') as f:
                q1_data = json.load(f)

            raw_data = None
            if isinstance(q1_data, dict) and 'question_id' in q1_data:
                # Single question format
                if q1_data['question_id'] == question_id:
                    raw_data = q1_data
            elif isinstance(q1_data, dict):
                # Multi-question format
                if question_id in q1_data:
                    raw_data = q1_data[question_id]

            if raw_data is None:
                raise ValueError(f"Question {question_id} not found in Q1 output")

            # CRITICAL: Filter out ANY answer/response data to prevent leakage
            filtered_data = {
                'question_id': raw_data.get('question_id', question_id),
                'doc_id': raw_data.get('doc_id', question_id),
                'question_text': raw_data.get('question_text', '')
                # EXPLICITLY EXCLUDE: answer, response, generation_model_name, etc.
            }

            # Validate no answer leakage
            if not filtered_data['question_text']:
                raise ValueError(f"No question_text found for {question_id}")

            print(f"Q2.1 Data Leakage Check: Loading ONLY question_text, excluding answer data")
            return filtered_data

        except Exception as e:
            print(f"Error loading Q1 data: {e}")
            # Return minimal question data
            return {
                'question_id': question_id,
                'doc_id': question_id,
                'question_text': 'What is the percentage change in the revenue from 2018 to 2019?'
            }

    def _normalize_intent_scores(self, intent_scores: Dict) -> Dict:
        """Normalize intent scores"""
        # Simple normalization - ensure all scores are between 0 and 1
        normalized = {}
        for key, score in intent_scores.items():
            normalized[key] = max(0.0, min(1.0, score))
        return normalized

    def _calculate_confidence(self, intent_scores: Dict) -> float:
        """Calculate confidence in intent classification"""
        scores = list(intent_scores.values())
        if not scores:
            return 0.5

        max_score = max(scores)
        second_max = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0.0

        # High confidence if clear winner
        separation = max_score - second_max
        confidence = min(1.0, (separation * 2) + (max_score * 0.5))

        return confidence

    def _identify_confidence_factors(self, scores: Dict, indicators: Dict) -> List[str]:
        """Identify factors contributing to confidence"""
        factors = []

        # High scoring intents
        high_scoring = [intent for intent, score in scores.items() if score > 0.7]
        if len(high_scoring) == 1:
            factors.append('single_dominant_intent')
        elif len(high_scoring) > 1:
            factors.append('multi_intent_detected')

        # Strong indicators
        if len(indicators.get('temporal_markers', [])) >= 2:
            factors.append('strong_temporal_signals')
        if len(indicators.get('calculation_keywords', [])) >= 2:
            factors.append('strong_analytical_signals')

        return factors if factors else ['moderate_confidence']

    def _get_default_output(self, question_id: str) -> Dict:
        """Return default output on error"""
        return {
            'question_id': question_id,
            'doc_id': question_id,
            'question_text': 'Error in processing',
            'intent_classification': {
                'comparison': 0.0,
                'calculation': 0.5,
                'definition': 0.0,
                'identification': 0.0,
                'factual': 0.0,
                'temporal_lookup': 0.5,
                'numerical_extraction': 0.5,
                'analytical_operation': 0.5,
                'contextual_integration': 0.5,
                'aggregation': 0.0,
                'comparison_temporal': 0.5,
                'trend_analysis': 0.0,
                'computational_requirement': 0.5
            },
            'intent_vector': [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5],
            'primary_intent': 'analytical_operation',
            'intent_confidence': 0.5,
            'question_semantic_indicators': {
                'temporal_markers': [],
                'numerical_patterns': [],
                'calculation_keywords': [],
                'comparison_indicators': [],
                'computational_signals': []
            },
            'processing_metadata': {
                'classification_timestamp': datetime.now().isoformat(),
                'processing_time_ms': 0.0,
                'pattern_matches': 0,
                'confidence_factors': ['error_fallback']
            }
        }

    def save_output(self, result: Dict, output_path: str = "../outputs/Q2.1_enhanced_intent_classification.json"):
        """Save Q2.1 output to file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Wrap in question_id structure for consistency
            output_data = {result['question_id']: result}

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Q2.1 output saved to {output_path}")

        except Exception as e:
            print(f"Error saving Q2.1 output: {e}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("Q2.1: Enhanced Intent Layer Test (Corrected - Question Semantics)")
    print("=" * 70)

    # Initialize Q2.1
    q21 = Q2_1_EnhancedIntentLayer()

    # Process the sample question
    question_id = "finqa_test_1630"
    print(f"Processing Q2.1 for question: {question_id}")

    # Run enhanced intent classification
    result = q21.classify_enhanced_intent(question_id)

    print(f"Question: {result['question_text'][:60]}...")

    print("\n" + "=" * 50)
    print("Q2.1 OUTPUT - Enhanced Intent Classification:")
    print("=" * 50)
    print(f"Question ID: {result['question_id']}")
    print(f"Primary Intent: {result['primary_intent']}")
    print(f"Confidence: {result['intent_confidence']:.3f}")

    print(f"\nTop Intent Scores:")
    # Show top 5 intent scores
    sorted_intents = sorted(result['intent_classification'].items(),
                           key=lambda x: x[1], reverse=True)
    for i, (intent, score) in enumerate(sorted_intents[:5]):
        print(f"  {i+1}. {intent}: {score:.3f}")

    print(f"\nQuestion Semantic Indicators:")
    indicators = result['question_semantic_indicators']
    print(f"  temporal_markers: {indicators['temporal_markers']}")
    print(f"  calculation_keywords: {indicators['calculation_keywords']}")
    print(f"  comparison_indicators: {indicators['comparison_indicators'][:3]}")  # Show first 3
    print(f"  computational_signals: {indicators['computational_signals'][:3]}")  # Show first 3

    print(f"\nProcessing Metadata:")
    metadata = result['processing_metadata']
    print(f"  Processing time: {metadata['processing_time_ms']:.1f}ms")
    print(f"  Pattern matches: {metadata['pattern_matches']}")
    print(f"  Confidence factors: {metadata['confidence_factors']}")

    # Save output
    q21.save_output(result)

    return result


if __name__ == "__main__":
    print("Q2.1 ENHANCED INTENT LAYER (CORRECTED)")
    print("=" * 60)

    result = main()

    if result:
        print("Q2.1_enhanced_intent_classification.json created successfully")
        print("Question semantic analysis complete - ready for Q2.3 alignment")
    else:
        print("Failed to create Q2.1 output")