#!/usr/bin/env python3
"""
B2.4: Temporal Analysis
Enhanced temporal question processing for time-based queries
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

def extract_temporal_entities(question: str) -> Dict[str, Any]:
    """
    Extract temporal entities and patterns from a question
    
    Args:
        question: Question text
        
    Returns:
        dict: Temporal entity analysis
    """
    question_lower = question.lower()
    
    # Temporal patterns
    temporal_patterns = {
        # Time indicators
        "when_indicators": ["when", "what time", "at what point"],
        "frequency_indicators": ["how often", "frequency", "annually", "monthly", "quarterly"],
        "duration_indicators": ["how long", "duration", "period", "span"],
        
        # Specific temporal terms
        "periods": {
            "annual": ["annual", "annually", "yearly", "year", "years"],
            "quarterly": ["quarterly", "quarter", "quarters"],
            "monthly": ["monthly", "month", "months"],
            "weekly": ["weekly", "week", "weeks"],
            "daily": ["daily", "day", "days"]
        },
        
        # Business temporal terms
        "business_periods": {
            "fiscal": ["fiscal year", "fiscal period", "fy"],
            "coverage": ["coverage period", "service period"],
            "agreement": ["agreement period", "contract period"],
            "billing": ["billing period", "invoice period", "payment period"]
        },
        
        # Temporal relationships
        "timing_relationships": {
            "before": ["before", "prior to", "ahead of"],
            "after": ["after", "following", "subsequent to"],
            "during": ["during", "throughout", "within"],
            "at": ["at the", "at", "on"]
        }
    }
    
    # Extract entities
    entities = {
        "temporal_type": None,
        "period_mentions": [],
        "business_periods": [],
        "timing_relationships": [],
        "frequency_indicators": [],
        "specific_times": [],
        "compound_temporal_terms": []
    }
    
    # Check for temporal type
    if any(indicator in question_lower for indicator in temporal_patterns["when_indicators"]):
        entities["temporal_type"] = "when"
    elif any(indicator in question_lower for indicator in temporal_patterns["frequency_indicators"]):
        entities["temporal_type"] = "frequency"
    elif any(indicator in question_lower for indicator in temporal_patterns["duration_indicators"]):
        entities["temporal_type"] = "duration"
    
    # Extract period mentions
    for period_type, terms in temporal_patterns["periods"].items():
        for term in terms:
            if term in question_lower:
                entities["period_mentions"].append({
                    "term": term,
                    "type": period_type,
                    "position": question_lower.find(term)
                })
    
    # Extract business periods
    for business_type, terms in temporal_patterns["business_periods"].items():
        for term in terms:
            if term in question_lower:
                entities["business_periods"].append({
                    "term": term,
                    "type": business_type,
                    "position": question_lower.find(term)
                })
    
    # Extract timing relationships
    for relation_type, terms in temporal_patterns["timing_relationships"].items():
        for term in terms:
            if term in question_lower:
                entities["timing_relationships"].append({
                    "term": term,
                    "type": relation_type,
                    "position": question_lower.find(term)
                })
    
    # Extract compound temporal terms (like "multi-year agreement")
    compound_patterns = [
        r"multi-year\s+(?:agreement|contract|license|deal|plan)s?",
        r"long-term\s+(?:asset|account|liability|debt|contract|plan)s?",
        r"(?:annual|quarterly|monthly)\s+(?:billing|payment|invoice|coverage)",
        r"(?:beginning|end)\s+of\s+(?:period|year|quarter|month)",
        r"coverage\s+period",
        r"service\s+period",
        r"billing\s+cycle"
    ]
    
    for pattern in compound_patterns:
        matches = re.finditer(pattern, question_lower)
        for match in matches:
            entities["compound_temporal_terms"].append({
                "term": match.group(),
                "pattern": pattern,
                "position": match.start()
            })
    
    # Extract specific times/dates (years, dates, etc.)
    time_patterns = [
        r"\b(19|20)\d{2}\b",  # Years like 2019, 2020
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",  # Month day
        r"\bq[1-4]\b",  # Quarters Q1, Q2, etc.
    ]
    
    for pattern in time_patterns:
        matches = re.finditer(pattern, question_lower)
        for match in matches:
            entities["specific_times"].append({
                "term": match.group(),
                "pattern": pattern,
                "position": match.start()
            })
    
    return entities

def analyze_temporal_context(question: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze temporal context and relationships
    
    Args:
        question: Question text
        entities: Extracted temporal entities
        
    Returns:
        dict: Temporal context analysis
    """
    question_lower = question.lower()
    
    context_analysis = {
        "temporal_complexity": "simple",
        "requires_sequence": False,
        "involves_comparison": False,
        "primary_temporal_focus": None,
        "expected_answer_granularity": "period",  # period, date, frequency
        "temporal_scope": "specific",  # specific, range, ongoing
        "confidence": 0.5
    }
    
    # Determine complexity
    temporal_indicators = (len(entities["period_mentions"]) + 
                          len(entities["business_periods"]) + 
                          len(entities["timing_relationships"]) +
                          len(entities["compound_temporal_terms"]))
    
    if temporal_indicators >= 3:
        context_analysis["temporal_complexity"] = "complex"
        context_analysis["confidence"] = 0.8
    elif temporal_indicators >= 2:
        context_analysis["temporal_complexity"] = "moderate"
        context_analysis["confidence"] = 0.7
    
    # Check for sequence requirements
    sequence_words = ["before", "after", "then", "next", "following", "prior"]
    if any(word in question_lower for word in sequence_words):
        context_analysis["requires_sequence"] = True
    
    # Check for comparison
    comparison_words = ["difference", "change", "compare", "versus", "vs"]
    if any(word in question_lower for word in comparison_words):
        context_analysis["involves_comparison"] = True
    
    # Determine primary focus
    if entities["temporal_type"]:
        context_analysis["primary_temporal_focus"] = entities["temporal_type"]
    elif entities["compound_temporal_terms"]:
        context_analysis["primary_temporal_focus"] = "business_temporal"
    elif entities["business_periods"]:
        context_analysis["primary_temporal_focus"] = "business_period"
    
    # Determine expected answer granularity
    if any("annual" in term["term"] for term in entities["period_mentions"]):
        context_analysis["expected_answer_granularity"] = "annual"
    elif any("quarter" in term["term"] for term in entities["period_mentions"]):
        context_analysis["expected_answer_granularity"] = "quarterly"
    elif any("month" in term["term"] for term in entities["period_mentions"]):
        context_analysis["expected_answer_granularity"] = "monthly"
    elif entities["specific_times"]:
        context_analysis["expected_answer_granularity"] = "specific_date"
    
    return context_analysis

def generate_temporal_search_terms(question: str, entities: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
    """
    Generate enhanced search terms for temporal queries
    
    Args:
        question: Question text
        entities: Extracted temporal entities
        context: Temporal context analysis
        
    Returns:
        list: Enhanced search terms
    """
    search_terms = []
    
    # Add compound temporal terms directly
    for compound in entities["compound_temporal_terms"]:
        search_terms.append(compound["term"])
    
    # Add business period terms
    for period in entities["business_periods"]:
        search_terms.append(period["term"])
    
    # Add timing relationship terms
    for timing in entities["timing_relationships"]:
        search_terms.append(timing["term"])
    
    # Add specific temporal combinations
    if entities["temporal_type"] == "when":
        search_terms.extend([
            "timing of",
            "time when",
            "period when",
            "date when"
        ])
    elif entities["temporal_type"] == "frequency":
        search_terms.extend([
            "how often",
            "frequency of",
            "regularly",
            "periodically"
        ])
    
    # Add enhanced terms based on context
    if context["primary_temporal_focus"] == "business_temporal":
        search_terms.extend([
            "invoice customers",
            "billing period",
            "payment timing",
            "coverage period"
        ])
    
    return list(set(search_terms))  # Remove duplicates

def process_temporal_question(question: str) -> Dict[str, Any]:
    """
    Complete temporal analysis of a question
    
    Args:
        question: Question text
        
    Returns:
        dict: Complete temporal analysis
    """
    # Extract temporal entities
    entities = extract_temporal_entities(question)
    
    # Analyze temporal context
    context = analyze_temporal_context(question, entities)
    
    # Generate search terms
    search_terms = generate_temporal_search_terms(question, entities, context)
    
    # Compile results
    result = {
        "question": question,
        "temporal_entities": entities,
        "temporal_context": context,
        "enhanced_search_terms": search_terms,
        "processing_timestamp": datetime.now().isoformat(),
        "is_temporal_question": bool(entities["temporal_type"] or 
                                   entities["compound_temporal_terms"] or 
                                   entities["business_periods"]),
        "temporal_confidence": context["confidence"]
    }
    
    return result

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="B2.4 Temporal Analysis")
    parser.add_argument("--question", type=str, required=True, help="Question to analyze")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    print("="*80)
    print("B2.4: Temporal Analysis")
    print("="*80)
    
    # Process the question
    result = process_temporal_question(args.question)
    
    print(f"Question: {args.question}")
    print(f"Is Temporal: {result['is_temporal_question']}")
    print(f"Confidence: {result['temporal_confidence']:.2f}")
    print(f"Temporal Type: {result['temporal_entities']['temporal_type']}")
    print(f"Compound Terms: {len(result['temporal_entities']['compound_temporal_terms'])}")
    print(f"Enhanced Search Terms: {result['enhanced_search_terms']}")
    
    # Save output if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\\nTemporal analysis saved to: {output_path}")
    
    print("\\nB2.4 Temporal Analysis completed successfully!")

if __name__ == "__main__":
    main()