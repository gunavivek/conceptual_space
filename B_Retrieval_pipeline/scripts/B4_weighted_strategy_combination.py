#!/usr/bin/env python3
"""
B4: Weighted Strategy Combination
Combines multiple matching strategies to identify best concepts for answering
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np

def load_matching_results():
    """
    Load results from the three matching strategies
    
    Returns:
        dict: Combined matching results
    """
    script_dir = Path(__file__).parent.parent
    
    # Try to load B3 outputs (would be created by B3.1, B3.2, B3.3)
    # For now, we'll simulate with mock data
    
    # Mock matching results from three strategies
    intent_matching = {
        "Financial Concepts": 0.85,
        "Revenue Analysis": 0.72,
        "Income Statement": 0.68,
        "Deferred Income": 0.95,
        "Time Period": 0.60
    }
    
    declarative_matching = {
        "Financial Concepts": 0.90,
        "Deferred Income": 0.88,
        "Revenue Recognition": 0.65,
        "Annual Report": 0.70,
        "Accounting Changes": 0.75
    }
    
    answer_backwards_matching = {
        "Financial Concepts": 0.78,
        "Deferred Income": 0.82,
        "Monetary Values": 0.88,
        "Year 2019": 0.85,
        "Million Units": 0.70
    }
    
    return {
        "intent_based": intent_matching,
        "declarative_form": declarative_matching,
        "answer_backwards": answer_backwards_matching
    }

def calculate_weighted_scores(matching_results, weights=None):
    """
    Calculate weighted combination of matching strategies
    
    Args:
        matching_results: Results from different strategies
        weights: Weight for each strategy
        
    Returns:
        dict: Combined weighted scores
    """
    if weights is None:
        # Default weights based on snapshot (Intent: 53.8%, Declarative: 36.2%, Backwards: 10%)
        weights = {
            "intent_based": 0.538,
            "declarative_form": 0.362,
            "answer_backwards": 0.100
        }
    
    # Collect all concepts
    all_concepts = set()
    for strategy_results in matching_results.values():
        all_concepts.update(strategy_results.keys())
    
    # Calculate weighted scores
    combined_scores = {}
    
    for concept in all_concepts:
        score = 0
        for strategy, strategy_results in matching_results.items():
            if concept in strategy_results:
                score += strategy_results[concept] * weights[strategy]
        combined_scores[concept] = score
    
    return combined_scores

def rank_concepts(combined_scores, top_k=5):
    """
    Rank concepts by combined score
    
    Args:
        combined_scores: Combined weighted scores
        top_k: Number of top concepts to return
        
    Returns:
        list: Top ranked concepts with scores
    """
    sorted_concepts = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_concepts[:top_k]

def calculate_confidence(top_concepts):
    """
    Calculate confidence score based on top concepts
    
    Args:
        top_concepts: List of top concepts with scores
        
    Returns:
        float: Confidence score
    """
    if not top_concepts:
        return 0.0
    
    # Confidence based on top score and score distribution
    top_score = top_concepts[0][1]
    
    if len(top_concepts) > 1:
        second_score = top_concepts[1][1]
        # Higher confidence if there's clear separation
        separation = top_score - second_score
        confidence = min(1.0, top_score * (1 + separation))
    else:
        confidence = top_score
    
    return min(1.0, confidence)

def load_input(input_path="outputs/B2_1_intent_layer_output.json"):
    """Load processed question data"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        # Try B1 output as fallback
        alt_path = script_dir / "outputs/B1_current_question.json"
        if alt_path.exists():
            full_path = alt_path
        else:
            raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_combination(question_data):
    """
    Process weighted combination of strategies
    
    Args:
        question_data: Question data with intent analysis
        
    Returns:
        dict: Combined matching results
    """
    # Load matching results from three strategies
    matching_results = load_matching_results()
    
    # Calculate weighted combination
    combined_scores = calculate_weighted_scores(matching_results)
    
    # Rank concepts
    top_concepts = rank_concepts(combined_scores, top_k=5)
    
    # Calculate confidence
    confidence = calculate_confidence(top_concepts)
    
    # Prepare detailed breakdown
    strategy_contributions = {}
    for concept, _ in top_concepts:
        contributions = {}
        for strategy, results in matching_results.items():
            if concept in results:
                contributions[strategy] = results[concept]
        strategy_contributions[concept] = contributions
    
    return {
        "question_id": question_data.get("question_id"),
        "question": question_data.get("question"),
        "matching_results": matching_results,
        "weights": {
            "intent_based": 0.538,
            "declarative_form": 0.362,
            "answer_backwards": 0.100
        },
        "combined_scores": combined_scores,
        "top_concepts": [
            {
                "concept": concept,
                "score": score,
                "contributions": strategy_contributions.get(concept, {})
            }
            for concept, score in top_concepts
        ],
        "confidence": confidence,
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/B4_weighted_combination_output.json"):
    """Save weighted combination results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved weighted combination to {full_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B4: Weighted Strategy Combination")
    print("="*60)
    
    try:
        # Load question data
        print("Loading question data...")
        question_data = load_input()
        
        # Process weighted combination
        print("Combining matching strategies...")
        output_data = process_combination(question_data)
        
        # Display results
        print(f"\nStrategy Weights:")
        for strategy, weight in output_data["weights"].items():
            print(f"  {strategy}: {weight:.1%}")
        
        print(f"\nTop Concepts:")
        for i, concept_data in enumerate(output_data["top_concepts"], 1):
            print(f"\n  {i}. {concept_data['concept']}")
            print(f"     Final Score: {concept_data['score']:.3f}")
            print(f"     Contributions:")
            for strategy, score in concept_data['contributions'].items():
                print(f"       - {strategy}: {score:.3f}")
        
        print(f"\nOverall Confidence: {output_data['confidence']:.3f}")
        
        # Save output
        save_output(output_data)
        
        print("\nB4 Weighted Strategy Combination completed successfully!")
        
    except Exception as e:
        print(f"Error in B4 Weighted Strategy Combination: {str(e)}")
        raise

if __name__ == "__main__":
    main()