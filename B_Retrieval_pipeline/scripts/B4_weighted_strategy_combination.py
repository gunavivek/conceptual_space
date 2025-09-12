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
    Load results from the three matching strategies from actual B3 output files
    
    Returns:
        dict: Combined matching results with chunk data
    """
    script_dir = Path(__file__).parent.parent
    outputs_dir = script_dir / "outputs"
    
    matching_results = {}
    chunk_content_map = {}
    
    # Load B3.1 Intent Matching results
    b31_path = outputs_dir / "B3.1_intent_matching_output.json"
    if b31_path.exists():
        with open(b31_path, 'r', encoding='utf-8') as f:
            b31_data = json.load(f)
            
        # Handle array structure (new format)
        if isinstance(b31_data, list) and b31_data:
            first_result = b31_data[0]
            intent_scores = {}
            
            for chunk in first_result.get("ranked_chunks", []):
                chunk_id = chunk.get("chunk_id", "")
                score = chunk.get("similarity_score", 0.0)
                intent_scores[chunk_id] = score
                chunk_content_map[chunk_id] = chunk.get("content", "")
                
            matching_results["intent_based"] = intent_scores
    else:
        print(f"Warning: B3.1 output not found at {b31_path}")
        matching_results["intent_based"] = {}
    
    # Load B3.2 Declarative Matching results
    b32_path = outputs_dir / "B3.2_declarative_matching_output.json"
    if b32_path.exists():
        with open(b32_path, 'r', encoding='utf-8') as f:
            b32_data = json.load(f)
            
        # Handle array structure (new format)
        if isinstance(b32_data, list) and b32_data:
            first_result = b32_data[0]
            declarative_scores = {}
            
            # B3.2 has "matches" instead of "ranked_chunks"
            for match in first_result.get("matches", []):
                concept_id = match.get("concept_id", "")
                score = match.get("final_score", 0.0)
                declarative_scores[concept_id] = score
                # B3.2 matches don't have content, use concept name as content
                concept_data = match.get("concept", {})
                chunk_content_map[concept_id] = concept_data.get("canonical_name", "")
                
            matching_results["declarative_form"] = declarative_scores
    else:
        print(f"Warning: B3.2 output not found at {b32_path}")
        matching_results["declarative_form"] = {}
    
    # Load B3.3 Answer Backward Matching results
    b33_path = outputs_dir / "B3.3_answer_backward_matching_output.json"
    if b33_path.exists():
        with open(b33_path, 'r', encoding='utf-8') as f:
            b33_data = json.load(f)
            
        # Handle array structure (new format)
        if isinstance(b33_data, list) and b33_data:
            first_result = b33_data[0]
            answer_scores = {}
            
            for chunk in first_result.get("ranked_chunks", []):
                chunk_id = chunk.get("chunk_id", "")
                score = chunk.get("similarity_score", 0.0)
                answer_scores[chunk_id] = score
                chunk_content_map[chunk_id] = chunk.get("content", "")
                
            matching_results["answer_backwards"] = answer_scores
    else:
        print(f"Warning: B3.3 output not found at {b33_path}")
        matching_results["answer_backwards"] = {}
    
    # Store the content map for later use
    matching_results["_chunk_content_map"] = chunk_content_map
    
    return matching_results

def calculate_weighted_scores(matching_results, weights=None, track_contributions=True):
    """
    Calculate weighted combination of matching strategies
    
    Args:
        matching_results: Results from different strategies
        weights: Weight for each strategy
        track_contributions: Whether to track individual strategy contributions
        
    Returns:
        dict: Combined weighted scores with optional contribution tracking
    """
    if weights is None:
        # Default weights based on snapshot (Intent: 53.8%, Declarative: 36.2%, Backwards: 10%)
        weights = {
            "intent_based": 0.538,
            "declarative_form": 0.362,
            "answer_backwards": 0.100
        }
    
    # Extract chunk content map if present
    chunk_content_map = matching_results.pop("_chunk_content_map", {})
    
    # Collect all chunk IDs
    all_chunks = set()
    for strategy, strategy_results in matching_results.items():
        if isinstance(strategy_results, dict):
            all_chunks.update(strategy_results.keys())
    
    # Calculate weighted scores and track contributions
    combined_scores = {}
    contributions = {}
    
    for chunk_id in all_chunks:
        score = 0
        chunk_contributions = {}
        
        for strategy, strategy_results in matching_results.items():
            if chunk_id in strategy_results:
                strategy_score = strategy_results[chunk_id]
                weighted_contribution = strategy_score * weights.get(strategy, 0)
                score += weighted_contribution
                
                if track_contributions:
                    chunk_contributions[strategy] = {
                        "raw_score": strategy_score,
                        "weight": weights.get(strategy, 0),
                        "weighted_contribution": weighted_contribution
                    }
        
        combined_scores[chunk_id] = score
        if track_contributions:
            contributions[chunk_id] = chunk_contributions
    
    # Add back the content map
    result = {
        "scores": combined_scores,
        "chunk_content_map": chunk_content_map
    }
    
    if track_contributions:
        result["contributions"] = contributions
    
    return result

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
        dict: Combined matching results with detailed tracking
    """
    # Load matching results from three strategies
    matching_results = load_matching_results()
    
    # Calculate weighted combination with contribution tracking
    weights = {
        "intent_based": 0.538,
        "declarative_form": 0.362,
        "answer_backwards": 0.100
    }
    
    weighted_result = calculate_weighted_scores(matching_results, weights, track_contributions=True)
    combined_scores = weighted_result["scores"]
    chunk_content_map = weighted_result["chunk_content_map"]
    contributions = weighted_result["contributions"]
    
    # Rank chunks
    top_chunks = rank_concepts(combined_scores, top_k=10)
    
    # Calculate confidence
    confidence = calculate_confidence(top_chunks)
    
    # Prepare detailed chunk information
    ranked_chunks = []
    for chunk_id, score in top_chunks:
        chunk_info = {
            "chunk_id": chunk_id,
            "content": chunk_content_map.get(chunk_id, ""),
            "combined_score": score,
            "strategy_contributions": contributions.get(chunk_id, {}),
            "strategies_participated": len([s for s in contributions.get(chunk_id, {}) if contributions[chunk_id][s]["raw_score"] > 0])
        }
        ranked_chunks.append(chunk_info)
    
    # Calculate strategy agreement metrics
    strategy_stats = {
        "total_chunks_from_intent": len(matching_results.get("intent_based", {})),
        "total_chunks_from_declarative": len(matching_results.get("declarative_form", {})),
        "total_chunks_from_answer": len(matching_results.get("answer_backwards", {})),
        "total_unique_chunks": len(combined_scores),
        "chunks_in_all_strategies": len([c for c in combined_scores if all(
            c in matching_results.get(s, {}) for s in ["intent_based", "declarative_form", "answer_backwards"]
        )])
    }
    
    return {
        "question_id": question_data.get("question_id"),
        "question": question_data.get("question"),
        "weights": weights,
        "ranked_chunks": ranked_chunks,
        "confidence": confidence,
        "strategy_statistics": strategy_stats,
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/B4_weighted_combination_output.json"):
    """Save weighted combination results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Saved weighted combination to {full_path}")

def main():
    """Main execution for processing all 20 questions"""
    print("="*60)
    print("B4: Weighted Strategy Combination")
    print("="*60)
    
    try:
        # Load question data
        print("Loading question data...")
        questions_data = load_input()
        
        # Handle single question or array
        if not isinstance(questions_data, list):
            questions_data = [questions_data]
        
        all_results = []
        
        print(f"Processing {len(questions_data)} questions...\n")
        
        # Process each question
        for i, question_data in enumerate(questions_data):
            question_text = question_data.get("question", "")
            print(f"[{i+1:2d}/20] Combining strategies for: {question_text[:60]}...")
            
            # Process weighted combination for this question
            output_data = process_combination(question_data)
            all_results.append(output_data)
            
            # Brief results display
            confidence = output_data.get("confidence", 0)
            ranked_chunks = output_data.get("ranked_chunks", [])
            print(f"       Combined chunks: {len(ranked_chunks)} (confidence: {confidence:.3f})")
        
        # Save all results
        save_output(all_results)
        
        # Summary statistics
        print(f"\n{'='*60}")
        print("WEIGHTED STRATEGY COMBINATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total questions processed: {len(all_results)}")
        
        # Confidence statistics
        confidences = [result.get("confidence", 0) for result in all_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        print(f"Average combination confidence: {avg_confidence:.3f}")
        
        # Strategy weight statistics (from first result)
        if all_results:
            weights = all_results[0].get("weights", {})
            print(f"\nStrategy Weights Used:")
            for strategy, weight in weights.items():
                print(f"  {strategy}: {weight:.1%}")
        
        # Chunk statistics
        total_chunks = sum(len(result.get("ranked_chunks", [])) for result in all_results)
        avg_chunks = total_chunks / len(all_results) if all_results else 0
        
        print(f"\nCombined Chunk Statistics:")
        print(f"  Total combined chunks: {total_chunks}")
        print(f"  Average chunks per question: {avg_chunks:.1f}")
        
        print("\nB4 Weighted Strategy Combination completed successfully!")
        
    except Exception as e:
        print(f"Error in B4 Weighted Strategy Combination: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def combine_strategy_results(b3_results):
    """
    Interface function for orchestrator to combine B3 strategy results
    
    Args:
        b3_results: Dictionary containing results from B3.1, B3.2, B3.3
        
    Returns:
        dict: Combined weighted ranking results
    """
    # Extract strategy results and build content mapping
    matching_results = {}
    chunk_content_map = {}  # Store content for each chunk ID
    
    # Intent matching results - convert to expected dict format
    if "intent_matching" in b3_results and "ranked_chunks" in b3_results["intent_matching"]:
        matching_results["intent"] = {}
        for chunk in b3_results["intent_matching"]["ranked_chunks"]:
            concept_id = chunk.get("chunk_id", "")
            matching_results["intent"][concept_id] = chunk.get("similarity_score", 0.0)
            chunk_content_map[concept_id] = chunk.get("content", "")
    
    # Declarative matching results - convert to expected dict format
    if "declarative_matching" in b3_results and "ranked_chunks" in b3_results["declarative_matching"]:
        matching_results["declarative"] = {}
        for chunk in b3_results["declarative_matching"]["ranked_chunks"]:
            concept_id = chunk.get("chunk_id", "")
            matching_results["declarative"][concept_id] = chunk.get("similarity_score", 0.0)
            chunk_content_map[concept_id] = chunk.get("content", "")
    
    # Answer-backward matching results - convert to expected dict format
    if "answer_backward" in b3_results and "ranked_chunks" in b3_results["answer_backward"]:
        matching_results["answer_backward"] = {}
        for chunk in b3_results["answer_backward"]["ranked_chunks"]:
            concept_id = chunk.get("chunk_id", "")
            matching_results["answer_backward"][concept_id] = chunk.get("similarity_score", 0.0)
            chunk_content_map[concept_id] = chunk.get("content", "")
    
    # Calculate weighted scores using architecture-specified weights
    weights = {
        "intent": 0.538,        # 53.8%
        "declarative": 0.362,   # 36.2%  
        "answer_backward": 0.10  # 10%
    }
    
    combined_scores = calculate_weighted_scores(matching_results, weights)
    
    # Rank concepts
    ranked_concepts = rank_concepts(combined_scores, top_k=10)
    
    # Calculate confidence
    confidence = calculate_confidence(ranked_concepts)
    
    # Convert to chunk format for consistency
    ranked_chunks = []
    for concept_id, combined_score in ranked_concepts:
        content = chunk_content_map.get(concept_id, "")
        
        ranked_chunks.append({
            "chunk_id": concept_id,
            "content": content,
            "combined_score": combined_score,
            "strategy_contributions": {},  # TODO: Could be enhanced to show which strategies contributed
            "match_strategy": "weighted_combination"
        })
    
    return {
        "strategy": "weighted_combination", 
        "ranked_chunks": ranked_chunks,
        "total_strategies_used": len([k for k in matching_results.keys() if matching_results[k]]),
        "confidence": confidence,
        "weights_used": weights,
        "processing_timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    main()