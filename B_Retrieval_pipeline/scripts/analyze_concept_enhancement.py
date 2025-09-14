#!/usr/bin/env python3
"""
Analyze B3.3 Concept Enhancement Results
Extracts and summarizes concept integration impact across all 20 questions
"""

import json
from pathlib import Path

def analyze_concept_enhancement_results():
    """Analyze concept enhancement results from B3.3 output"""
    
    # Load B3.3 results
    script_dir = Path(__file__).parent.parent
    results_path = script_dir / "outputs" / "B3.3_answer_capability_assessment_output.json"
    
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("="*80)
    print("B3.3 CONCEPT ENHANCEMENT ANALYSIS - 20 QUESTIONS")
    print("="*80)
    
    # Overall statistics
    total_questions = len(results)
    total_chunks = sum(len(result.get("ranked_chunks", [])) for result in results)
    
    print(f"Total questions processed: {total_questions}")
    print(f"Total chunks assessed: {total_chunks}")
    
    # Concept enhancement metrics
    enhancement_stats = {
        "chunks_with_concepts": 0,
        "chunks_without_concepts": 0,
        "total_concept_boost": 0,
        "total_importance_multiplier": 0,
        "total_keyword_enhancement": 0,
        "base_score_sum": 0,
        "enhanced_score_sum": 0,
        "enhancement_improvements": []
    }
    
    question_details = []
    
    # Process each question
    for i, result in enumerate(results):
        question = result.get("question", "")
        question_id = result.get("question_id", f"q_{i}")
        ranked_chunks = result.get("ranked_chunks", [])
        
        if not ranked_chunks:
            continue
            
        # Get top chunk for analysis
        top_chunk = ranked_chunks[0]
        assessment = top_chunk.get("assessment_details", {})
        concept_enhancement = assessment.get("concept_enhancement", {})
        
        # Extract enhancement metrics
        concept_memberships = concept_enhancement.get("concept_memberships", [])
        concept_boost = concept_enhancement.get("concept_boost", 0)
        importance_multiplier = concept_enhancement.get("importance_multiplier", 1.0)
        keyword_enhancement = concept_enhancement.get("keyword_enhancement", 0)
        base_score = concept_enhancement.get("base_score", 0)
        enhanced_score = concept_enhancement.get("enhanced_score", 0)
        
        # Calculate improvement
        if base_score > 0:
            improvement_pct = ((enhanced_score - base_score) / base_score) * 100
        else:
            improvement_pct = 0
        
        # Update statistics
        if concept_memberships:
            enhancement_stats["chunks_with_concepts"] += 1
        else:
            enhancement_stats["chunks_without_concepts"] += 1
            
        enhancement_stats["total_concept_boost"] += concept_boost
        enhancement_stats["total_importance_multiplier"] += importance_multiplier
        enhancement_stats["total_keyword_enhancement"] += keyword_enhancement
        enhancement_stats["base_score_sum"] += base_score
        enhancement_stats["enhanced_score_sum"] += enhanced_score
        enhancement_stats["enhancement_improvements"].append(improvement_pct)
        
        # Store question details
        question_details.append({
            "question_num": i + 1,
            "question_id": question_id,
            "question": question[:60] + "..." if len(question) > 60 else question,
            "expected_type": assessment.get("expected_answer_type", "unknown"),
            "concept_count": len(concept_memberships),
            "concept_boost": concept_boost,
            "importance_multiplier": importance_multiplier,
            "keyword_enhancement": keyword_enhancement,
            "base_score": base_score,
            "enhanced_score": enhanced_score,
            "improvement_pct": improvement_pct,
            "top_concepts": concept_memberships[:3]  # Top 3 concepts
        })
    
    # Calculate averages
    total_with_concepts = enhancement_stats["chunks_with_concepts"]
    if total_with_concepts > 0:
        avg_concept_boost = enhancement_stats["total_concept_boost"] / total_with_concepts
        avg_importance_multiplier = enhancement_stats["total_importance_multiplier"] / total_with_concepts
        avg_keyword_enhancement = enhancement_stats["total_keyword_enhancement"] / total_with_concepts
        avg_base_score = enhancement_stats["base_score_sum"] / total_with_concepts
        avg_enhanced_score = enhancement_stats["enhanced_score_sum"] / total_with_concepts
        avg_improvement = sum(enhancement_stats["enhancement_improvements"]) / len(enhancement_stats["enhancement_improvements"])
    else:
        avg_concept_boost = avg_importance_multiplier = avg_keyword_enhancement = 0
        avg_base_score = avg_enhanced_score = avg_improvement = 0
    
    # Print summary statistics
    print(f"\nCONCEPT ENHANCEMENT SUMMARY:")
    print(f"Chunks with concept memberships: {enhancement_stats['chunks_with_concepts']}")
    print(f"Chunks without concepts: {enhancement_stats['chunks_without_concepts']}")
    print(f"Average concept boost: {avg_concept_boost:.4f}")
    print(f"Average importance multiplier: {avg_importance_multiplier:.4f}")
    print(f"Average keyword enhancement: {avg_keyword_enhancement:.4f}")
    print(f"Average base score: {avg_base_score:.4f}")
    print(f"Average enhanced score: {avg_enhanced_score:.4f}")
    print(f"Average improvement: {avg_improvement:.2f}%")
    
    # Print detailed question analysis
    print(f"\nDETAILED QUESTION ANALYSIS:")
    print(f"{'#':<3} {'Question':<50} {'Type':<8} {'Concepts':<8} {'Base':<6} {'Enhanced':<8} {'Improvement':<11}")
    print("-" * 100)
    
    for detail in question_details:
        print(f"{detail['question_num']:<3} "
              f"{detail['question']:<50} "
              f"{detail['expected_type']:<8} "
              f"{detail['concept_count']:<8} "
              f"{detail['base_score']:<6.3f} "
              f"{detail['enhanced_score']:<8.3f} "
              f"{detail['improvement_pct']:<11.1f}%")
    
    # Answer type analysis
    type_stats = {}
    for detail in question_details:
        answer_type = detail['expected_type']
        if answer_type not in type_stats:
            type_stats[answer_type] = {
                'count': 0,
                'total_improvement': 0,
                'total_base': 0,
                'total_enhanced': 0
            }
        type_stats[answer_type]['count'] += 1
        type_stats[answer_type]['total_improvement'] += detail['improvement_pct']
        type_stats[answer_type]['total_base'] += detail['base_score']
        type_stats[answer_type]['total_enhanced'] += detail['enhanced_score']
    
    print(f"\nANSWER TYPE PERFORMANCE:")
    print(f"{'Type':<10} {'Count':<6} {'Avg Base':<10} {'Avg Enhanced':<12} {'Avg Improvement':<15}")
    print("-" * 60)
    
    for answer_type, stats in type_stats.items():
        avg_base = stats['total_base'] / stats['count']
        avg_enhanced = stats['total_enhanced'] / stats['count']
        avg_improvement = stats['total_improvement'] / stats['count']
        
        print(f"{answer_type:<10} "
              f"{stats['count']:<6} "
              f"{avg_base:<10.3f} "
              f"{avg_enhanced:<12.3f} "
              f"{avg_improvement:<15.1f}%")
    
    print("\n" + "="*80)
    print("CONCEPT ENHANCEMENT ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    analyze_concept_enhancement_results()