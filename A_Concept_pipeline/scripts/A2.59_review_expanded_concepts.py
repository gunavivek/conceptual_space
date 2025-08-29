#!/usr/bin/env python3
"""
A2.59: Review Expanded Concepts
Reviews and validates the quality of expanded concepts from A2.5
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

def analyze_concept_expansion_quality(concept_data):
    """
    Analyze the quality of a single concept's expansion
    
    Args:
        concept_data: Expanded concept data
        
    Returns:
        dict: Quality analysis
    """
    original_concept = concept_data.get("original_concept", {})
    all_expanded_terms = concept_data.get("all_expanded_terms", [])
    strategy_contributions = concept_data.get("strategy_contributions", {})
    expansion_scores = concept_data.get("expansion_scores", {})
    
    # Calculate expansion metrics
    original_terms = original_concept.get("primary_keywords", [])
    expansion_ratio = len(all_expanded_terms) / max(len(original_terms), 1)
    
    # Strategy diversity
    num_strategies = len(strategy_contributions)
    strategy_diversity_score = num_strategies / 5.0  # Assuming 5 total strategies
    
    # Term quality assessment
    common_stopwords = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "to", "for"}
    quality_terms = [term for term in all_expanded_terms if len(term) > 2 and term.lower() not in common_stopwords]
    term_quality_ratio = len(quality_terms) / max(len(all_expanded_terms), 1)
    
    # Coherence assessment (terms that appear in multiple strategies)
    term_frequency = Counter()
    for strategy_name, contrib in strategy_contributions.items():
        for term in contrib.get("terms", []):
            term_frequency[term] += 1
    
    coherent_terms = [term for term, freq in term_frequency.items() if freq > 1]
    coherence_ratio = len(coherent_terms) / max(len(all_expanded_terms), 1)
    
    # Overall quality score
    quality_components = {
        "expansion_appropriateness": min(1.0, expansion_ratio / 2.0),  # Good expansion is 1-2x original
        "strategy_diversity": strategy_diversity_score,
        "term_quality": term_quality_ratio,
        "coherence": coherence_ratio,
        "overall_quality": expansion_scores.get("overall_quality", 0.0)
    }
    
    overall_quality = (
        quality_components["expansion_appropriateness"] * 0.2 +
        quality_components["strategy_diversity"] * 0.2 +
        quality_components["term_quality"] * 0.2 +
        quality_components["coherence"] * 0.2 +
        quality_components["overall_quality"] * 0.2
    )
    
    # Quality rating
    if overall_quality >= 0.8:
        rating = "excellent"
    elif overall_quality >= 0.6:
        rating = "good"
    elif overall_quality >= 0.4:
        rating = "fair"
    else:
        rating = "poor"
    
    return {
        "concept_id": concept_data.get("concept_id"),
        "theme_name": original_concept.get("theme_name", "Unknown"),
        "quality_components": quality_components,
        "overall_quality_score": overall_quality,
        "quality_rating": rating,
        "expansion_ratio": expansion_ratio,
        "strategy_count": num_strategies,
        "coherent_terms_count": len(coherent_terms),
        "recommendations": generate_improvement_recommendations(concept_data, quality_components)
    }

def generate_improvement_recommendations(concept_data, quality_components):
    """
    Generate recommendations for improving concept expansion
    
    Args:
        concept_data: Expanded concept data
        quality_components: Quality component scores
        
    Returns:
        list: Improvement recommendations
    """
    recommendations = []
    
    if quality_components["expansion_appropriateness"] < 0.5:
        if concept_data.get("expansion_scores", {}).get("expansion_ratio", 0) < 1.0:
            recommendations.append("Insufficient expansion - consider running additional expansion strategies")
        else:
            recommendations.append("Over-expansion detected - filter terms more selectively")
    
    if quality_components["strategy_diversity"] < 0.6:
        recommendations.append("Low strategy diversity - ensure all expansion strategies are contributing")
    
    if quality_components["term_quality"] < 0.7:
        recommendations.append("Poor term quality - filter out stop words and very short terms")
    
    if quality_components["coherence"] < 0.3:
        recommendations.append("Low coherence - terms don't align across strategies, review expansion logic")
    
    # Strategy-specific recommendations
    strategy_contributions = concept_data.get("strategy_contributions", {})
    for strategy_name, contrib in strategy_contributions.items():
        if contrib.get("count", 0) == 0:
            recommendations.append(f"No contribution from {strategy_name} strategy - investigate why")
    
    return recommendations

def review_expansion_coverage(expanded_concepts):
    """
    Review overall expansion coverage and patterns
    
    Args:
        expanded_concepts: All expanded concepts
        
    Returns:
        dict: Coverage analysis
    """
    total_concepts = len(expanded_concepts)
    
    # Quality distribution
    quality_ratings = defaultdict(int)
    expansion_ratios = []
    strategy_usage = defaultdict(int)
    
    for concept_id, concept_data in expanded_concepts.items():
        # Analyze this concept
        quality_analysis = analyze_concept_expansion_quality(concept_data)
        quality_ratings[quality_analysis["quality_rating"]] += 1
        expansion_ratios.append(quality_analysis["expansion_ratio"])
        
        # Count strategy usage
        for strategy_name in concept_data.get("strategy_contributions", {}):
            strategy_usage[strategy_name] += 1
    
    # Calculate statistics
    avg_expansion_ratio = sum(expansion_ratios) / max(len(expansion_ratios), 1)
    
    # Strategy coverage
    strategy_coverage = {}
    for strategy, count in strategy_usage.items():
        strategy_coverage[strategy] = {
            "concept_count": count,
            "coverage_percentage": (count / total_concepts) * 100
        }
    
    return {
        "total_concepts": total_concepts,
        "quality_distribution": dict(quality_ratings),
        "quality_percentages": {rating: (count/total_concepts)*100 for rating, count in quality_ratings.items()},
        "average_expansion_ratio": avg_expansion_ratio,
        "strategy_coverage": strategy_coverage,
        "expansion_ratio_distribution": {
            "min": min(expansion_ratios) if expansion_ratios else 0,
            "max": max(expansion_ratios) if expansion_ratios else 0,
            "avg": avg_expansion_ratio
        }
    }

def identify_problematic_concepts(expanded_concepts, quality_threshold=0.4):
    """
    Identify concepts with expansion quality issues
    
    Args:
        expanded_concepts: All expanded concepts
        quality_threshold: Minimum quality threshold
        
    Returns:
        list: Problematic concepts with details
    """
    problematic_concepts = []
    
    for concept_id, concept_data in expanded_concepts.items():
        quality_analysis = analyze_concept_expansion_quality(concept_data)
        
        if quality_analysis["overall_quality_score"] < quality_threshold:
            problematic_concepts.append({
                "concept_id": concept_id,
                "theme_name": quality_analysis["theme_name"],
                "quality_score": quality_analysis["overall_quality_score"],
                "quality_rating": quality_analysis["quality_rating"],
                "main_issues": [rec for rec in quality_analysis["recommendations"]],
                "expansion_ratio": quality_analysis["expansion_ratio"],
                "strategy_count": quality_analysis["strategy_count"]
            })
    
    # Sort by quality score (worst first)
    problematic_concepts.sort(key=lambda x: x["quality_score"])
    
    return problematic_concepts

def generate_expansion_summary(expanded_concepts):
    """
    Generate comprehensive summary of expansion results
    
    Args:
        expanded_concepts: All expanded concepts
        
    Returns:
        dict: Expansion summary
    """
    # Individual concept analyses
    concept_analyses = []
    for concept_id, concept_data in expanded_concepts.items():
        analysis = analyze_concept_expansion_quality(concept_data)
        concept_analyses.append(analysis)
    
    # Coverage analysis
    coverage_analysis = review_expansion_coverage(expanded_concepts)
    
    # Problematic concepts
    problematic_concepts = identify_problematic_concepts(expanded_concepts)
    
    # Top performing concepts
    top_concepts = sorted(concept_analyses, key=lambda x: x["overall_quality_score"], reverse=True)[:5]
    
    return {
        "concept_analyses": concept_analyses,
        "coverage_analysis": coverage_analysis,
        "problematic_concepts": problematic_concepts,
        "top_performing_concepts": top_concepts,
        "summary_statistics": {
            "total_concepts_reviewed": len(concept_analyses),
            "average_quality": sum(c["overall_quality_score"] for c in concept_analyses) / max(len(concept_analyses), 1),
            "excellent_count": len([c for c in concept_analyses if c["quality_rating"] == "excellent"]),
            "good_count": len([c for c in concept_analyses if c["quality_rating"] == "good"]),
            "fair_count": len([c for c in concept_analyses if c["quality_rating"] == "fair"]),
            "poor_count": len([c for c in concept_analyses if c["quality_rating"] == "poor"])
        }
    }

def load_input(input_path="outputs/A2.5_expanded_concepts.json"):
    """Load expanded concepts from A2.5"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_output(review_data, output_path="outputs/A2.59_concept_review_report.json"):
    """Save concept review report"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved concept review report to {full_path}")
    
    # Save a summary text file for easy reading
    summary_path = full_path.with_suffix('.txt').with_name(full_path.stem + '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        stats = review_data["summary_statistics"]
        coverage = review_data["coverage_analysis"]
        
        f.write("EXPANDED CONCEPTS REVIEW SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Concepts Reviewed: {stats['total_concepts_reviewed']}\n")
        f.write(f"Average Quality Score: {stats['average_quality']:.3f}\n\n")
        
        f.write("Quality Distribution:\n")
        f.write(f"  Excellent: {stats['excellent_count']} concepts\n")
        f.write(f"  Good: {stats['good_count']} concepts\n")
        f.write(f"  Fair: {stats['fair_count']} concepts\n")
        f.write(f"  Poor: {stats['poor_count']} concepts\n\n")
        
        f.write(f"Average Expansion Ratio: {coverage['average_expansion_ratio']:.2f}\n\n")
        
        f.write("Strategy Coverage:\n")
        for strategy, info in coverage["strategy_coverage"].items():
            f.write(f"  {strategy}: {info['concept_count']} concepts ({info['coverage_percentage']:.1f}%)\n")
        
        if review_data["problematic_concepts"]:
            f.write(f"\nProblematic Concepts ({len(review_data['problematic_concepts'])}):\n")
            for prob in review_data["problematic_concepts"][:5]:
                f.write(f"  - {prob['theme_name']}: {prob['quality_score']:.3f} ({prob['quality_rating']})\n")

def main():
    """Main execution"""
    print("="*60)
    print("A2.59: Review Expanded Concepts")
    print("="*60)
    
    try:
        # Load expanded concepts
        print("Loading expanded concepts...")
        input_data = load_input()
        expanded_concepts = input_data.get("expanded_concepts", {})
        
        # Generate comprehensive review
        print(f"Reviewing {len(expanded_concepts)} expanded concepts...")
        review_summary = generate_expansion_summary(expanded_concepts)
        
        # Add metadata
        review_data = {
            "review_metadata": {
                "review_timestamp": datetime.now().isoformat(),
                "source_data": "A2.5_expanded_concepts.json",
                "total_concepts": len(expanded_concepts)
            },
            **review_summary
        }
        
        # Display results
        stats = review_data["summary_statistics"]
        coverage = review_data["coverage_analysis"]
        
        print(f"\nExpansion Review Results:")
        print(f"  Total Concepts: {stats['total_concepts_reviewed']}")
        print(f"  Average Quality: {stats['average_quality']:.3f}")
        print(f"  Average Expansion Ratio: {coverage['average_expansion_ratio']:.2f}")
        
        print(f"\nQuality Distribution:")
        print(f"  Excellent: {stats['excellent_count']} ({coverage['quality_percentages'].get('excellent', 0):.1f}%)")
        print(f"  Good: {stats['good_count']} ({coverage['quality_percentages'].get('good', 0):.1f}%)")
        print(f"  Fair: {stats['fair_count']} ({coverage['quality_percentages'].get('fair', 0):.1f}%)")
        print(f"  Poor: {stats['poor_count']} ({coverage['quality_percentages'].get('poor', 0):.1f}%)")
        
        if review_data["problematic_concepts"]:
            print(f"\nProblematic Concepts ({len(review_data['problematic_concepts'])}):")
            for i, prob in enumerate(review_data["problematic_concepts"][:3], 1):
                print(f"  {i}. {prob['theme_name']}")
                print(f"     Quality: {prob['quality_score']:.3f} ({prob['quality_rating']})")
                print(f"     Main Issue: {prob['main_issues'][0] if prob['main_issues'] else 'Unknown'}")
        
        print(f"\nTop Performing Concepts:")
        for i, top in enumerate(review_data["top_performing_concepts"][:3], 1):
            print(f"  {i}. {top['theme_name']}")
            print(f"     Quality: {top['overall_quality_score']:.3f} ({top['quality_rating']})")
        
        # Save results
        save_output(review_data)
        
        print("\nA2.59 Expanded Concepts Review completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.59 Review: {str(e)}")
        raise

if __name__ == "__main__":
    main()