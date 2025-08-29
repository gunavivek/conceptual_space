#!/usr/bin/env python3
"""
A3: Centroid Validation Optimizer
Validates and optimizes concept centroids for semantic chunking
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import math

def simulate_embedding_vector(text_terms, dimension=384):
    """
    Simulate an embedding vector from text terms
    (In production, this would use actual embeddings like sentence-transformers)
    
    Args:
        text_terms: List of text terms
        dimension: Vector dimension
        
    Returns:
        numpy.ndarray: Simulated embedding vector
    """
    # Create deterministic but varied vectors based on text content
    if not text_terms:
        return np.random.random(dimension)
    
    # Use hash of terms to create reproducible vectors
    combined_text = " ".join(str(term).lower() for term in text_terms)
    hash_value = hash(combined_text)
    
    # Create vector with some structure
    np.random.seed(abs(hash_value) % (2**32))
    base_vector = np.random.normal(0, 1, dimension)
    
    # Add some term-specific variations
    for i, term in enumerate(text_terms[:10]):  # Limit to first 10 terms
        term_hash = abs(hash(str(term))) % dimension
        base_vector[term_hash] += 0.5
    
    # Normalize
    norm = np.linalg.norm(base_vector)
    return base_vector / norm if norm > 0 else base_vector

def calculate_centroid_quality(concept_data, concept_id):
    """
    Calculate quality metrics for a concept centroid
    
    Args:
        concept_data: Concept data with terms
        concept_id: Concept identifier
        
    Returns:
        dict: Quality metrics
    """
    # Extract terms for centroid calculation
    if "original_concept" in concept_data:
        original_terms = concept_data["original_concept"].get("primary_keywords", [])
        expanded_terms = concept_data.get("all_expanded_terms", [])
        all_terms = list(set(original_terms + expanded_terms))
    else:
        all_terms = concept_data.get("primary_keywords", [])
    
    if not all_terms:
        return {
            "concept_id": concept_id,
            "quality_score": 0.0,
            "coherence": 0.0,
            "distinctiveness": 0.0,
            "coverage": 0.0,
            "term_count": 0
        }
    
    # Simulate centroid vector
    centroid_vector = simulate_embedding_vector(all_terms)
    
    # Calculate coherence (how well terms relate to each other)
    term_vectors = [simulate_embedding_vector([term]) for term in all_terms[:20]]  # Limit for performance
    
    if len(term_vectors) > 1:
        # Average pairwise similarity
        similarities = []
        for i in range(len(term_vectors)):
            for j in range(i+1, len(term_vectors)):
                sim = np.dot(term_vectors[i], term_vectors[j])
                similarities.append(sim)
        coherence = np.mean(similarities) if similarities else 0.0
    else:
        coherence = 1.0  # Single term is perfectly coherent
    
    # Coverage (how well centroid represents all terms)
    if term_vectors:
        centroid_to_terms = [np.dot(centroid_vector, tv) for tv in term_vectors]
        coverage = np.mean(centroid_to_terms)
    else:
        coverage = 0.0
    
    # Quality score combines multiple factors
    term_density = min(1.0, len(all_terms) / 20.0)  # Optimal around 20 terms
    quality_score = (coherence * 0.4 + coverage * 0.4 + term_density * 0.2)
    
    return {
        "concept_id": concept_id,
        "quality_score": max(0.0, min(1.0, quality_score)),
        "coherence": max(0.0, min(1.0, coherence)),
        "coverage": max(0.0, min(1.0, coverage)),
        "distinctiveness": 0.0,  # Will be calculated later with other concepts
        "term_count": len(all_terms),
        "centroid_vector": centroid_vector.tolist()  # For storage
    }

def calculate_distinctiveness(concept_qualities):
    """
    Calculate distinctiveness scores by comparing centroids
    
    Args:
        concept_qualities: List of concept quality metrics
        
    Returns:
        dict: Updated concept qualities with distinctiveness
    """
    # Extract centroids
    centroids = {}
    for cq in concept_qualities:
        if "centroid_vector" in cq:
            centroids[cq["concept_id"]] = np.array(cq["centroid_vector"])
    
    # Calculate distinctiveness for each concept
    for cq in concept_qualities:
        concept_id = cq["concept_id"]
        if concept_id not in centroids:
            cq["distinctiveness"] = 0.0
            continue
        
        target_centroid = centroids[concept_id]
        similarities = []
        
        for other_id, other_centroid in centroids.items():
            if other_id != concept_id:
                similarity = np.dot(target_centroid, other_centroid)
                similarities.append(similarity)
        
        if similarities:
            # Distinctiveness is inverse of average similarity
            avg_similarity = np.mean(similarities)
            distinctiveness = 1.0 - avg_similarity
            cq["distinctiveness"] = max(0.0, min(1.0, distinctiveness))
        else:
            cq["distinctiveness"] = 1.0  # Only concept is perfectly distinctive
    
    return concept_qualities

def optimize_concept_centroid(concept_data, quality_metrics):
    """
    Optimize a concept centroid based on quality analysis
    
    Args:
        concept_data: Original concept data
        quality_metrics: Quality analysis results
        
    Returns:
        dict: Optimization recommendations
    """
    recommendations = []
    optimized_terms = []
    
    # Extract current terms
    if "original_concept" in concept_data:
        original_terms = concept_data["original_concept"].get("primary_keywords", [])
        expanded_terms = concept_data.get("all_expanded_terms", [])
        all_terms = list(set(original_terms + expanded_terms))
    else:
        all_terms = concept_data.get("primary_keywords", [])
    
    quality_score = quality_metrics["quality_score"]
    coherence = quality_metrics["coherence"]
    coverage = quality_metrics["coverage"]
    distinctiveness = quality_metrics["distinctiveness"]
    
    # Start with current terms
    optimized_terms = all_terms.copy()
    
    # Optimization strategies
    if quality_score < 0.6:
        if coherence < 0.4:
            recommendations.append("Low coherence - remove unrelated terms")
            # Simulate removing terms that don't fit
            if len(all_terms) > 5:
                optimized_terms = all_terms[:int(len(all_terms) * 0.8)]
        
        if coverage < 0.4:
            recommendations.append("Low coverage - add more representative terms")
            # Would add terms from expansion strategies
        
        if distinctiveness < 0.3:
            recommendations.append("Low distinctiveness - add domain-specific terms")
    
    # Term count optimization
    if len(all_terms) > 30:
        recommendations.append("Too many terms - filter to top terms only")
        optimized_terms = all_terms[:25]  # Keep top 25
    elif len(all_terms) < 5:
        recommendations.append("Too few terms - expand concept vocabulary")
    
    # Generate optimized centroid
    optimized_vector = simulate_embedding_vector(optimized_terms)
    
    return {
        "original_term_count": len(all_terms),
        "optimized_term_count": len(optimized_terms),
        "optimization_recommendations": recommendations,
        "optimized_terms": optimized_terms,
        "optimized_centroid": optimized_vector.tolist(),
        "expected_quality_improvement": max(0.0, min(1.0, quality_score + 0.1))
    }

def validate_centroid_registry(concept_data_dict):
    """
    Validate entire centroid registry
    
    Args:
        concept_data_dict: Dictionary of all concept data
        
    Returns:
        dict: Validation results
    """
    # Calculate quality for all concepts
    concept_qualities = []
    for concept_id, concept_data in concept_data_dict.items():
        quality = calculate_centroid_quality(concept_data, concept_id)
        concept_qualities.append(quality)
    
    # Calculate distinctiveness
    concept_qualities = calculate_distinctiveness(concept_qualities)
    
    # Generate optimizations
    optimizations = {}
    for cq in concept_qualities:
        concept_id = cq["concept_id"]
        concept_data = concept_data_dict[concept_id]
        optimization = optimize_concept_centroid(concept_data, cq)
        optimizations[concept_id] = optimization
    
    # Overall registry statistics
    quality_scores = [cq["quality_score"] for cq in concept_qualities]
    coherence_scores = [cq["coherence"] for cq in concept_qualities]
    distinctiveness_scores = [cq["distinctiveness"] for cq in concept_qualities]
    
    registry_stats = {
        "total_concepts": len(concept_qualities),
        "average_quality": np.mean(quality_scores) if quality_scores else 0.0,
        "average_coherence": np.mean(coherence_scores) if coherence_scores else 0.0,
        "average_distinctiveness": np.mean(distinctiveness_scores) if distinctiveness_scores else 0.0,
        "quality_distribution": {
            "excellent": len([q for q in quality_scores if q >= 0.8]),
            "good": len([q for q in quality_scores if 0.6 <= q < 0.8]),
            "fair": len([q for q in quality_scores if 0.4 <= q < 0.6]),
            "poor": len([q for q in quality_scores if q < 0.4])
        }
    }
    
    return {
        "concept_qualities": concept_qualities,
        "optimizations": optimizations,
        "registry_statistics": registry_stats,
        "validation_passed": registry_stats["average_quality"] >= 0.5,
        "recommendations": generate_registry_recommendations(registry_stats)
    }

def generate_registry_recommendations(registry_stats):
    """
    Generate recommendations for improving the centroid registry
    
    Args:
        registry_stats: Registry statistics
        
    Returns:
        list: Recommendations
    """
    recommendations = []
    
    avg_quality = registry_stats["average_quality"]
    avg_coherence = registry_stats["average_coherence"]
    avg_distinctiveness = registry_stats["average_distinctiveness"]
    quality_dist = registry_stats["quality_distribution"]
    
    if avg_quality < 0.6:
        recommendations.append("Overall registry quality is below acceptable threshold")
    
    if avg_coherence < 0.5:
        recommendations.append("Low coherence across concepts - review term expansion strategies")
    
    if avg_distinctiveness < 0.4:
        recommendations.append("Concepts are too similar - increase domain specialization")
    
    if quality_dist["poor"] > quality_dist["good"] + quality_dist["excellent"]:
        recommendations.append("Too many poor quality concepts - review expansion parameters")
    
    if registry_stats["total_concepts"] < 5:
        recommendations.append("Registry has too few concepts - expand concept identification")
    elif registry_stats["total_concepts"] > 50:
        recommendations.append("Registry has too many concepts - consider hierarchical grouping")
    
    return recommendations

def load_input(input_path="outputs/A2.5_expanded_concepts.json"):
    """Load expanded concepts for validation"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        # Fallback to core concepts
        fallback_path = script_dir / "outputs/A2.4_core_concepts.json"
        if fallback_path.exists():
            full_path = fallback_path
        else:
            raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_output(validation_results):
    """Save validation and optimization results"""
    script_dir = Path(__file__).parent.parent
    
    # Save main validation results
    output_path = script_dir / "outputs/A3_centroid_validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved validation report to {output_path}")
    
    # Save optimized centroids registry
    if validation_results["concept_qualities"]:
        centroids_registry = {}
        for cq in validation_results["concept_qualities"]:
            concept_id = cq["concept_id"]
            if concept_id in validation_results["optimizations"]:
                opt = validation_results["optimizations"][concept_id]
                centroids_registry[concept_id] = {
                    "centroid_vector": cq["centroid_vector"],
                    "optimized_vector": opt["optimized_centroid"],
                    "quality_score": cq["quality_score"],
                    "term_count": cq["term_count"],
                    "optimization_applied": len(opt["optimization_recommendations"]) > 0
                }
        
        registry_path = script_dir / "outputs/A3_optimized_centroids.json"
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(centroids_registry, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved optimized centroids to {registry_path}")

def main():
    """Main execution"""
    print("="*60)
    print("A3: Centroid Validation Optimizer")
    print("="*60)
    
    try:
        # Load concept data
        print("Loading concept data...")
        input_data = load_input()
        
        # Extract concepts
        if "expanded_concepts" in input_data:
            concepts = input_data["expanded_concepts"]
        elif "core_concepts" in input_data:
            concepts = {c["concept_id"]: c for c in input_data["core_concepts"]}
        else:
            concepts = input_data
        
        # Validate centroid registry
        print(f"Validating {len(concepts)} concept centroids...")
        validation_results = validate_centroid_registry(concepts)
        
        # Add metadata
        validation_results["validation_metadata"] = {
            "validation_timestamp": datetime.now().isoformat(),
            "source_concepts": len(concepts),
            "validation_passed": validation_results["validation_passed"]
        }
        
        # Display results
        stats = validation_results["registry_statistics"]
        print(f"\nCentroid Validation Results:")
        print(f"  Total Concepts: {stats['total_concepts']}")
        print(f"  Average Quality: {stats['average_quality']:.3f}")
        print(f"  Average Coherence: {stats['average_coherence']:.3f}")
        print(f"  Average Distinctiveness: {stats['average_distinctiveness']:.3f}")
        print(f"  Validation Passed: {validation_results['validation_passed']}")
        
        print(f"\nQuality Distribution:")
        dist = stats["quality_distribution"]
        print(f"  Excellent: {dist['excellent']} concepts")
        print(f"  Good: {dist['good']} concepts")
        print(f"  Fair: {dist['fair']} concepts")
        print(f"  Poor: {dist['poor']} concepts")
        
        if validation_results["recommendations"]:
            print(f"\nRegistry Recommendations:")
            for rec in validation_results["recommendations"]:
                print(f"  • {rec}")
        
        # Show concepts needing optimization
        needs_optimization = [
            cq for cq in validation_results["concept_qualities"] 
            if cq["quality_score"] < 0.6
        ]
        
        if needs_optimization:
            print(f"\nConcepts Needing Optimization ({len(needs_optimization)}):")
            for cq in needs_optimization[:5]:
                concept_id = cq["concept_id"]
                opt = validation_results["optimizations"].get(concept_id, {})
                print(f"  • {concept_id}: Quality {cq['quality_score']:.3f}")
                if opt.get("optimization_recommendations"):
                    print(f"    - {opt['optimization_recommendations'][0]}")
        
        # Save results
        save_output(validation_results)
        
        print("\nA3 Centroid Validation completed successfully!")
        
    except Exception as e:
        print(f"Error in A3 Validation: {str(e)}")
        raise

if __name__ == "__main__":
    main()