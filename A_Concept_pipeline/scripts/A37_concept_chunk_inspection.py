#!/usr/bin/env python3
"""
A37: Concept Chunk Inspection
Inspects and analyzes concept-based chunks for quality and semantic alignment
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import re

def analyze_chunk_concept_alignment(chunk_data, concept_registry):
    """
    Analyze how well a chunk aligns with its assigned concepts
    
    Args:
        chunk_data: Individual chunk data
        concept_registry: Registry of concept centroids
        
    Returns:
        dict: Alignment analysis
    """
    chunk_text = chunk_data.get("text", "")
    assigned_concepts = chunk_data.get("assigned_concepts", [])
    chunk_keywords = extract_keywords_from_text(chunk_text)
    
    if not assigned_concepts:
        return {
            "alignment_score": 0.0,
            "concept_coverage": 0.0,
            "keyword_overlap": 0,
            "analysis": "No concepts assigned"
        }
    
    alignment_scores = []
    total_keyword_overlap = 0
    
    for concept_id in assigned_concepts:
        if concept_id not in concept_registry:
            continue
            
        concept_data = concept_registry[concept_id]
        
        # Get concept terms
        if "optimized_terms" in concept_data:
            concept_terms = concept_data["optimized_terms"]
        elif "original_concept" in concept_data:
            concept_terms = concept_data["original_concept"].get("primary_keywords", [])
        else:
            concept_terms = concept_data.get("primary_keywords", [])
        
        # Calculate keyword overlap
        chunk_kw_set = set(kw.lower() for kw in chunk_keywords)
        concept_kw_set = set(kw.lower() for kw in concept_terms)
        
        overlap = len(chunk_kw_set & concept_kw_set)
        total_keyword_overlap += overlap
        
        # Calculate alignment score
        if chunk_kw_set and concept_kw_set:
            jaccard = overlap / len(chunk_kw_set | concept_kw_set)
            alignment_scores.append(jaccard)
        else:
            alignment_scores.append(0.0)
    
    # Calculate overall metrics
    avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
    concept_coverage = len([s for s in alignment_scores if s > 0.1]) / max(len(assigned_concepts), 1)
    
    return {
        "alignment_score": avg_alignment,
        "concept_coverage": concept_coverage,
        "keyword_overlap": total_keyword_overlap,
        "individual_alignments": dict(zip(assigned_concepts, alignment_scores)),
        "chunk_keywords": chunk_keywords[:10],  # Top 10 for inspection
        "analysis": determine_alignment_quality(avg_alignment, concept_coverage)
    }

def extract_keywords_from_text(text, max_keywords=20):
    """
    Extract keywords from text using simple frequency analysis
    
    Args:
        text: Text to analyze
        max_keywords: Maximum number of keywords
        
    Returns:
        list: Extracted keywords
    """
    if not text:
        return []
    
    # Clean and tokenize
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 
        'had', 'but', 'have', 'there', 'what', 'said', 'each', 'which', 'she', 'how', 'will', 
        'about', 'they', 'were', 'been', 'their', 'has', 'would', 'more', 'when', 'them', 'these',
        'also', 'from', 'than', 'into', 'during', 'including', 'such', 'other', 'through'
    }
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count frequency
    word_freq = Counter(filtered_words)
    
    # Return top keywords
    return [word for word, count in word_freq.most_common(max_keywords)]

def determine_alignment_quality(alignment_score, concept_coverage):
    """
    Determine quality category for chunk-concept alignment
    
    Args:
        alignment_score: Average alignment score
        concept_coverage: Concept coverage ratio
        
    Returns:
        str: Quality description
    """
    if alignment_score >= 0.6 and concept_coverage >= 0.8:
        return "Excellent alignment - chunk strongly matches assigned concepts"
    elif alignment_score >= 0.4 and concept_coverage >= 0.6:
        return "Good alignment - chunk reasonably matches most concepts"
    elif alignment_score >= 0.2 and concept_coverage >= 0.4:
        return "Fair alignment - chunk partially matches some concepts"
    else:
        return "Poor alignment - chunk poorly matches assigned concepts"

def analyze_chunk_quality_metrics(chunk_data):
    """
    Analyze general quality metrics for a chunk
    
    Args:
        chunk_data: Chunk data
        
    Returns:
        dict: Quality metrics
    """
    chunk_text = chunk_data.get("text", "")
    
    # Basic metrics
    word_count = len(chunk_text.split())
    char_count = len(chunk_text)
    
    # Content density (non-whitespace characters / total characters)
    non_whitespace = len(re.sub(r'\s', '', chunk_text))
    content_density = non_whitespace / max(char_count, 1)
    
    # Sentence structure
    sentences = re.split(r'[.!?]+', chunk_text)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Vocabulary richness
    unique_words = len(set(chunk_text.lower().split()))
    vocabulary_richness = unique_words / max(word_count, 1)
    
    # Quality score calculation
    size_score = 1.0 if 50 <= word_count <= 500 else 0.5  # Optimal chunk size
    density_score = content_density
    structure_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.6
    vocab_score = min(1.0, vocabulary_richness * 2)  # Cap at 1.0
    
    quality_score = (size_score * 0.3 + density_score * 0.2 + 
                    structure_score * 0.2 + vocab_score * 0.3)
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "vocabulary_richness": vocabulary_richness,
        "content_density": content_density,
        "quality_score": quality_score,
        "size_category": categorize_chunk_size(word_count)
    }

def categorize_chunk_size(word_count):
    """Categorize chunk by size"""
    if word_count < 30:
        return "too_small"
    elif word_count <= 150:
        return "optimal"
    elif word_count <= 300:
        return "large"
    else:
        return "too_large"

def inspect_chunk_distribution(all_chunks, concept_registry):
    """
    Inspect overall distribution and patterns in chunks
    
    Args:
        all_chunks: List of all chunks
        concept_registry: Concept registry
        
    Returns:
        dict: Distribution analysis
    """
    # Size distribution
    size_categories = defaultdict(int)
    concept_usage = defaultdict(int)
    quality_scores = []
    alignment_scores = []
    
    for chunk in all_chunks:
        quality = analyze_chunk_quality_metrics(chunk)
        alignment = analyze_chunk_concept_alignment(chunk, concept_registry)
        
        size_categories[quality["size_category"]] += 1
        quality_scores.append(quality["quality_score"])
        alignment_scores.append(alignment["alignment_score"])
        
        # Count concept usage
        for concept_id in chunk.get("assigned_concepts", []):
            concept_usage[concept_id] += 1
    
    # Calculate statistics
    total_chunks = len(all_chunks)
    avg_quality = np.mean(quality_scores) if quality_scores else 0.0
    avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
    
    # Identify unused concepts
    all_concept_ids = set(concept_registry.keys())
    used_concept_ids = set(concept_usage.keys())
    unused_concepts = all_concept_ids - used_concept_ids
    
    return {
        "total_chunks": total_chunks,
        "size_distribution": dict(size_categories),
        "size_percentages": {cat: (count/total_chunks)*100 for cat, count in size_categories.items()},
        "average_quality": avg_quality,
        "average_alignment": avg_alignment,
        "concept_usage_stats": {
            "total_concepts": len(all_concept_ids),
            "used_concepts": len(used_concept_ids),
            "unused_concepts": len(unused_concepts),
            "avg_usage_per_concept": np.mean(list(concept_usage.values())) if concept_usage else 0.0
        },
        "most_used_concepts": sorted(concept_usage.items(), key=lambda x: x[1], reverse=True)[:10],
        "unused_concept_list": list(unused_concepts)
    }

def identify_problematic_chunks(all_chunks, concept_registry, quality_threshold=0.4, alignment_threshold=0.3):
    """
    Identify chunks with quality or alignment issues
    
    Args:
        all_chunks: All chunk data
        concept_registry: Concept registry
        quality_threshold: Minimum quality threshold
        alignment_threshold: Minimum alignment threshold
        
    Returns:
        list: Problematic chunks with details
    """
    problematic_chunks = []
    
    for i, chunk in enumerate(all_chunks):
        quality = analyze_chunk_quality_metrics(chunk)
        alignment = analyze_chunk_concept_alignment(chunk, concept_registry)
        
        issues = []
        
        if quality["quality_score"] < quality_threshold:
            issues.append(f"Low quality score: {quality['quality_score']:.3f}")
        
        if alignment["alignment_score"] < alignment_threshold:
            issues.append(f"Poor concept alignment: {alignment['alignment_score']:.3f}")
        
        if quality["size_category"] in ["too_small", "too_large"]:
            issues.append(f"Suboptimal size: {quality['word_count']} words ({quality['size_category']})")
        
        if alignment["concept_coverage"] < 0.5:
            issues.append(f"Low concept coverage: {alignment['concept_coverage']:.1%}")
        
        if issues:
            problematic_chunks.append({
                "chunk_index": i,
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "issues": issues,
                "quality_score": quality["quality_score"],
                "alignment_score": alignment["alignment_score"],
                "word_count": quality["word_count"],
                "assigned_concepts": chunk.get("assigned_concepts", []),
                "text_preview": chunk.get("text", "")[:100] + "..." if len(chunk.get("text", "")) > 100 else chunk.get("text", "")
            })
    
    # Sort by severity (lowest scores first)
    problematic_chunks.sort(key=lambda x: x["quality_score"] + x["alignment_score"])
    
    return problematic_chunks

def load_inputs():
    """Load chunk data and concept registry"""
    script_dir = Path(__file__).parent.parent
    
    # Try to load semantic chunks (could be from A2.7 or similar)
    chunk_paths = [
        "outputs/A2.7_semantic_chunks.json",
        "outputs/A2.8_concept_aware_semantic_chunks.json",
        "outputs/semantic_chunks.json"
    ]
    
    chunk_data = None
    for path in chunk_paths:
        full_path = script_dir / path
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            break
    
    if not chunk_data:
        # Create mock chunk data for testing
        chunk_data = {
            "chunks": [
                {
                    "chunk_id": "chunk_1",
                    "text": "The company reported revenue of $50.2 million for the quarter, representing a significant increase from the previous period. This growth was driven by strong performance in the financial services sector.",
                    "assigned_concepts": ["core_1", "core_2"]
                },
                {
                    "chunk_id": "chunk_2", 
                    "text": "Current deferred income has changed substantially, with implications for future earnings projections and cash flow management strategies.",
                    "assigned_concepts": ["core_1"]
                }
            ]
        }
    
    # Load concept registry
    concept_paths = [
        "outputs/A3_optimized_centroids.json",
        "outputs/A2.5_expanded_concepts.json",
        "outputs/A2.4_core_concepts.json"
    ]
    
    concept_registry = {}
    for path in concept_paths:
        full_path = script_dir / path
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if "expanded_concepts" in data:
                    concept_registry = data["expanded_concepts"]
                elif "core_concepts" in data:
                    concept_registry = {c["concept_id"]: c for c in data["core_concepts"]}
                else:
                    concept_registry = data
            break
    
    if not concept_registry:
        # Mock concept registry
        concept_registry = {
            "core_1": {
                "concept_id": "core_1",
                "theme_name": "Financial Metrics",
                "primary_keywords": ["revenue", "income", "financial", "current", "deferred"],
                "domain": "finance"
            },
            "core_2": {
                "concept_id": "core_2",
                "theme_name": "Performance Analysis",
                "primary_keywords": ["growth", "performance", "increase", "quarter"],
                "domain": "general"
            }
        }
    
    return chunk_data, concept_registry

def process_chunk_inspection(chunk_data, concept_registry):
    """
    Process comprehensive chunk inspection
    
    Args:
        chunk_data: All chunk data
        concept_registry: Concept registry
        
    Returns:
        dict: Inspection results
    """
    chunks = chunk_data.get("chunks", [])
    if not chunks:
        return {"error": "No chunks found for inspection"}
    
    # Analyze individual chunks
    individual_analyses = []
    for i, chunk in enumerate(chunks):
        quality = analyze_chunk_quality_metrics(chunk)
        alignment = analyze_chunk_concept_alignment(chunk, concept_registry)
        
        individual_analyses.append({
            "chunk_index": i,
            "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
            "quality_analysis": quality,
            "alignment_analysis": alignment
        })
    
    # Overall distribution analysis
    distribution = inspect_chunk_distribution(chunks, concept_registry)
    
    # Identify problems
    problematic_chunks = identify_problematic_chunks(chunks, concept_registry)
    
    return {
        "inspection_summary": {
            "total_chunks_inspected": len(chunks),
            "total_concepts_in_registry": len(concept_registry),
            "average_chunk_quality": distribution["average_quality"],
            "average_concept_alignment": distribution["average_alignment"],
            "problematic_chunks_count": len(problematic_chunks)
        },
        "individual_analyses": individual_analyses,
        "distribution_analysis": distribution,
        "problematic_chunks": problematic_chunks,
        "inspection_recommendations": generate_inspection_recommendations(distribution, problematic_chunks)
    }

def generate_inspection_recommendations(distribution, problematic_chunks):
    """
    Generate recommendations based on inspection results
    
    Args:
        distribution: Distribution analysis
        problematic_chunks: List of problematic chunks
        
    Returns:
        list: Recommendations
    """
    recommendations = []
    
    # Size distribution recommendations
    size_dist = distribution["size_percentages"]
    if size_dist.get("too_small", 0) > 20:
        recommendations.append("Many chunks are too small - consider merging adjacent chunks")
    
    if size_dist.get("too_large", 0) > 15:
        recommendations.append("Many chunks are too large - implement hierarchical chunking")
    
    # Quality recommendations
    if distribution["average_quality"] < 0.5:
        recommendations.append("Overall chunk quality is low - review chunking strategy")
    
    # Alignment recommendations
    if distribution["average_alignment"] < 0.4:
        recommendations.append("Poor concept-chunk alignment - review concept assignment logic")
    
    # Concept usage recommendations
    usage_stats = distribution["concept_usage_stats"]
    unused_ratio = usage_stats["unused_concepts"] / usage_stats["total_concepts"]
    if unused_ratio > 0.3:
        recommendations.append(f"{unused_ratio:.1%} of concepts are unused - review concept relevance")
    
    # Problematic chunks
    if len(problematic_chunks) > len(distribution.get("size_distribution", {}).get("optimal", [])) * 0.2:
        recommendations.append("High number of problematic chunks - review chunking parameters")
    
    return recommendations

def save_output(inspection_results):
    """Save inspection results"""
    script_dir = Path(__file__).parent.parent
    
    # Add metadata
    inspection_results["inspection_metadata"] = {
        "inspection_timestamp": datetime.now().isoformat(),
        "inspector_version": "A37_v1.0"
    }
    
    # Save main inspection report
    output_path = script_dir / "outputs/A37_chunk_inspection_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inspection_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved chunk inspection report to {output_path}")
    
    # Save summary text file
    summary_path = output_path.with_suffix('.txt').with_name(output_path.stem + '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        summary = inspection_results["inspection_summary"]
        dist = inspection_results["distribution_analysis"]
        problems = inspection_results["problematic_chunks"]
        
        f.write("CONCEPT CHUNK INSPECTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Chunks Inspected: {summary['total_chunks_inspected']}\n")
        f.write(f"Average Quality Score: {summary['average_chunk_quality']:.3f}\n")
        f.write(f"Average Alignment Score: {summary['average_concept_alignment']:.3f}\n")
        f.write(f"Problematic Chunks: {summary['problematic_chunks_count']}\n\n")
        
        f.write("Size Distribution:\n")
        for size_cat, percentage in dist["size_percentages"].items():
            f.write(f"  {size_cat}: {percentage:.1f}%\n")
        
        f.write(f"\nConcept Usage:\n")
        usage = dist["concept_usage_stats"]
        f.write(f"  Used Concepts: {usage['used_concepts']}/{usage['total_concepts']}\n")
        f.write(f"  Unused Concepts: {usage['unused_concepts']}\n")
        
        if inspection_results["inspection_recommendations"]:
            f.write(f"\nRecommendations:\n")
            for rec in inspection_results["inspection_recommendations"]:
                f.write(f"  • {rec}\n")

def main():
    """Main execution"""
    print("="*60)
    print("A37: Concept Chunk Inspection")
    print("="*60)
    
    try:
        # Load inputs
        print("Loading chunk data and concept registry...")
        chunk_data, concept_registry = load_inputs()
        
        # Process inspection
        chunks = chunk_data.get("chunks", [])
        print(f"Inspecting {len(chunks)} chunks against {len(concept_registry)} concepts...")
        
        inspection_results = process_chunk_inspection(chunk_data, concept_registry)
        
        if "error" in inspection_results:
            print(f"❌ {inspection_results['error']}")
            return
        
        # Display results
        summary = inspection_results["inspection_summary"]
        print(f"\nChunk Inspection Results:")
        print(f"  Total Chunks: {summary['total_chunks_inspected']}")
        print(f"  Average Quality: {summary['average_chunk_quality']:.3f}")
        print(f"  Average Alignment: {summary['average_concept_alignment']:.3f}")
        print(f"  Problematic Chunks: {summary['problematic_chunks_count']}")
        
        # Size distribution
        dist = inspection_results["distribution_analysis"]
        print(f"\nSize Distribution:")
        for size_cat, percentage in dist["size_percentages"].items():
            print(f"  {size_cat.replace('_', ' ').title()}: {percentage:.1f}%")
        
        # Concept usage
        usage = dist["concept_usage_stats"]
        print(f"\nConcept Usage:")
        print(f"  Used: {usage['used_concepts']}/{usage['total_concepts']} concepts")
        print(f"  Unused: {usage['unused_concepts']} concepts")
        
        # Top problematic chunks
        problems = inspection_results["problematic_chunks"]
        if problems:
            print(f"\nTop Problematic Chunks:")
            for i, prob in enumerate(problems[:3], 1):
                print(f"  {i}. {prob['chunk_id']}")
                print(f"     Quality: {prob['quality_score']:.3f}, Alignment: {prob['alignment_score']:.3f}")
                print(f"     Issues: {prob['issues'][0] if prob['issues'] else 'None'}")
        
        # Recommendations
        recommendations = inspection_results["inspection_recommendations"]
        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        # Save results
        save_output(inspection_results)
        
        print("\nA37 Concept Chunk Inspection completed successfully!")
        
    except Exception as e:
        print(f"Error in A37 Inspection: {str(e)}")
        raise

if __name__ == "__main__":
    main()