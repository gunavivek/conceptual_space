#!/usr/bin/env python3
"""
R2: Concept Validation
Validates extracted concepts against BizBOK reference concepts to assess
concept quality, coverage, and alignment with established business knowledge
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

def load_reference_concepts(reference_path="output/R1_bizbok_concepts.json"):
    """
    Load BizBOK reference concepts
    
    Args:
        reference_path: Path to BizBOK concepts file
        
    Returns:
        dict: Reference concepts and mappings
    """
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / reference_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Reference concepts not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_pipeline_concepts():
    """
    Load concepts from A-pipeline for validation
    
    Returns:
        dict: Pipeline concepts to validate
    """
    script_dir = Path(__file__).parent.parent
    
    # Try multiple potential concept sources
    concept_paths = [
        "../A_concept_pipeline/outputs/A2.5_expanded_concepts.json",
        "../A_concept_pipeline/outputs/A2.4_core_concepts.json",
        "../outputs/A2.5_expanded_concepts.json"
    ]
    
    for concept_path in concept_paths:
        full_path = script_dir / concept_path
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different formats
                if "expanded_concepts" in data:
                    return data["expanded_concepts"]
                elif "core_concepts" in data:
                    return {c["concept_id"]: c for c in data["core_concepts"]}
                else:
                    return data
    
    # Create mock pipeline concepts for testing
    return {
        "core_1": {
            "concept_id": "core_1",
            "theme_name": "Financial Performance",
            "primary_keywords": ["revenue", "income", "financial", "performance"],
            "domain": "finance",
            "importance_score": 0.8
        },
        "core_2": {
            "concept_id": "core_2", 
            "theme_name": "Cash Management",
            "primary_keywords": ["cash", "flow", "liquidity", "working"],
            "domain": "finance",
            "importance_score": 0.7
        }
    }

def calculate_concept_similarity(pipeline_concept, reference_concept):
    """
    Calculate similarity between pipeline and reference concepts
    
    Args:
        pipeline_concept: Concept from A-pipeline
        reference_concept: BizBOK reference concept
        
    Returns:
        dict: Similarity analysis
    """
    # Extract terms from pipeline concept
    if "original_concept" in pipeline_concept:
        pipeline_terms = pipeline_concept["original_concept"].get("primary_keywords", [])
        pipeline_terms.extend(pipeline_concept.get("all_expanded_terms", []))
    else:
        pipeline_terms = pipeline_concept.get("primary_keywords", [])
    
    # Extract terms from reference concept
    reference_terms = reference_concept.get("all_terms", [])
    
    # Convert to lowercase sets
    pipeline_set = set(term.lower() for term in pipeline_terms)
    reference_set = set(term.lower() for term in reference_terms)
    
    # Calculate Jaccard similarity
    if pipeline_set and reference_set:
        intersection = len(pipeline_set & reference_set)
        union = len(pipeline_set | reference_set)
        jaccard_similarity = intersection / union
    else:
        jaccard_similarity = 0.0
    
    # Domain alignment bonus
    pipeline_domain = pipeline_concept.get("domain", "general")
    reference_domain = reference_concept.get("domain", "general")
    domain_bonus = 0.2 if pipeline_domain == reference_domain else 0.0
    
    # Calculate overall similarity
    overall_similarity = jaccard_similarity + domain_bonus
    
    return {
        "jaccard_similarity": jaccard_similarity,
        "domain_alignment": pipeline_domain == reference_domain,
        "domain_bonus": domain_bonus,
        "overall_similarity": min(1.0, overall_similarity),
        "common_terms": list(pipeline_set & reference_set),
        "pipeline_unique": list(pipeline_set - reference_set),
        "reference_unique": list(reference_set - pipeline_set)
    }

def validate_pipeline_concepts(pipeline_concepts, reference_data):
    """
    Validate pipeline concepts against BizBOK references
    
    Args:
        pipeline_concepts: Concepts from A-pipeline
        reference_data: BizBOK reference data
        
    Returns:
        dict: Validation results
    """
    reference_concepts = reference_data["bizbok_concepts"]
    validation_results = {}
    
    for pipeline_id, pipeline_concept in pipeline_concepts.items():
        concept_validation = {
            "pipeline_concept_id": pipeline_id,
            "pipeline_concept": pipeline_concept,
            "reference_matches": [],
            "best_match": None,
            "validation_score": 0.0,
            "coverage_assessment": "unknown"
        }
        
        # Compare with all reference concepts
        for ref_id, ref_concept in reference_concepts.items():
            similarity = calculate_concept_similarity(pipeline_concept, ref_concept)
            
            if similarity["overall_similarity"] > 0.1:  # Minimum similarity threshold
                match_data = {
                    "reference_id": ref_id,
                    "reference_concept": ref_concept,
                    "similarity_analysis": similarity
                }
                concept_validation["reference_matches"].append(match_data)
        
        # Sort matches by similarity
        concept_validation["reference_matches"].sort(
            key=lambda x: x["similarity_analysis"]["overall_similarity"], 
            reverse=True
        )
        
        # Set best match and validation score
        if concept_validation["reference_matches"]:
            concept_validation["best_match"] = concept_validation["reference_matches"][0]
            concept_validation["validation_score"] = concept_validation["best_match"]["similarity_analysis"]["overall_similarity"]
            
            # Determine coverage assessment
            if concept_validation["validation_score"] >= 0.7:
                concept_validation["coverage_assessment"] = "excellent"
            elif concept_validation["validation_score"] >= 0.5:
                concept_validation["coverage_assessment"] = "good"
            elif concept_validation["validation_score"] >= 0.3:
                concept_validation["coverage_assessment"] = "fair"
            else:
                concept_validation["coverage_assessment"] = "poor"
        else:
            concept_validation["coverage_assessment"] = "no_match"
        
        validation_results[pipeline_id] = concept_validation
    
    return validation_results

def analyze_validation_coverage(validation_results):
    """
    Analyze overall validation coverage and quality
    
    Args:
        validation_results: Individual concept validation results
        
    Returns:
        dict: Coverage analysis
    """
    total_concepts = len(validation_results)
    coverage_counts = Counter()
    validation_scores = []
    domain_performance = defaultdict(list)
    
    for concept_id, validation in validation_results.items():
        coverage = validation["coverage_assessment"]
        score = validation["validation_score"]
        
        coverage_counts[coverage] += 1
        validation_scores.append(score)
        
        # Domain performance tracking
        if "original_concept" in validation["pipeline_concept"]:
            domain = validation["pipeline_concept"]["original_concept"].get("domain", "general")
        else:
            domain = validation["pipeline_concept"].get("domain", "general")
        
        domain_performance[domain].append(score)
    
    # Calculate domain averages
    domain_averages = {}
    for domain, scores in domain_performance.items():
        domain_averages[domain] = sum(scores) / len(scores) if scores else 0.0
    
    # Overall statistics
    avg_validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
    
    return {
        "total_concepts_validated": total_concepts,
        "coverage_distribution": dict(coverage_counts),
        "coverage_percentages": {k: (v/total_concepts)*100 for k, v in coverage_counts.items()},
        "average_validation_score": avg_validation_score,
        "domain_performance": domain_averages,
        "score_distribution": {
            "high_quality": len([s for s in validation_scores if s >= 0.7]),
            "medium_quality": len([s for s in validation_scores if 0.4 <= s < 0.7]),
            "low_quality": len([s for s in validation_scores if s < 0.4])
        }
    }

def identify_coverage_gaps(validation_results, reference_data):
    """
    Identify coverage gaps in pipeline concepts
    
    Args:
        validation_results: Validation results
        reference_data: BizBOK reference data
        
    Returns:
        dict: Coverage gap analysis
    """
    reference_concepts = reference_data["bizbok_concepts"]
    
    # Track which reference concepts are covered
    covered_references = set()
    coverage_quality = {}
    
    for validation in validation_results.values():
        if validation["best_match"]:
            ref_id = validation["best_match"]["reference_id"]
            score = validation["validation_score"]
            
            covered_references.add(ref_id)
            if ref_id not in coverage_quality or coverage_quality[ref_id] < score:
                coverage_quality[ref_id] = score
    
    # Identify uncovered references
    all_references = set(reference_concepts.keys())
    uncovered_references = all_references - covered_references
    
    # Analyze uncovered by domain and importance
    gap_analysis = defaultdict(list)
    for ref_id in uncovered_references:
        ref_concept = reference_concepts[ref_id]
        domain = ref_concept["domain"]
        importance = ref_concept["importance_score"]
        
        gap_analysis[domain].append({
            "reference_id": ref_id,
            "concept_name": ref_concept["name"],
            "importance": importance,
            "category": ref_concept["category"]
        })
    
    # Sort gaps by importance
    for domain in gap_analysis:
        gap_analysis[domain].sort(key=lambda x: x["importance"], reverse=True)
    
    return {
        "total_references": len(all_references),
        "covered_references": len(covered_references),
        "uncovered_references": len(uncovered_references),
        "coverage_ratio": len(covered_references) / len(all_references),
        "coverage_gaps_by_domain": dict(gap_analysis),
        "high_importance_gaps": [
            gap for gaps in gap_analysis.values() 
            for gap in gaps if gap["importance"] >= 0.8
        ]
    }

def generate_validation_recommendations(coverage_analysis, gap_analysis):
    """
    Generate recommendations for improving concept validation
    
    Args:
        coverage_analysis: Overall coverage analysis
        gap_analysis: Coverage gap analysis
        
    Returns:
        list: Validation improvement recommendations
    """
    recommendations = []
    
    # Overall quality recommendations
    avg_score = coverage_analysis["average_validation_score"]
    if avg_score < 0.5:
        recommendations.append("Overall validation quality is low - review concept extraction strategies")
    
    # Coverage recommendations
    coverage_ratio = gap_analysis["coverage_ratio"]
    if coverage_ratio < 0.7:
        recommendations.append(f"Low reference coverage ({coverage_ratio:.1%}) - expand concept identification scope")
    
    # Domain-specific recommendations
    for domain, avg_score in coverage_analysis["domain_performance"].items():
        if avg_score < 0.4:
            recommendations.append(f"Poor {domain} domain performance - strengthen domain-specific concept extraction")
    
    # High-importance gap recommendations
    high_gaps = gap_analysis["high_importance_gaps"]
    if len(high_gaps) > 3:
        recommendations.append(f"{len(high_gaps)} high-importance concepts missing - prioritize these for extraction improvement")
    
    # Quality distribution recommendations
    score_dist = coverage_analysis["score_distribution"]
    if score_dist["low_quality"] > score_dist["high_quality"]:
        recommendations.append("More low-quality than high-quality matches - improve concept extraction precision")
    
    return recommendations

def save_validation_results(validation_results, coverage_analysis, gap_analysis, recommendations):
    """
    Save validation results and analysis
    
    Args:
        validation_results: Individual concept validations
        coverage_analysis: Coverage analysis
        gap_analysis: Gap analysis  
        recommendations: Improvement recommendations
    """
    script_dir = Path(__file__).parent.parent
    
    # Main validation report
    report_data = {
        "validation_results": validation_results,
        "coverage_analysis": coverage_analysis,
        "gap_analysis": gap_analysis,
        "recommendations": recommendations,
        "validation_metadata": {
            "validation_timestamp": datetime.now().isoformat(),
            "total_concepts_validated": len(validation_results),
            "validation_method": "BizBOK_reference_comparison"
        }
    }
    
    output_path = script_dir / "output/R2_concept_validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved validation report to {output_path}")
    
    # Summary text file
    summary_path = output_path.with_suffix('.txt').with_name(output_path.stem + '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CONCEPT VALIDATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Concepts Validated: {coverage_analysis['total_concepts_validated']}\n")
        f.write(f"Average Validation Score: {coverage_analysis['average_validation_score']:.3f}\n")
        f.write(f"Reference Coverage: {gap_analysis['coverage_ratio']:.1%}\n\n")
        
        f.write("Coverage Distribution:\n")
        for coverage, percentage in coverage_analysis["coverage_percentages"].items():
            f.write(f"  {coverage.replace('_', ' ').title()}: {percentage:.1f}%\n")
        
        f.write(f"\nDomain Performance:\n")
        for domain, score in coverage_analysis["domain_performance"].items():
            f.write(f"  {domain.title()}: {score:.3f}\n")
        
        f.write(f"\nRecommendations:\n")
        for rec in recommendations:
            f.write(f"  • {rec}\n")

def main():
    """Main execution"""
    print("="*60)
    print("R2: Concept Validation")
    print("="*60)
    
    try:
        # Load reference concepts
        print("Loading BizBOK reference concepts...")
        reference_data = load_reference_concepts()
        
        # Load pipeline concepts
        print("Loading pipeline concepts for validation...")
        pipeline_concepts = load_pipeline_concepts()
        
        # Perform validation
        print(f"Validating {len(pipeline_concepts)} pipeline concepts...")
        validation_results = validate_pipeline_concepts(pipeline_concepts, reference_data)
        
        # Analyze coverage
        print("Analyzing validation coverage...")
        coverage_analysis = analyze_validation_coverage(validation_results)
        
        # Identify gaps
        print("Identifying coverage gaps...")
        gap_analysis = identify_coverage_gaps(validation_results, reference_data)
        
        # Generate recommendations
        recommendations = generate_validation_recommendations(coverage_analysis, gap_analysis)
        
        # Display results
        print(f"\nValidation Results:")
        print(f"  Total Concepts: {coverage_analysis['total_concepts_validated']}")
        print(f"  Average Score: {coverage_analysis['average_validation_score']:.3f}")
        print(f"  Reference Coverage: {gap_analysis['coverage_ratio']:.1%}")
        
        print(f"\nCoverage Quality:")
        for coverage, percentage in coverage_analysis["coverage_percentages"].items():
            print(f"  {coverage.replace('_', ' ').title()}: {percentage:.1f}%")
        
        print(f"\nDomain Performance:")
        for domain, score in coverage_analysis["domain_performance"].items():
            print(f"  {domain.title()}: {score:.3f}")
        
        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations[:3]:
                print(f"  • {rec}")
        
        # Save results
        save_validation_results(validation_results, coverage_analysis, gap_analysis, recommendations)
        
        print("\nR2 Concept Validation completed successfully!")
        
    except Exception as e:
        print(f"Error in R2 Concept Validation: {str(e)}")
        raise

if __name__ == "__main__":
    main()