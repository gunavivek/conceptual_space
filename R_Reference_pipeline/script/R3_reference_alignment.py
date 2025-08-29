#!/usr/bin/env python3
"""
R3: Reference Alignment
Aligns pipeline concepts with reference standards and creates alignment mappings
for improved concept consistency and standardization across the system
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

def load_validation_results(validation_path="output/R2_concept_validation_report.json"):
    """
    Load concept validation results from R2
    
    Args:
        validation_path: Path to validation report
        
    Returns:
        dict: Validation results and analysis
    """
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / validation_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Validation results not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_alignment_mappings(validation_results):
    """
    Create alignment mappings between pipeline and reference concepts
    
    Args:
        validation_results: Validation results from R2
        
    Returns:
        dict: Alignment mappings
    """
    alignment_mappings = {
        "direct_alignments": {},  # High-confidence alignments
        "suggested_alignments": {},  # Medium-confidence alignments
        "concept_standardizations": {},  # Standardized concept definitions
        "term_unifications": defaultdict(set)  # Unified terminology
    }
    
    for pipeline_id, validation in validation_results.items():
        validation_score = validation["validation_score"]
        
        if validation_score >= 0.7:
            # High-confidence direct alignment
            best_match = validation["best_match"]
            reference_concept = best_match["reference_concept"]
            
            alignment_mappings["direct_alignments"][pipeline_id] = {
                "reference_id": best_match["reference_id"],
                "reference_name": reference_concept["name"],
                "alignment_confidence": validation_score,
                "standardized_definition": reference_concept["definition"],
                "standardized_terms": reference_concept["related_terms"],
                "alignment_type": "direct"
            }
            
            # Add to term unifications
            similarity = best_match["similarity_analysis"]
            common_terms = similarity["common_terms"]
            for term in common_terms:
                alignment_mappings["term_unifications"][term].add(pipeline_id)
                alignment_mappings["term_unifications"][term].add(best_match["reference_id"])
        
        elif validation_score >= 0.4:
            # Medium-confidence suggested alignment
            best_match = validation["best_match"]
            reference_concept = best_match["reference_concept"]
            
            alignment_mappings["suggested_alignments"][pipeline_id] = {
                "reference_id": best_match["reference_id"],
                "reference_name": reference_concept["name"],
                "alignment_confidence": validation_score,
                "suggested_definition": reference_concept["definition"],
                "suggested_terms": reference_concept["related_terms"],
                "alignment_type": "suggested",
                "review_required": True
            }
    
    # Convert term unifications to lists
    alignment_mappings["term_unifications"] = {
        term: list(concepts) for term, concepts in alignment_mappings["term_unifications"].items()
    }
    
    return alignment_mappings

def generate_standardized_concepts(alignment_mappings, validation_data):
    """
    Generate standardized concept definitions based on alignments
    
    Args:
        alignment_mappings: Alignment mappings
        validation_data: Original validation data
        
    Returns:
        dict: Standardized concept definitions
    """
    standardized_concepts = {}
    validation_results = validation_data["validation_results"]
    
    # Process direct alignments
    for pipeline_id, alignment in alignment_mappings["direct_alignments"].items():
        pipeline_concept = validation_results[pipeline_id]["pipeline_concept"]
        
        # Create standardized concept
        standardized_concept = {
            "concept_id": pipeline_id,
            "standardized_name": alignment["reference_name"],
            "original_name": get_concept_name(pipeline_concept),
            "standardized_definition": alignment["standardized_definition"],
            "standardized_terms": alignment["standardized_terms"],
            "domain": get_concept_domain(pipeline_concept),
            "alignment_confidence": alignment["alignment_confidence"],
            "source": "reference_aligned",
            "reference_id": alignment["reference_id"]
        }
        
        # Merge original terms with standardized terms
        original_terms = get_concept_terms(pipeline_concept)
        merged_terms = list(set(original_terms + alignment["standardized_terms"]))
        standardized_concept["merged_terms"] = merged_terms
        
        standardized_concepts[pipeline_id] = standardized_concept
    
    # Process unaligned concepts (create custom standardizations)
    for pipeline_id, validation in validation_results.items():
        if pipeline_id not in standardized_concepts:
            pipeline_concept = validation["pipeline_concept"]
            
            # Create custom standardization for unaligned concepts
            standardized_concept = {
                "concept_id": pipeline_id,
                "standardized_name": get_concept_name(pipeline_concept),
                "original_name": get_concept_name(pipeline_concept),
                "standardized_definition": f"Custom concept: {get_concept_name(pipeline_concept)}",
                "standardized_terms": get_concept_terms(pipeline_concept),
                "domain": get_concept_domain(pipeline_concept),
                "alignment_confidence": 0.0,
                "source": "pipeline_custom",
                "reference_id": None,
                "custom_standardization": True
            }
            
            standardized_concepts[pipeline_id] = standardized_concept
    
    return standardized_concepts

def get_concept_name(concept):
    """Extract concept name from various concept formats"""
    if "original_concept" in concept:
        return concept["original_concept"].get("theme_name", "Unknown")
    return concept.get("theme_name", concept.get("name", "Unknown"))

def get_concept_domain(concept):
    """Extract concept domain from various concept formats"""
    if "original_concept" in concept:
        return concept["original_concept"].get("domain", "general")
    return concept.get("domain", "general")

def get_concept_terms(concept):
    """Extract concept terms from various concept formats"""
    if "original_concept" in concept:
        terms = concept["original_concept"].get("primary_keywords", [])
        terms.extend(concept.get("all_expanded_terms", []))
        return terms
    return concept.get("primary_keywords", [])

def analyze_alignment_quality(alignment_mappings, standardized_concepts):
    """
    Analyze quality of alignment results
    
    Args:
        alignment_mappings: Alignment mappings
        standardized_concepts: Standardized concepts
        
    Returns:
        dict: Alignment quality analysis
    """
    total_concepts = len(standardized_concepts)
    direct_alignments = len(alignment_mappings["direct_alignments"])
    suggested_alignments = len(alignment_mappings["suggested_alignments"])
    custom_concepts = total_concepts - direct_alignments - suggested_alignments
    
    # Domain alignment analysis
    domain_alignment_quality = defaultdict(list)
    for concept in standardized_concepts.values():
        domain = concept["domain"]
        confidence = concept["alignment_confidence"]
        domain_alignment_quality[domain].append(confidence)
    
    domain_averages = {}
    for domain, confidences in domain_alignment_quality.items():
        domain_averages[domain] = sum(confidences) / len(confidences)
    
    # Term unification analysis
    unified_terms = len(alignment_mappings["term_unifications"])
    avg_concepts_per_term = np.mean([
        len(concepts) for concepts in alignment_mappings["term_unifications"].values()
    ]) if unified_terms > 0 else 0
    
    return {
        "total_concepts": total_concepts,
        "alignment_distribution": {
            "direct_alignments": direct_alignments,
            "suggested_alignments": suggested_alignments,
            "custom_concepts": custom_concepts
        },
        "alignment_percentages": {
            "direct": (direct_alignments / total_concepts) * 100,
            "suggested": (suggested_alignments / total_concepts) * 100,
            "custom": (custom_concepts / total_concepts) * 100
        },
        "domain_alignment_quality": domain_averages,
        "terminology_unification": {
            "unified_terms": unified_terms,
            "avg_concepts_per_term": avg_concepts_per_term
        },
        "overall_alignment_rate": ((direct_alignments + suggested_alignments) / total_concepts) * 100
    }

def generate_alignment_recommendations(quality_analysis, alignment_mappings):
    """
    Generate recommendations for improving alignment
    
    Args:
        quality_analysis: Alignment quality analysis
        alignment_mappings: Current alignment mappings
        
    Returns:
        list: Alignment improvement recommendations
    """
    recommendations = []
    
    # Overall alignment rate recommendations
    alignment_rate = quality_analysis["overall_alignment_rate"]
    if alignment_rate < 70:
        recommendations.append(f"Low alignment rate ({alignment_rate:.1f}%) - review concept extraction to better match reference standards")
    
    # Custom concept recommendations
    custom_percentage = quality_analysis["alignment_percentages"]["custom"]
    if custom_percentage > 40:
        recommendations.append(f"High custom concepts ({custom_percentage:.1f}%) - consider expanding reference concept library")
    
    # Domain-specific recommendations
    for domain, avg_confidence in quality_analysis["domain_alignment_quality"].items():
        if avg_confidence < 0.5:
            recommendations.append(f"Poor {domain} alignment quality - strengthen {domain} reference concepts")
    
    # Suggested alignment recommendations
    suggested_count = quality_analysis["alignment_distribution"]["suggested_alignments"]
    if suggested_count > 5:
        recommendations.append(f"{suggested_count} concepts need manual review for alignment confirmation")
    
    # Term unification recommendations
    unified_terms = quality_analysis["terminology_unification"]["unified_terms"]
    if unified_terms < 10:
        recommendations.append("Low term unification - improve concept term standardization")
    
    return recommendations

def create_alignment_export(standardized_concepts, alignment_mappings):
    """
    Create exportable alignment data for use in other systems
    
    Args:
        standardized_concepts: Standardized concept definitions
        alignment_mappings: Alignment mappings
        
    Returns:
        dict: Export-ready alignment data
    """
    export_data = {
        "concept_dictionary": {},
        "term_mappings": {},
        "domain_hierarchies": defaultdict(list),
        "alignment_registry": {}
    }
    
    # Create concept dictionary
    for concept_id, concept in standardized_concepts.items():
        export_data["concept_dictionary"][concept_id] = {
            "name": concept["standardized_name"],
            "definition": concept["standardized_definition"],
            "terms": concept.get("merged_terms", concept["standardized_terms"]),
            "domain": concept["domain"],
            "source": concept["source"]
        }
        
        # Add to domain hierarchy
        export_data["domain_hierarchies"][concept["domain"]].append(concept_id)
    
    # Create term mappings
    for term, concept_ids in alignment_mappings["term_unifications"].items():
        export_data["term_mappings"][term] = concept_ids
    
    # Create alignment registry
    for pipeline_id, alignment in alignment_mappings["direct_alignments"].items():
        export_data["alignment_registry"][pipeline_id] = {
            "type": "direct",
            "reference_id": alignment["reference_id"],
            "confidence": alignment["alignment_confidence"]
        }
    
    for pipeline_id, alignment in alignment_mappings["suggested_alignments"].items():
        export_data["alignment_registry"][pipeline_id] = {
            "type": "suggested", 
            "reference_id": alignment["reference_id"],
            "confidence": alignment["alignment_confidence"]
        }
    
    return export_data

def save_alignment_results(alignment_mappings, standardized_concepts, quality_analysis, recommendations, export_data):
    """
    Save alignment results and analysis
    
    Args:
        alignment_mappings: Alignment mappings
        standardized_concepts: Standardized concepts
        quality_analysis: Quality analysis
        recommendations: Improvement recommendations
        export_data: Export-ready data
    """
    script_dir = Path(__file__).parent.parent
    
    # Main alignment report
    report_data = {
        "alignment_mappings": alignment_mappings,
        "standardized_concepts": standardized_concepts,
        "quality_analysis": quality_analysis,
        "recommendations": recommendations,
        "alignment_metadata": {
            "alignment_timestamp": datetime.now().isoformat(),
            "total_concepts_aligned": len(standardized_concepts),
            "alignment_method": "reference_similarity_matching"
        }
    }
    
    output_path = script_dir / "output/R3_reference_alignment_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved alignment report to {output_path}")
    
    # Export data for other systems
    export_path = script_dir / "output/R3_concept_alignment_export.json"
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved alignment export to {export_path}")

def main():
    """Main execution"""
    print("="*60)
    print("R3: Reference Alignment")
    print("="*60)
    
    try:
        # Load validation results
        print("Loading concept validation results...")
        validation_data = load_validation_results()
        validation_results = validation_data["validation_results"]
        
        # Create alignment mappings
        print(f"Creating alignment mappings for {len(validation_results)} concepts...")
        alignment_mappings = create_alignment_mappings(validation_results)
        
        # Generate standardized concepts
        print("Generating standardized concept definitions...")
        standardized_concepts = generate_standardized_concepts(alignment_mappings, validation_data)
        
        # Analyze alignment quality
        print("Analyzing alignment quality...")
        quality_analysis = analyze_alignment_quality(alignment_mappings, standardized_concepts)
        
        # Generate recommendations
        recommendations = generate_alignment_recommendations(quality_analysis, alignment_mappings)
        
        # Create export data
        print("Creating alignment export data...")
        export_data = create_alignment_export(standardized_concepts, alignment_mappings)
        
        # Display results
        print(f"\nAlignment Results:")
        print(f"  Total Concepts: {quality_analysis['total_concepts']}")
        print(f"  Direct Alignments: {quality_analysis['alignment_distribution']['direct_alignments']} ({quality_analysis['alignment_percentages']['direct']:.1f}%)")
        print(f"  Suggested Alignments: {quality_analysis['alignment_distribution']['suggested_alignments']} ({quality_analysis['alignment_percentages']['suggested']:.1f}%)")
        print(f"  Custom Concepts: {quality_analysis['alignment_distribution']['custom_concepts']} ({quality_analysis['alignment_percentages']['custom']:.1f}%)")
        print(f"  Overall Alignment Rate: {quality_analysis['overall_alignment_rate']:.1f}%")
        
        print(f"\nDomain Alignment Quality:")
        for domain, quality in quality_analysis["domain_alignment_quality"].items():
            print(f"  {domain.title()}: {quality:.3f}")
        
        print(f"\nTerminology Unification:")
        print(f"  Unified Terms: {quality_analysis['terminology_unification']['unified_terms']}")
        print(f"  Avg Concepts/Term: {quality_analysis['terminology_unification']['avg_concepts_per_term']:.1f}")
        
        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations[:3]:
                print(f"  • {rec}")
        
        # Save results
        save_alignment_results(alignment_mappings, standardized_concepts, quality_analysis, recommendations, export_data)
        
        print("\nR3 Reference Alignment completed successfully!")
        
    except Exception as e:
        print(f"Error in R3 Reference Alignment: {str(e)}")
        raise

if __name__ == "__main__":
    main()