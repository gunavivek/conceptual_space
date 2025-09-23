#!/usr/bin/env python3
"""
A2.5: Expanded Concepts Orchestrator
Orchestrates and combines results from all concept expansion strategies
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import importlib.util
import sys

# Strategy weights for combining results
STRATEGY_WEIGHTS = {
    "semantic_similarity": 0.25,
    "domain_knowledge": 0.30,
    "hierarchical_clustering": 0.20,
    "frequency_based": 0.15,
    "contextual_embedding": 0.10
}

def load_strategy_results():
    """
    Load results from all expansion strategies
    
    Returns:
        dict: Combined strategy results
    """
    script_dir = Path(__file__).parent.parent
    strategy_results = {}
    
    strategy_files = {
        "semantic_similarity": "outputs/A2.5.1_semantic_expansion.json",
        "domain_knowledge": "outputs/A2.5.2_domain_expansion.json",
        "hierarchical_clustering": "outputs/A2.5.3_hierarchical_expansion.json",
        "frequency_based": "outputs/A2.5.4_frequency_expansion.json",
        "contextual_embedding": "outputs/A2.5.5_contextual_expansion.json"
    }
    
    print("Loading expansion strategy results...")
    
    for strategy_name, file_path in strategy_files.items():
        full_path = script_dir / file_path
        
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    strategy_results[strategy_name] = data
                    print(f"  [OK] Loaded {strategy_name}")
            except Exception as e:
                print(f"  [FAIL] Failed to load {strategy_name}: {e}")
                strategy_results[strategy_name] = None
        else:
            print(f"  [WARNING] {strategy_name} results not found, running strategy...")
            # Try to run the strategy
            strategy_results[strategy_name] = run_strategy(strategy_name, script_dir)
    
    return strategy_results

def run_strategy(strategy_name, script_dir):
    """
    Run a specific expansion strategy
    
    Args:
        strategy_name: Name of strategy
        script_dir: Script directory
        
    Returns:
        dict or None: Strategy results
    """
    strategy_scripts = {
        "semantic_similarity": "A2.5.1_semantic_similarity_expansion.py",
        "domain_knowledge": "A2.5.2_domain_knowledge_expansion.py",
        "hierarchical_clustering": "A2.5.3_hierarchical_clustering_expansion.py",
        "frequency_based": "A2.5.4_frequency_based_expansion.py",
        "contextual_embedding": "A2.5.5_contextual_embedding_expansion.py"
    }
    
    script_name = strategy_scripts.get(strategy_name)
    if not script_name:
        return None
    
    script_path = script_dir / "scripts" / script_name
    
    if not script_path.exists():
        print(f"    Strategy script not found: {script_path}")
        return None
    
    try:
        # Import and run the strategy module
        spec = importlib.util.spec_from_file_location(strategy_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[strategy_name] = module
        spec.loader.exec_module(module)
        
        # Run the main function if available
        if hasattr(module, 'main'):
            module.main()
        
        # Try to load the results after running
        output_files = {
            "semantic_similarity": "outputs/A2.5.1_semantic_expansion.json",
            "domain_knowledge": "outputs/A2.5.2_domain_expansion.json",
            "hierarchical_clustering": "outputs/A2.5.3_hierarchical_expansion.json",
            "frequency_based": "outputs/A2.5.4_frequency_expansion.json",
            "contextual_embedding": "outputs/A2.5.5_contextual_expansion.json"
        }
        
        output_file = output_files.get(strategy_name)
        if output_file:
            output_path = script_dir / output_file
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
    except Exception as e:
        print(f"    Error running {strategy_name}: {e}")
    
    return None

def combine_expansion_results(strategy_results):
    """
    Combine results from all expansion strategies
    
    Args:
        strategy_results: Results from all strategies
        
    Returns:
        dict: Combined expansion results
    """
    combined_expansions = {}
    
    # Get all concepts that were processed - all strategies now use "expansions"
    all_concept_ids = set()
    for strategy_name, data in strategy_results.items():
        if data and "results" in data and "expansions" in data["results"]:
            expansions = data["results"]["expansions"]
            for expansion in expansions:
                concept_id = expansion.get("concept_id")
                all_concept_ids.add(concept_id)
    
    # Combine expansions for each concept
    for concept_id in all_concept_ids:
        combined_expansions[concept_id] = {
            "concept_id": concept_id,
            "strategy_contributions": {},
            "all_expanded_terms": set(),
            "weighted_terms": defaultdict(float),
            "expansion_scores": {},
            "original_concept": None
        }
        
        # Collect expansions from each strategy
        for strategy_name, data in strategy_results.items():
            weight = STRATEGY_WEIGHTS.get(strategy_name, 0.1)

            if not data or "results" not in data or "expansions" not in data["results"]:
                continue

            # Find this concept's expansion in this strategy
            concept_expansion = None
            for expansion in data["results"]["expansions"]:
                if expansion.get("concept_id") == concept_id:
                    concept_expansion = expansion
                    break

            if not concept_expansion:
                continue
            
            # Store original concept data
            if combined_expansions[concept_id]["original_concept"] is None:
                combined_expansions[concept_id]["original_concept"] = concept_expansion

            # Extract expanded terms - all strategies now use "keywords" field
            strategy_terms = set(concept_expansion.get("keywords", []))
            
            # Add to combined results
            combined_expansions[concept_id]["strategy_contributions"][strategy_name] = {
                "terms": list(strategy_terms),
                "count": len(strategy_terms),
                "weight": weight
            }
            
            combined_expansions[concept_id]["all_expanded_terms"].update(strategy_terms)
            
            # Weight the terms
            for term in strategy_terms:
                combined_expansions[concept_id]["weighted_terms"][term] += weight
    
    # Convert sets to lists and calculate final scores
    for concept_id, expansion_data in combined_expansions.items():
        expansion_data["all_expanded_terms"] = list(expansion_data["all_expanded_terms"])
        expansion_data["weighted_terms"] = dict(expansion_data["weighted_terms"])
        
        # Calculate expansion quality score - handle both field names
        original_concept = expansion_data["original_concept"]
        original_terms = len(original_concept.get("keywords", original_concept.get("primary_keywords", [])))
        expanded_terms = len(expansion_data["all_expanded_terms"])
        expansion_ratio = expanded_terms / max(original_terms, 1)
        
        # Calculate strategy diversity score
        strategy_count = len(expansion_data["strategy_contributions"])
        strategy_diversity = strategy_count / len(STRATEGY_WEIGHTS)
        
        # Calculate average term weight
        avg_term_weight = sum(expansion_data["weighted_terms"].values()) / max(len(expansion_data["weighted_terms"]), 1)
        
        expansion_data["expansion_scores"] = {
            "expansion_ratio": expansion_ratio,
            "strategy_diversity": strategy_diversity,
            "average_term_weight": avg_term_weight,
            "overall_quality": (expansion_ratio * 0.4 + strategy_diversity * 0.3 + avg_term_weight * 0.3)
        }
    
    return combined_expansions

def generate_expansion_summary(combined_expansions, strategy_results):
    """
    Generate summary of expansion results

    Args:
        combined_expansions: Combined expansion results
        strategy_results: Original strategy results

    Returns:
        dict: Expansion summary
    """
    total_concepts = len(combined_expansions)

    # Calculate statistics
    expansion_ratios = [exp["expansion_scores"]["expansion_ratio"] for exp in combined_expansions.values()]
    avg_expansion_ratio = sum(expansion_ratios) / max(len(expansion_ratios), 1)

    # Enhanced strategy coverage with original performance metrics
    strategy_coverage = {}
    strategy_performance = {}

    for strategy_name in STRATEGY_WEIGHTS:
        covered_concepts = len([exp for exp in combined_expansions.values()
                              if strategy_name in exp["strategy_contributions"]])
        strategy_coverage[strategy_name] = {
            "covered_concepts": covered_concepts,
            "coverage_ratio": covered_concepts / max(total_concepts, 1)
        }

        # Extract original strategy performance metrics
        strategy_data = strategy_results.get(strategy_name)
        if strategy_data and "results" in strategy_data:
            results = strategy_data["results"]
            if "statistics" in results:
                stats = results["statistics"]
                strategy_performance[strategy_name] = {
                    "total_concepts": stats.get("total_concepts", 0),
                    "concepts_expanded": stats.get("concepts_expanded", 0),
                    "expansion_coverage": stats.get("expansion_coverage", 0.0),
                    "average_expansion_ratio": stats.get("average_expansion_ratio", 1.0),
                    "total_original_keywords": stats.get("total_original_keywords", 0),
                    "total_expanded_keywords": stats.get("total_expanded_keywords", 0),
                    "processing_time": stats.get("processing_time", 0),
                    "performance_metrics": stats.get("performance_metrics", {})
                }
            else:
                # Fallback for strategies without detailed statistics
                strategy_performance[strategy_name] = {
                    "total_concepts": 0,
                    "concepts_expanded": 0,
                    "expansion_coverage": 0.0,
                    "average_expansion_ratio": 1.0,
                    "total_original_keywords": 0,
                    "total_expanded_keywords": 0,
                    "processing_time": 0,
                    "performance_metrics": {}
                }
        else:
            strategy_performance[strategy_name] = {
                "total_concepts": 0,
                "concepts_expanded": 0,
                "expansion_coverage": 0.0,
                "average_expansion_ratio": 1.0,
                "total_original_keywords": 0,
                "total_expanded_keywords": 0,
                "processing_time": 0,
                "performance_metrics": {}
            }

    # Quality distribution
    quality_scores = [exp["expansion_scores"]["overall_quality"] for exp in combined_expansions.values()]
    high_quality_concepts = len([score for score in quality_scores if score > 0.7])

    # Calculate aggregate performance metrics
    total_processing_time = sum(perf.get("processing_time", 0) for perf in strategy_performance.values())
    total_concepts_expanded_by_strategies = sum(perf.get("concepts_expanded", 0) for perf in strategy_performance.values())

    return {
        "total_concepts": total_concepts,
        "average_expansion_ratio": avg_expansion_ratio,
        "strategy_coverage": strategy_coverage,
        "strategy_performance": strategy_performance,
        "aggregate_metrics": {
            "total_processing_time": total_processing_time,
            "total_strategy_expansions": total_concepts_expanded_by_strategies,
            "average_processing_time_per_strategy": total_processing_time / len(strategy_performance),
        },
        "quality_distribution": {
            "high_quality_concepts": high_quality_concepts,
            "average_quality_score": sum(quality_scores) / max(len(quality_scores), 1)
        }
    }

def save_output(combined_expansions, summary):
    """
    Save orchestrated expansion results
    
    Args:
        combined_expansions: Combined expansion results
        summary: Expansion summary
    """
    script_dir = Path(__file__).parent.parent
    
    # Prepare output data
    output_data = {
        "orchestration_metadata": {
            "strategy_weights": STRATEGY_WEIGHTS,
            "processing_timestamp": datetime.now().isoformat(),
            "total_concepts": len(combined_expansions)
        },
        "expanded_concepts": combined_expansions,
        "expansion_summary": summary
    }
    
    # Save main results
    output_path = script_dir / "outputs/A2.5_expanded_concepts.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved orchestrated results to {output_path}")
    
    # Save summary separately
    summary_path = script_dir / "outputs/A2.5_expansion_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def main():
    """Main execution"""
    print("="*60)
    print("A2.5: Expanded Concepts Orchestrator")
    print("="*60)
    
    try:
        # Load strategy results
        strategy_results = load_strategy_results()
        
        # Check if we have any valid results
        valid_strategies = [name for name, data in strategy_results.items() if data is not None]
        if not valid_strategies:
            print("[WARNING] No valid strategy results found. Run individual A2.5.x scripts first.")
            return
        
        print(f"\nValid strategies loaded: {len(valid_strategies)}")
        for strategy in valid_strategies:
            print(f"  [OK] {strategy}")
        
        # Combine expansion results
        print(f"\nCombining expansion results...")
        combined_expansions = combine_expansion_results(strategy_results)
        
        # Generate summary
        summary = generate_expansion_summary(combined_expansions, strategy_results)
        
        # Display results
        print(f"\nExpansion Orchestration Results:")
        print(f"  Total Concepts: {summary['total_concepts']}")
        print(f"  Average Expansion Ratio: {summary['average_expansion_ratio']:.2f}")
        print(f"  High Quality Concepts: {summary['quality_distribution']['high_quality_concepts']}")
        print(f"  Average Quality Score: {summary['quality_distribution']['average_quality_score']:.3f}")
        
        print(f"\nIndividual Strategy Performance:")
        for strategy, performance in summary["strategy_performance"].items():
            coverage = summary["strategy_coverage"][strategy]
            print(f"  {strategy}:")
            print(f"    Concepts Expanded: {performance['concepts_expanded']}/{performance['total_concepts']}")
            print(f"    Expansion Coverage: {performance['expansion_coverage']:.1%}")
            print(f"    Avg Expansion Ratio: {performance['average_expansion_ratio']:.2f}x")
            print(f"    Processing Time: {performance['processing_time']:.2f}s")
            if performance['performance_metrics'].get('algorithm_complexity'):
                print(f"    Complexity: {performance['performance_metrics']['algorithm_complexity']}")
            print(f"    Orchestrator Coverage: {coverage['covered_concepts']}/{summary['total_concepts']} ({coverage['coverage_ratio']:.1%})")

        print(f"\nAggregate Performance:")
        agg = summary["aggregate_metrics"]
        print(f"  Total Processing Time: {agg['total_processing_time']:.2f}s")
        print(f"  Total Strategy Expansions: {agg['total_strategy_expansions']}")
        print(f"  Average Time per Strategy: {agg['average_processing_time_per_strategy']:.2f}s")
        
        # Show top expanded concepts
        print(f"\nTop Expanded Concepts:")
        sorted_concepts = sorted(combined_expansions.values(), 
                               key=lambda x: x["expansion_scores"]["overall_quality"], 
                               reverse=True)
        
        for i, concept_data in enumerate(sorted_concepts[:5], 1):
            concept = concept_data["original_concept"]
            scores = concept_data["expansion_scores"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Quality: {scores['overall_quality']:.3f}")
            original_keywords = concept.get("keywords", concept.get("primary_keywords", []))
            print(f"     Expansion: {len(original_keywords)} -> {len(concept_data['all_expanded_terms'])} terms")
            print(f"     Strategies: {len(concept_data['strategy_contributions'])}")
        
        # Save results
        save_output(combined_expansions, summary)
        
        print("\nA2.5 Expanded Concepts Orchestrator completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.5 Orchestrator: {str(e)}")
        raise

if __name__ == "__main__":
    main()