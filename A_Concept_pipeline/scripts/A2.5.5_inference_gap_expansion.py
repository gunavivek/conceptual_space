#!/usr/bin/env python3
"""
A2.5.5: Advanced Inference Gap Identification and Filling
Identifies and fills conceptual gaps, implicit concepts, and logical inference bridges
Implements gap detection, implicit concept discovery, and conceptual bridge filling
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import time

# Add expansion_modules to path
sys.path.append(str(Path(__file__).parent.parent / "expansion_modules"))

# Try to import both engines
USE_SEMANTIC_ENGINE = True
try:
    from inference_engine_semantic import SemanticInferenceEngine
    from embedding_manager import EmbeddingManager
    print("[INFO] Semantic inference engine available - using optimized O(n log n) implementation")
except ImportError:
    USE_SEMANTIC_ENGINE = False
    print("[WARNING] Semantic engine not available, falling back to original O(n²) implementation")

try:
    from inference_engine import InferenceEngine
except ImportError:
    if not USE_SEMANTIC_ENGINE:
        print("[ERROR] Neither semantic nor original InferenceEngine available.")
        sys.exit(1)

def expand_concept_with_inference_gaps(concept, all_concepts, inference_engine):
    """
    Expand a concept by filling inference gaps and discovering implicit concepts

    Args:
        concept: Target concept to expand
        all_concepts: All available concepts for gap analysis
        inference_engine: InferenceEngine instance

    Returns:
        dict: Expanded concept with gap-filling terms
    """
    # Detect conceptual gaps around this concept
    concept_gaps = inference_engine.detect_concept_specific_gaps(concept, all_concepts)

    # Fill gaps and generate bridge concepts
    gap_fills = inference_engine.fill_concept_gaps(concept, concept_gaps)

    # Extract expansion terms from gap filling
    gap_expansion_terms = []
    for bridge in gap_fills.get("bridge_concepts", []):
        for keyword in bridge.get("keywords", []):
            gap_expansion_terms.append({
                "term": keyword,
                "source": "gap_filling",
                "bridge_type": bridge.get("bridge_type", "unknown"),
                "confidence": bridge.get("confidence", 0.5),
                "gap_type": bridge.get("concept_type", "inferred")
            })

    # Add relationship bridge terms
    for relationship in gap_fills.get("filled_gaps", {}).get("relationship_bridges", []):
        gap_expansion_terms.append({
            "term": f"related_to_{relationship['concept2']}",
            "source": "relationship_bridge",
            "bridge_type": "relationship",
            "confidence": relationship.get("confidence", 0.6),
            "gap_type": "relationship_inference"
        })

    # Add expansion terms to keywords
    original_keywords = concept.get("keywords", [])
    expanded_keywords = original_keywords.copy()

    for expansion in gap_expansion_terms:
        expanded_keywords.append(expansion["term"])

    # Create expanded concept
    expanded_concept = concept.copy()
    expanded_concept["keywords"] = expanded_keywords
    expanded_concept["expansion_metadata"] = {
        "strategy": "inference_gaps",
        "original_keyword_count": len(original_keywords),
        "expanded_keyword_count": len(expanded_keywords),
        "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
        "gap_expansions": len(gap_expansion_terms),
        "detected_gaps": concept_gaps,
        "gap_fills": gap_fills,
        "expansion_terms": gap_expansion_terms,
        "inference_engine_info": inference_engine.get_inference_info()
    }

    return expanded_concept

def process_inference_gap_expansion(core_concepts):
    """
    Process advanced inference gap expansion for all concepts

    Args:
        core_concepts: List of core concepts from A2.4

    Returns:
        dict: Inference gap expansion results
    """
    start_time = time.time()

    # Choose and initialize inference engine
    if USE_SEMANTIC_ENGINE:
        print("  Initializing semantic inference engine (O(n log n) complexity)...")

        # Initialize embedding manager
        embedding_manager = EmbeddingManager(model_type='sentence_transformer')

        # Configure semantic engine
        config = {
            'clustering_method': 'auto',
            'n_clusters': 'auto',
            'min_gap_distance': 0.3,
            'bridge_confidence_threshold': 0.6,
            'dimensionality_reduction': True,
            'batch_size': 50,
            'verbose': False  # Reduce verbosity for cleaner output
        }

        inference_engine = SemanticInferenceEngine(embedding_manager, config)

        # Use semantic gap detection (much faster)
        print("  Performing semantic gap detection...")
        result = inference_engine.detect_conceptual_gaps(core_concepts)

        global_gaps = result.get('gaps', [])
        bridges = result.get('bridges', [])
        clusters = result.get('clusters', [])

        # Format for compatibility
        global_gap_fills = {
            'bridge_concepts': bridges,
            'filled_gaps': {
                'intermediate_bridges': [g for g in global_gaps if g['type'] == 'inter_cluster'],
                'relationship_bridges': [g for g in global_gaps if g['type'] == 'density']
            }
        }

        # Convert to expected format
        formatted_gaps = {
            'missing_intermediates': [g for g in global_gaps if g['type'] == 'inter_cluster'],
            'missing_generalizations': [],
            'missing_specializations': [],
            'missing_relationships': [g for g in global_gaps if g['type'] == 'density'],
            'orphaned_concepts': result.get('isolated_concepts', [])
        }

        global_gaps = formatted_gaps

    else:
        print("  Initializing original inference engine (O(n²) complexity)...")
        inference_engine = InferenceEngine(gap_threshold=0.3, bridge_confidence=0.6)

        # Global gap detection across all concepts
        print("  Detecting global conceptual gaps...")
        global_gaps = inference_engine.detect_conceptual_gaps(core_concepts)

        # Fill global gaps
        print("  Filling global conceptual gaps...")
        global_gap_fills = inference_engine.fill_conceptual_gaps(core_concepts, global_gaps)

    expanded_concepts = []
    expansion_stats = {
        "total_concepts": len(core_concepts),
        "concepts_expanded": 0,
        "total_original_keywords": 0,
        "total_expanded_keywords": 0,
        "expansion_ratios": [],
        "gap_type_totals": {
            "gap_filling": 0,
            "relationship_bridge": 0,
            "implicit_concept": 0,
            "bridge_concept": 0
        },
        "total_gap_expansions": 0,
        "global_gaps": global_gaps,
        "global_gap_fills": global_gap_fills
    }

    print("  Computing inference gap expansions...")

    if USE_SEMANTIC_ENGINE:
        # For semantic engine, we process concepts more efficiently
        for i, concept in enumerate(core_concepts):
            # Use cached gap information
            concept_gaps = inference_engine.detect_concept_specific_gaps(concept, core_concepts)
            gap_fills = inference_engine.fill_concept_gaps(concept, concept_gaps)

            # Create expanded concept
            original_keywords = concept.get("keywords", [])
            expansion_terms = []

            # Add bridge concept keywords
            for bridge in gap_fills.get("bridge_concepts", []):
                for keyword in bridge.get("keywords", []):
                    expansion_terms.append({
                        "term": keyword,
                        "source": "gap_filling",
                        "bridge_type": bridge.get("bridge_type", "semantic"),
                        "confidence": bridge.get("confidence", 0.5),
                        "gap_type": bridge.get("concept_type", "inferred")
                    })

            expanded_keywords = original_keywords.copy()
            for expansion in expansion_terms:
                if expansion["term"] not in expanded_keywords:
                    expanded_keywords.append(expansion["term"])

            expanded_concept = concept.copy()
            expanded_concept["keywords"] = expanded_keywords
            expanded_concept["expansion_metadata"] = {
                "strategy": "inference_gaps",
                "original_keyword_count": len(original_keywords),
                "expanded_keyword_count": len(expanded_keywords),
                "expansion_ratio": len(expanded_keywords) / max(len(original_keywords), 1),
                "gap_expansions": len(expansion_terms),
                "detected_gaps": concept_gaps,
                "gap_fills": gap_fills,
                "expansion_terms": expansion_terms,
                "inference_engine_info": inference_engine.get_inference_info()
            }

            expanded_concepts.append(expanded_concept)

            # Update statistics
            metadata = expanded_concept["expansion_metadata"]
            expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
            expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
            expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])
            expansion_stats["total_gap_expansions"] += metadata["gap_expansions"]

            for expansion in metadata["expansion_terms"]:
                source = expansion.get("source", "unknown")
                if source in expansion_stats["gap_type_totals"]:
                    expansion_stats["gap_type_totals"][source] += 1

            if metadata["expansion_ratio"] > 1.0:
                expansion_stats["concepts_expanded"] += 1

            # Progress indicator for large datasets
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(core_concepts)} concepts...")

    else:
        # Original implementation
        for concept in core_concepts:
            # Expand the concept
            expanded_concept = expand_concept_with_inference_gaps(
                concept, core_concepts, inference_engine
            )
            expanded_concepts.append(expanded_concept)

            # Update statistics
            metadata = expanded_concept["expansion_metadata"]
            expansion_stats["total_original_keywords"] += metadata["original_keyword_count"]
            expansion_stats["total_expanded_keywords"] += metadata["expanded_keyword_count"]
            expansion_stats["expansion_ratios"].append(metadata["expansion_ratio"])
            expansion_stats["total_gap_expansions"] += metadata["gap_expansions"]

            # Update gap type statistics
            for expansion in metadata["expansion_terms"]:
                source = expansion.get("source", "unknown")
                if source in expansion_stats["gap_type_totals"]:
                    expansion_stats["gap_type_totals"][source] += 1

            if metadata["expansion_ratio"] > 1.0:
                expansion_stats["concepts_expanded"] += 1

    # Calculate overall statistics
    expansion_stats["average_expansion_ratio"] = sum(expansion_stats["expansion_ratios"]) / len(expansion_stats["expansion_ratios"]) if expansion_stats["expansion_ratios"] else 0
    expansion_stats["expansion_coverage"] = expansion_stats["concepts_expanded"] / expansion_stats["total_concepts"]
    expansion_stats["avg_gap_expansions"] = expansion_stats["total_gap_expansions"] / len(core_concepts)

    # Global gap statistics
    expansion_stats["global_gap_statistics"] = {
        "total_gaps_detected": sum(len(gaps) for gaps in global_gaps.values()) if isinstance(global_gaps, dict) else len(global_gaps),
        "gaps_filled": len(global_gap_fills.get("bridge_concepts", [])),
        "gap_types": {gap_type: len(gaps) for gap_type, gaps in global_gaps.items()} if isinstance(global_gaps, dict) else {}
    }

    # Add performance metrics
    processing_time = time.time() - start_time
    expansion_stats["processing_time"] = processing_time
    expansion_stats["performance_metrics"] = {
        "total_time_seconds": processing_time,
        "time_per_concept": processing_time / len(core_concepts) if core_concepts else 0,
        "algorithm_complexity": "O(n log n)" if USE_SEMANTIC_ENGINE else "O(n²-n³)",
        "engine_type": "semantic_embedding" if USE_SEMANTIC_ENGINE else "pairwise_comparison"
    }

    return {
        "strategy": "inference_gaps",
        "expansions": expanded_concepts,
        "statistics": expansion_stats,
        "engine_info": inference_engine.get_inference_info()
    }

def load_input(input_path="outputs/A2.4_core_concepts.json"):
    """Load core concepts from A2.4"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path

    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")

    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Main execution"""
    print("="*60)
    print("A2.5.5: Advanced Inference Gap Identification and Filling")
    print("="*60)

    try:
        # Load core concepts
        print("Loading core concepts...")
        input_data = load_input()

        # Extract core concepts from document structure
        core_concepts = []
        for doc in input_data.get("documents", []):
            core_concepts.extend(doc.get("core_concepts", []))

        print(f"Processing inference gap expansion for {len(core_concepts)} concepts...")

        # Process inference gap expansion
        expansion_results = process_inference_gap_expansion(core_concepts)

        # Display results
        stats = expansion_results["statistics"]
        engine_info = expansion_results["engine_info"]
        global_gap_stats = stats["global_gap_statistics"]

        print(f"\nAdvanced Inference Gap Results:")
        print(f"  Concepts Processed: {stats['total_concepts']}")
        print(f"  Concepts Expanded: {stats['concepts_expanded']}")
        print(f"  Expansion Coverage: {stats['expansion_coverage']:.1%}")
        print(f"  Original Keywords: {stats['total_original_keywords']}")
        print(f"  Expanded Keywords: {stats['total_expanded_keywords']}")
        print(f"  Average Expansion Ratio: {stats['average_expansion_ratio']:.2f}")
        print(f"  Avg Gap Expansions: {stats['avg_gap_expansions']:.1f}")

        print(f"\nGap Type Statistics:")
        for gap_type, count in stats["gap_type_totals"].items():
            print(f"  {gap_type.replace('_', ' ').title()}: {count}")

        print(f"\nGlobal Gap Analysis:")
        print(f"  Total Gaps Detected: {global_gap_stats['total_gaps_detected']}")
        print(f"  Gaps Filled: {global_gap_stats['gaps_filled']}")
        print(f"  Gap Types:")
        for gap_type, count in global_gap_stats["gap_types"].items():
            print(f"    {gap_type.replace('_', ' ').title()}: {count}")

        print(f"\nInference Engine Info:")
        if USE_SEMANTIC_ENGINE:
            print(f"  Engine Type: {engine_info.get('engine_type', 'semantic_embedding')}")
            print(f"  Algorithm Complexity: {engine_info.get('complexity', 'O(n log n)')}")
            print(f"  Clustering Method: {engine_info.get('clustering_method', 'auto')}")
        print(f"  Gap Threshold: {engine_info.get('gap_threshold', 'N/A')}")
        print(f"  Bridge Confidence: {engine_info.get('bridge_confidence', 'N/A')}")
        if not USE_SEMANTIC_ENGINE:
            print(f"  Logical Patterns: {engine_info.get('logical_patterns_count', 0)}")
            print(f"  Inference Rules: {engine_info.get('inference_rules_count', 0)}")

        # Show performance metrics
        perf_metrics = stats.get("performance_metrics", {})
        if perf_metrics:
            print(f"\nPerformance Metrics:")
            print(f"  Total Processing Time: {perf_metrics['total_time_seconds']:.2f} seconds")
            print(f"  Time per Concept: {perf_metrics['time_per_concept']:.4f} seconds")
            print(f"  Algorithm Complexity: {perf_metrics['algorithm_complexity']}")
            print(f"  Engine Type: {perf_metrics['engine_type']}")

            # Estimate performance improvement
            if USE_SEMANTIC_ENGINE and stats['total_concepts'] > 50:
                estimated_old_time = (stats['total_concepts'] ** 2) * 0.001  # Rough estimate for O(n²)
                speedup = estimated_old_time / perf_metrics['total_time_seconds']
                print(f"  Estimated Speedup: {speedup:.1f}x faster than O(n²) implementation")

        # Show sample expansions
        print(f"\nSample Inference Gap Expansions:")
        for i, concept in enumerate(expansion_results["expansions"][:3], 1):
            metadata = concept["expansion_metadata"]
            print(f"  {i}. {concept.get('canonical_name', concept.get('concept_id', 'Unknown'))}")
            print(f"     Expansion: {metadata['original_keyword_count']} -> {metadata['expanded_keyword_count']} keywords")
            print(f"     Ratio: {metadata['expansion_ratio']:.2f}")
            print(f"     Gap Expansions: {metadata['gap_expansions']}")

        # Save results for A2.5 orchestrator
        output_data = {
            "strategy_name": "inference_gaps",
            "results": expansion_results,
            "processing_timestamp": datetime.now().isoformat()
        }

        output_path = Path(__file__).parent.parent / "outputs/A2.5.5_inference_gap_expansion.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {output_path}")
        print("\nA2.5.5 Advanced Inference Gap Identification and Filling completed successfully!")

    except Exception as e:
        print(f"Error in A2.5.5: {str(e)}")
        raise

if __name__ == "__main__":
    main()