#!/usr/bin/env python3
"""
Test A3 Strategy Weights Impact
Compare performance with and without strategy weights
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'A_Concept_pipeline', 'scripts'))

from A3_concept_based_chunking import A3ConceptChunkingOrchestrator
import json
from pathlib import Path

def run_comparison_test():
    """Compare A3 performance with and without strategy weights"""

    print("=" * 80)
    print("A3 STRATEGY WEIGHTS COMPARISON TEST")
    print("=" * 80)

    base_path = Path("A_Concept_pipeline/outputs")

    # Test 1: WITH strategy weights (current default)
    print("\n1. RUNNING A3 WITH STRATEGY WEIGHTS...")
    orchestrator_weighted = A3ConceptChunkingOrchestrator(use_strategy_weights=True)

    # Run strategies
    concepts = orchestrator_weighted.load_concepts()
    documents = orchestrator_weighted.load_documents()

    if not documents:
        print("No documents found. Make sure A2.1 has been run.")
        return

    # Run all strategies with weights
    strategy_chunks_weighted = {}
    for strategy_name in orchestrator_weighted.enabled_strategies:
        chunks = orchestrator_weighted.run_strategy(strategy_name, documents, concepts)
        strategy_chunks_weighted[strategy_name] = chunks

    # Aggregate with weights
    weighted_result = orchestrator_weighted.aggregate_chunks(strategy_chunks_weighted)

    # Save weighted results
    weighted_output = base_path / "A3_weighted_comparison.json"
    output_data_weighted = {
        'chunks': weighted_result,
        'total_chunks': len(weighted_result),
        'configuration': 'WITH strategy weights',
        'strategy_weights': orchestrator_weighted.strategy_weights
    }

    with open(weighted_output, 'w', encoding='utf-8') as f:
        json.dump(output_data_weighted, f, indent=2, ensure_ascii=False)

    print(f"WITH weights: {len(weighted_result)} final chunks")

    # Test 2: WITHOUT strategy weights (equal weights)
    print("\n2. RUNNING A3 WITHOUT STRATEGY WEIGHTS (EQUAL WEIGHTS)...")
    orchestrator_equal = A3ConceptChunkingOrchestrator(use_strategy_weights=False)

    # Run all strategies with equal weights
    strategy_chunks_equal = {}
    for strategy_name in orchestrator_equal.enabled_strategies:
        chunks = orchestrator_equal.run_strategy(strategy_name, documents, concepts)
        strategy_chunks_equal[strategy_name] = chunks

    # Aggregate with equal weights
    equal_result = orchestrator_equal.aggregate_chunks(strategy_chunks_equal)

    # Save equal weights results
    equal_output = base_path / "A3_equal_weights_comparison.json"
    output_data_equal = {
        'chunks': equal_result,
        'total_chunks': len(equal_result),
        'configuration': 'WITHOUT strategy weights (all equal)',
        'strategy_weights': orchestrator_equal.strategy_weights
    }

    with open(equal_output, 'w', encoding='utf-8') as f:
        json.dump(output_data_equal, f, indent=2, ensure_ascii=False)

    print(f"WITHOUT weights: {len(equal_result)} final chunks")

    # Analysis
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)

    # Strategy contribution comparison
    from collections import Counter

    weighted_strategies = Counter(c['chunk_type'] for c in weighted_result)
    equal_strategies = Counter(c['chunk_type'] for c in equal_result)

    print(f"\nSTRATEGY CONTRIBUTION COMPARISON:")
    print(f"{'Strategy':<20} {'With Weights':<15} {'Equal Weights':<15} {'Difference':<15}")
    print("-" * 70)

    all_strategies = set(weighted_strategies.keys()) | set(equal_strategies.keys())
    for strategy in sorted(all_strategies):
        weighted_count = weighted_strategies.get(strategy, 0)
        equal_count = equal_strategies.get(strategy, 0)
        difference = equal_count - weighted_count
        sign = "+" if difference > 0 else ""
        print(f"{strategy:<20} {weighted_count:<15} {equal_count:<15} {sign}{difference:<15}")

    print(f"\nTOTAL CHUNKS:")
    print(f"  With weights: {len(weighted_result)}")
    print(f"  Equal weights: {len(equal_result)}")
    print(f"  Difference: {len(equal_result) - len(weighted_result)}")

    # Concept analysis
    weighted_concepts = sum(len(c.get('concept_memberships', [])) for c in weighted_result)
    equal_concepts = sum(len(c.get('concept_memberships', [])) for c in equal_result)

    print(f"\nCONCEPT MEMBERSHIPS:")
    print(f"  With weights: {weighted_concepts} total ({weighted_concepts/len(weighted_result):.2f} avg)")
    print(f"  Equal weights: {equal_concepts} total ({equal_concepts/len(equal_result):.2f} avg)")

    print(f"\nWHAT STRATEGY WEIGHTS CONTROL:")
    print(f"  - Deduplication preference: Higher weight chunks kept when >85% similar")
    print(f"  - Quality ranking: Better strategies prioritized in final selection")
    print(f"  - NOT creation: All strategies still run and create initial chunks")

    print(f"\nCONCLUSION:")
    if len(weighted_result) != len(equal_result):
        print(f"  Strategy weights DO impact final chunk selection")
        print(f"  Different weights cause different deduplication outcomes")
    else:
        print(f"  Strategy weights have MINIMAL impact on final chunk count")
        print(f"  Document-aware deduplication may be the primary factor")

if __name__ == "__main__":
    run_comparison_test()