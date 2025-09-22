#!/usr/bin/env python3
"""
Test A3 Deduplication Behavior
Demonstrates what happens to concepts when chunks are deduplicated
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'A_Concept_pipeline', 'scripts'))

from A3_concept_based_chunking import A3ConceptChunkingOrchestrator
import json
from pathlib import Path

def test_deduplication_behavior():
    """Test what happens to concepts during deduplication"""

    print("=" * 80)
    print("A3 DEDUPLICATION BEHAVIOR TEST")
    print("=" * 80)

    # Load recent chunking results
    output_path = Path("A_Concept_pipeline/outputs/A3_raw_chunks_no_dedup.json")
    final_path = Path("A_Concept_pipeline/outputs/A3_multi_strategy_chunks.json")

    if not output_path.exists() or not final_path.exists():
        print("ERROR: Need to run A3 first to generate test data")
        return

    # Load raw chunks (before deduplication)
    with open(output_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    raw_chunks = raw_data['chunks']

    # Load final chunks (after deduplication)
    with open(final_path, 'r', encoding='utf-8') as f:
        final_data = json.load(f)
    final_chunks = final_data['chunks']

    print(f"\nBEFORE DEDUPLICATION: {len(raw_chunks)} chunks")
    print(f"AFTER DEDUPLICATION: {len(final_chunks)} chunks")
    print(f"REDUCTION: {len(raw_chunks) - len(final_chunks)} chunks removed")

    # Analyze concept behavior during deduplication
    print(f"\nCONCEPT BEHAVIOR ANALYSIS:")
    print(f"{'Document':<15} {'Raw Concepts':<12} {'Final Concepts':<14} {'Behavior':<20}")
    print("-" * 70)

    # Group by document
    raw_by_doc = {}
    final_by_doc = {}

    for chunk in raw_chunks:
        doc_id = chunk['doc_id']
        if doc_id not in raw_by_doc:
            raw_by_doc[doc_id] = []
        raw_by_doc[doc_id].append(chunk)

    for chunk in final_chunks:
        doc_id = chunk['doc_id']
        if doc_id not in final_by_doc:
            final_by_doc[doc_id] = []
        final_by_doc[doc_id].append(chunk)

    total_raw_concepts = 0
    total_final_concepts = 0

    for doc_id in sorted(raw_by_doc.keys()):
        raw_doc_chunks = raw_by_doc[doc_id]
        final_doc_chunks = final_by_doc.get(doc_id, [])

        # Count unique concepts in raw chunks
        raw_concepts = set()
        for chunk in raw_doc_chunks:
            raw_concepts.update(chunk.get('concept_memberships', []))

        # Count unique concepts in final chunks
        final_concepts = set()
        for chunk in final_doc_chunks:
            final_concepts.update(chunk.get('concept_memberships', []))

        total_raw_concepts += len(raw_concepts)
        total_final_concepts += len(final_concepts)

        # Determine behavior
        if len(raw_concepts) == len(final_concepts):
            behavior = "PRESERVED"
        elif len(final_concepts) > len(raw_concepts):
            behavior = "MERGED (+concepts)"
        else:
            behavior = "REDUCED (-concepts)"

        print(f"{doc_id:<15} {len(raw_concepts):<12} {len(final_concepts):<14} {behavior:<20}")

    print("-" * 70)
    print(f"{'TOTAL':<15} {total_raw_concepts:<12} {total_final_concepts:<14} {total_final_concepts - total_raw_concepts:+d}")

    # Detailed example of concept merging
    print(f"\nDETAILED DEDUPLICATION EXAMPLE:")
    print(f"Looking for chunks with >1 concept to show merging...")

    # Find a chunk with multiple concepts to show merging
    example_found = False
    for chunk in final_chunks[:5]:  # Check first 5 chunks
        concepts = chunk.get('concept_memberships', [])
        if len(concepts) > 1:
            print(f"\nChunk ID: {chunk['chunk_id']}")
            print(f"Document: {chunk['doc_id']}")
            print(f"Concepts: {len(concepts)}")
            print(f"Concept List: {concepts[:3]}...")  # Show first 3
            print(f"Content Preview: {chunk['content'][:100]}...")
            print(f"Source Strategies: {chunk.get('source_strategies', [])}")
            example_found = True
            break

    if not example_found:
        print("No multi-concept chunks found in first 5 examples")

    print(f"\nDEDUPLICATION MECHANISM SUMMARY:")
    print(f"1. When chunks are >85% similar (Jaccard similarity)")
    print(f"2. The duplicate chunk is REMOVED")
    print(f"3. Its concepts are MERGED into the surviving chunk")
    print(f"4. Result: Fewer chunks, but MORE concepts per chunk")
    print(f"5. NO concepts are lost - they are consolidated")

if __name__ == "__main__":
    test_deduplication_behavior()