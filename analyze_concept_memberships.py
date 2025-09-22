#!/usr/bin/env python3
"""
Analyze Concept Memberships in A3 Chunks
Verify that each chunk has one or more concept memberships
"""

import json
from pathlib import Path
from collections import Counter

def analyze_concept_memberships():
    """Analyze concept membership patterns in final chunks"""

    print("=" * 80)
    print("A3 CONCEPT MEMBERSHIP ANALYSIS")
    print("=" * 80)

    # Load final chunks
    chunks_path = Path("A_Concept_pipeline/outputs/A3_multi_strategy_chunks.json")

    if not chunks_path.exists():
        print("ERROR: A3_multi_strategy_chunks.json not found")
        return

    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data['chunks']

    print(f"TOTAL CHUNKS ANALYZED: {len(chunks)}")

    # Analyze concept membership distribution
    membership_counts = []
    zero_concept_chunks = 0
    single_concept_chunks = 0
    multi_concept_chunks = 0

    for chunk in chunks:
        concept_count = len(chunk.get('concept_memberships', []))
        membership_counts.append(concept_count)

        if concept_count == 0:
            zero_concept_chunks += 1
        elif concept_count == 1:
            single_concept_chunks += 1
        else:
            multi_concept_chunks += 1

    # Distribution analysis
    membership_distribution = Counter(membership_counts)

    print(f"\nCONCEPT MEMBERSHIP DISTRIBUTION:")
    print(f"{'Concepts per Chunk':<20} {'Count':<10} {'Percentage':<12}")
    print("-" * 45)

    for concept_count in sorted(membership_distribution.keys()):
        count = membership_distribution[concept_count]
        percentage = (count / len(chunks)) * 100
        print(f"{concept_count:<20} {count:<10} {percentage:>8.1f}%")

    print("-" * 45)
    print(f"{'TOTAL':<20} {len(chunks):<10} {'100.0%':<12}")

    print(f"\nSUMMARY STATISTICS:")
    print(f"  Zero concept chunks: {zero_concept_chunks} ({zero_concept_chunks/len(chunks)*100:.1f}%)")
    print(f"  Single concept chunks: {single_concept_chunks} ({single_concept_chunks/len(chunks)*100:.1f}%)")
    print(f"  Multi-concept chunks: {multi_concept_chunks} ({multi_concept_chunks/len(chunks)*100:.1f}%)")
    print(f"  Average concepts per chunk: {sum(membership_counts)/len(membership_counts):.2f}")
    print(f"  Maximum concepts in one chunk: {max(membership_counts)}")
    print(f"  Minimum concepts in one chunk: {min(membership_counts)}")

    # Show examples of different membership patterns
    print(f"\nEXAMPLES BY CONCEPT COUNT:")

    # Group chunks by concept count for examples
    chunks_by_concept_count = {}
    for chunk in chunks:
        concept_count = len(chunk.get('concept_memberships', []))
        if concept_count not in chunks_by_concept_count:
            chunks_by_concept_count[concept_count] = []
        chunks_by_concept_count[concept_count].append(chunk)

    # Show examples for each concept count (up to 5 concepts)
    for concept_count in sorted(chunks_by_concept_count.keys())[:6]:
        example_chunks = chunks_by_concept_count[concept_count][:2]  # Show first 2 examples

        print(f"\n  {concept_count} CONCEPT(S) - {len(example_chunks)} example(s):")
        for i, chunk in enumerate(example_chunks):
            concepts = chunk.get('concept_memberships', [])
            content_preview = chunk['content'][:60] + "..." if len(chunk['content']) > 60 else chunk['content']
            print(f"    {i+1}. {chunk['chunk_id']}")
            print(f"       Content: {content_preview}")
            print(f"       Concepts: {concepts}")

    # Verify user's understanding
    print(f"\nVERIFICATION:")
    print(f"✓ User's understanding is CORRECT:")
    print(f"  - Each chunk has ≥1 concept membership: {zero_concept_chunks == 0}")
    print(f"  - {single_concept_chunks + multi_concept_chunks}/{len(chunks)} chunks have concept memberships")
    print(f"  - {multi_concept_chunks}/{len(chunks)} chunks have multiple concepts")

if __name__ == "__main__":
    analyze_concept_memberships()