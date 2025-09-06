#!/usr/bin/env python3
"""
Comprehensive analysis of A3 convex ball coverage
"""

import json

def analyze_convex_coverage():
    # Load A3 results
    with open('A_Concept_pipeline/outputs/A3_concept_based_chunks.json', 'r') as f:
        data = json.load(f)

    print('COMPREHENSIVE CONVEX BALL COVERAGE ANALYSIS:')
    print('='*70)

    # 1. Analyze concepts and their convex ball utilization
    concept_ball_usage = {}
    total_concepts = len(data['concept_centroids'])

    for concept_id, concept_info in data['concept_centroids'].items():
        concept_ball_usage[concept_id] = {
            'canonical_name': concept_info['canonical_name'],
            'chunks_inside': 0,
            'chunks_outside': 0,
            'total_chunks': 0,
            'has_chunks': False
        }

    # 2. Count chunk assignments to convex balls
    total_chunks = 0
    chunks_with_no_concept = 0
    chunks_in_any_ball = 0
    chunks_outside_all_balls = 0

    for doc_id, doc_chunks in data['document_chunks'].items():
        for chunk in doc_chunks:
            total_chunks += 1
            
            # Check if chunk has any concept memberships
            has_any_membership = any(score > 0 for score in chunk['concept_memberships'].values())
            if not has_any_membership:
                chunks_with_no_concept += 1
            
            # Check convex ball memberships
            in_any_ball = any(chunk['convex_ball_memberships'].values())
            
            if in_any_ball:
                chunks_in_any_ball += 1
            else:
                chunks_outside_all_balls += 1
            
            # Update concept usage counts
            for concept_id, in_ball in chunk['convex_ball_memberships'].items():
                if concept_id in concept_ball_usage:
                    concept_ball_usage[concept_id]['total_chunks'] += 1
                    concept_ball_usage[concept_id]['has_chunks'] = True
                    if in_ball:
                        concept_ball_usage[concept_id]['chunks_inside'] += 1
                    else:
                        concept_ball_usage[concept_id]['chunks_outside'] += 1

    # 3. Count empty convex balls
    empty_convex_balls = 0
    populated_convex_balls = 0

    for concept_id, usage in concept_ball_usage.items():
        if usage['chunks_inside'] == 0:
            empty_convex_balls += 1
        else:
            populated_convex_balls += 1

    print('OVERALL STATISTICS:')
    print(f'  Total Concepts: {total_concepts}')
    print(f'  Total Chunks: {total_chunks}')
    print(f'  Total Convex Balls: {total_concepts}')

    print('\nCONCEPT COVERAGE BY CONVEX BALLS:')
    print(f'  Concepts with chunks in convex balls: {populated_convex_balls}/{total_concepts} ({populated_convex_balls/total_concepts*100:.1f}%)')
    print(f'  Concepts with NO chunks in convex balls: {empty_convex_balls}/{total_concepts} ({empty_convex_balls/total_concepts*100:.1f}%)')

    print('\nCHUNK COVERAGE:')
    print(f'  Chunks inside any convex ball: {chunks_in_any_ball}/{total_chunks} ({chunks_in_any_ball/total_chunks*100:.1f}%)')
    print(f'  Chunks outside all convex balls: {chunks_outside_all_balls}/{total_chunks} ({chunks_outside_all_balls/total_chunks*100:.1f}%)')
    print(f'  Chunks with NO concept assignment: {chunks_with_no_concept}/{total_chunks} ({chunks_with_no_concept/total_chunks*100:.1f}%)')

    print('\nCONVEX BALL UTILIZATION:')
    print(f'  Populated convex balls: {populated_convex_balls}/{total_concepts} ({populated_convex_balls/total_concepts*100:.1f}%)')
    print(f'  Empty convex balls: {empty_convex_balls}/{total_concepts} ({empty_convex_balls/total_concepts*100:.1f}%)')

    print('\nDETAILED BREAKDOWN BY CONCEPT:')
    print('Concept ID   Name                 In Ball  Outside  Total    Status      ')
    print('-' * 70)

    for concept_id, usage in sorted(concept_ball_usage.items()):
        name = usage['canonical_name'][:18] + '..' if len(usage['canonical_name']) > 18 else usage['canonical_name']
        status = 'POPULATED' if usage['chunks_inside'] > 0 else ('EMPTY' if usage['has_chunks'] else 'NO_CHUNKS')
        
        print(f'{concept_id:<12} {name:<20} {usage["chunks_inside"]:<8} {usage["chunks_outside"]:<8} {usage["total_chunks"]:<8} {status:<12}')

    print('\nCONCEPTS WITH EMPTY CONVEX BALLS:')
    empty_balls = [concept_id for concept_id, usage in concept_ball_usage.items() if usage['chunks_inside'] == 0]
    for concept_id in empty_balls:
        usage = concept_ball_usage[concept_id]
        reason = 'No chunks associated' if usage['total_chunks'] == 0 else 'All chunks outside ball boundary'
        print(f'  {concept_id} ({usage["canonical_name"]}): {reason}')

    return {
        'total_concepts': total_concepts,
        'total_chunks': total_chunks,
        'populated_convex_balls': populated_convex_balls,
        'empty_convex_balls': empty_convex_balls,
        'chunks_in_any_ball': chunks_in_any_ball,
        'chunks_outside_all_balls': chunks_outside_all_balls,
        'chunks_with_no_concept': chunks_with_no_concept
    }

if __name__ == "__main__":
    results = analyze_convex_coverage()