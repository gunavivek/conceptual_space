"""
Full Q-Pipeline Runner for 20 Test Records
Processes questions through all stages: Q1 -> Q2.5 -> Q3.1 -> Q3.2 -> Q3.3 -> Q4 -> Q5
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime
import traceback

# Import all Q-Pipeline modules
sys.path.append('.')
sys.path.append('Q_Question_Pipeline/scripts')

print("="*70)
print("Q-PIPELINE FULL TEST RUN - 20 RECORDS")
print("="*70)
print(f"Start time: {datetime.now()}")
print()

def run_pipeline_for_question(question_id, question_text, doc_id, ground_truth_response=None):
    """Run complete pipeline for a single question."""

    results = {
        'question_id': question_id,
        'question_text': question_text,
        'doc_id': doc_id,
        'stages_completed': [],
        'errors': []
    }

    try:
        print(f"\n{'='*60}")
        print(f"Processing Question: {question_id}")
        print(f"Question: {question_text[:100]}...")
        print(f"{'='*60}")

        # Q1: Question Ingestion (already done via parquet)
        print("\n[Q1] Question already ingested from parquet")
        results['stages_completed'].append('Q1')

        # Q2.1-Q2.5: Question Analysis & Assignment
        print("\n[Q2] Running question analysis pipeline...")
        try:
            # Q2.5: Document-Aware Assignment with geometric filtering
            from Q2_5_document_aware_assignment import Q25_DocumentAwareAssignment
            q25 = Q25_DocumentAwareAssignment()

            # Process with geometric filtering
            assignment_data = q25.process_question_with_geometric_filtering(
                question_id=question_id,
                question_text=question_text,
                doc_id=doc_id
            )

            # Save Q2.5 output
            q25.save_enhanced_results(
                question_id=question_id,
                question_text=question_text,
                assignment_data=assignment_data,
                doc_id=doc_id
            )

            print(f"[Q2.5] Document-aware assignment complete")
            print(f"       - Filtered chunks: {len(assignment_data.get('geometric_filtering', {}).get('filtered_chunks', []))}")
            results['stages_completed'].append('Q2.5')

        except Exception as e:
            print(f"[ERROR] Q2.5 failed: {e}")
            results['errors'].append(f"Q2.5: {str(e)}")
            return results

        # Q3.1: Geometric Filtering (using already filtered chunks from Q2.5)
        print("\n[Q3.1] Running geometric filtering...")
        try:
            # Q3.1 now uses the enhanced Q2.5 output directly
            from Q3_1_semantic_ranking import Q31_SemanticRanking
            q31 = Q31_SemanticRanking()

            # Apply semantic ranking to Q2.5 filtered chunks
            ranked_chunks, ranking_metrics = q31.apply_semantic_ranking(
                question_id=question_id,
                top_k=10
            )

            # Save results
            q31.save_results(question_id, ranked_chunks, ranking_metrics)

            print(f"[Q3.1] Semantic ranking complete: {len(ranked_chunks)} chunks")
            results['stages_completed'].append('Q3.1')

        except Exception as e:
            print(f"[ERROR] Q3.1 failed: {e}")
            results['errors'].append(f"Q3.1: {str(e)}")
            # Try alternative Q3.1 if the semantic ranking fails
            try:
                from Q3_1_geometric_filtering import Q31_GeometricFiltering
                q31_geo = Q31_GeometricFiltering()
                filtered_chunks, filter_metrics = q31_geo.apply_geometric_filter(
                    question_id=question_id,
                    doc_id=doc_id
                )
                q31_geo.save_results(question_id, filtered_chunks, filter_metrics)
                print(f"[Q3.1] Geometric filtering (fallback) complete: {len(filtered_chunks)} chunks")
                results['stages_completed'].append('Q3.1_geometric')
            except Exception as e2:
                print(f"[ERROR] Q3.1 geometric fallback also failed: {e2}")
                return results

        # Q3.2: Semantic Ranking
        print("\n[Q3.2] Running semantic ranking...")
        try:
            from Q3_2_semantic_ranking import Q32_SemanticRanking
            q32 = Q32_SemanticRanking()

            ranked_chunks, ranking_metrics = q32.rank_chunks_semantically(
                question_id=question_id,
                question_text=question_text,
                top_k=10
            )

            q32.save_results(question_id, ranked_chunks, ranking_metrics)

            print(f"[Q3.2] Semantic ranking complete: {len(ranked_chunks)} chunks")
            print(f"       - Avg semantic score: {ranking_metrics.get('avg_semantic_score', 0):.3f}")
            results['stages_completed'].append('Q3.2')

        except Exception as e:
            print(f"[ERROR] Q3.2 failed: {e}")
            results['errors'].append(f"Q3.2: {str(e)}")
            return results

        # Q3.3: Concept Boosting
        print("\n[Q3.3] Running concept boosting...")
        try:
            from Q3_3_concept_boosting import Q33_ConceptBoosting
            q33 = Q33_ConceptBoosting()

            final_chunks, boost_metrics = q33.apply_concept_boosting(
                question_id=question_id,
                question_text=question_text,
                top_k=5
            )

            q33.save_results(question_id, final_chunks, boost_metrics)

            print(f"[Q3.3] Concept boosting complete: {len(final_chunks)} final chunks")
            results['stages_completed'].append('Q3.3')

        except Exception as e:
            print(f"[ERROR] Q3.3 failed: {e}")
            results['errors'].append(f"Q3.3: {str(e)}")
            # Continue without boosting

        # Q4: Answer Generation
        print("\n[Q4] Running answer generation...")
        try:
            from Q4_llm_answer_generation import Q4_AnswerGeneration
            q4 = Q4_AnswerGeneration()

            answer_data = q4.generate_answer(question_id=question_id)

            print(f"[Q4] Answer generated successfully")
            print(f"      - Method: {answer_data.get('generation_method', 'unknown')}")
            print(f"      - Confidence: {answer_data.get('confidence', 0):.2f}")
            results['stages_completed'].append('Q4')
            results['generated_answer'] = answer_data.get('answer', '')

        except Exception as e:
            print(f"[ERROR] Q4 failed: {e}")
            results['errors'].append(f"Q4: {str(e)}")
            return results

        # Q5: Answer Validation (if ground truth available)
        if ground_truth_response:
            print("\n[Q5] Running answer validation...")
            try:
                from Q5_answer_validation import Q5_AnswerValidation
                q5 = Q5_AnswerValidation()

                validation_results = q5.validate_answer(
                    question_id=question_id,
                    ground_truth_response=ground_truth_response
                )

                print(f"[Q5] Validation complete")
                print(f"      - Status: {validation_results.get('validation_status', 'UNKNOWN')}")
                print(f"      - Similarity: {validation_results.get('similarity_score', 0):.3f}")
                results['stages_completed'].append('Q5')
                results['validation'] = validation_results

            except Exception as e:
                print(f"[ERROR] Q5 failed: {e}")
                results['errors'].append(f"Q5: {str(e)}")

        print(f"\n[SUCCESS] Pipeline completed for {question_id}")
        print(f"         Stages completed: {', '.join(results['stages_completed'])}")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed for {question_id}: {e}")
        traceback.print_exc()
        results['errors'].append(f"Critical: {str(e)}")

    return results

def main():
    """Run full pipeline on 20 test records."""

    # Load test data
    print("\nLoading test data...")
    try:
        df = pd.read_parquet('../../sample_20_records.parquet')
        print(f"Loaded {len(df)} test records")
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return

    # Process each question
    all_results = []
    successful = 0
    failed = 0

    for idx, row in df.iterrows():
        question_id = row.get('id', f'q_{idx}')  # Changed from 'question_id' to 'id'
        question_text = row.get('question', '')
        doc_id = question_id  # Use question_id as doc_id (they match in this dataset)
        ground_truth = row.get('response', None)

        if not question_text:
            print(f"\nSkipping row {idx}: No question text")
            continue

        print(f"\n\n{'#'*70}")
        print(f"PROCESSING RECORD {idx+1}/{len(df)}")
        print(f"{'#'*70}")

        result = run_pipeline_for_question(
            question_id=question_id,
            question_text=question_text,
            doc_id=doc_id,
            ground_truth_response=ground_truth
        )

        all_results.append(result)

        if result['errors']:
            failed += 1
        else:
            successful += 1

        # Save intermediate results
        with open('Q_Question_Pipeline/outputs/pipeline_run_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    print("\n\n" + "="*70)
    print("PIPELINE RUN COMPLETE")
    print("="*70)
    print(f"Total records: {len(df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(df)*100:.1f}%")

    # Analyze stage completion
    stage_stats = {}
    for result in all_results:
        for stage in result['stages_completed']:
            stage_stats[stage] = stage_stats.get(stage, 0) + 1

    print("\nStage Completion Stats:")
    for stage in ['Q1', 'Q2.5', 'Q3.1', 'Q3.2', 'Q3.3', 'Q4', 'Q5']:
        count = stage_stats.get(stage, 0)
        print(f"  {stage}: {count}/{len(all_results)} ({count/len(all_results)*100:.1f}%)")

    # Save final summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df),
        'successful': successful,
        'failed': failed,
        'success_rate': successful/len(df)*100 if len(df) > 0 else 0,
        'stage_stats': stage_stats,
        'results': all_results
    }

    with open('Q_Question_Pipeline/outputs/pipeline_run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to Q_Question_Pipeline/outputs/pipeline_run_summary.json")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()