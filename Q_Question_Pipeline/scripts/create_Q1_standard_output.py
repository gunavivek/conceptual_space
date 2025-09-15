"""
Create Q1 Standard Output File
Generates Q1_Question_ingestion.json with proper naming convention
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(__file__))

from Q1_question_ingestion import Q1_QuestionIngestion


def create_q1_standard_output():
    """
    Create the standard Q1_Question_ingestion.json output file
    """
    print("Creating Q1_Question_ingestion.json with standardized naming...")

    # Initialize Q1 with A-Pipeline data
    q1 = Q1_QuestionIngestion()

    # Load the first question as example
    try:
        import pandas as pd
        data_path = "../../A_Concept_pipeline/data/sample_20_records.parquet"
        df = pd.read_parquet(data_path)

        # Get first question
        first_question_id = df['id'].iloc[0]
        print(f"Processing sample question: {first_question_id}")

        # Process through Q1
        processed_question = q1.load_question(first_question_id)

        print(f"Question processed successfully:")
        print(f"  Question ID: {processed_question['question_id']}")
        print(f"  Doc ID: {processed_question['doc_id']}")
        print(f"  Question: {processed_question['question_text'][:60]}...")
        print(f"  Pipeline Ready: {processed_question['pipeline_ready']}")

        # Save with standard naming
        standard_output_path = "../outputs/Q1_Question_ingestion.json"
        q1.save_output(processed_question, standard_output_path)

        print(f"\n✅ Standard Q1 output created at: {standard_output_path}")

        return processed_question

    except Exception as e:
        print(f"Error creating Q1 standard output: {e}")
        return None


if __name__ == "__main__":
    print("Q1 STANDARD OUTPUT FILE CREATION")
    print("=" * 50)

    result = create_q1_standard_output()

    if result:
        print("✅ Q1_Question_ingestion.json created successfully")
        print("Ready for Q2.5 integration with standardized naming")
    else:
        print("❌ Failed to create standard output file")