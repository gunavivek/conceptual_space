"""
Regenerate Q1 Question Ingestion output in INFERENCE mode
This ensures no answer/response data leakage into downstream Q-Pipeline modules
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(__file__))

from Q1_question_ingestion import Q1_QuestionIngestion


def regenerate_q1_inference_mode():
    """
    Regenerate Q1 output with answer data EXCLUDED to prevent leakage
    """
    print("REGENERATING Q1 IN INFERENCE MODE")
    print("=" * 60)
    print("CRITICAL: This removes answer data to prevent leakage in Q2.1 and downstream modules")

    # Initialize Q1 in INFERENCE mode (excludes answer data)
    q1 = Q1_QuestionIngestion(mode="inference")

    # Process the sample question
    question_id = "finqa_test_1630"
    print(f"\nProcessing question: {question_id}")

    try:
        # Load question WITHOUT answer data
        processed_question = q1.load_question(question_id)

        print(f"\nQuestion data structure:")
        for key in processed_question.keys():
            if key == "question_text":
                print(f"  {key}: {processed_question[key][:60]}...")
            else:
                print(f"  {key}: {processed_question[key]}")

        # Verify NO answer data is present
        forbidden_keys = ['answer', 'response', 'generation_model_name']
        leaked_data = [key for key in processed_question.keys() if key in forbidden_keys]

        if leaked_data:
            print(f"\nDATA LEAKAGE DETECTED: {leaked_data}")
            return False
        else:
            print(f"\nDATA LEAKAGE CHECK PASSED: No answer/response data found")

        # Save with standard naming
        standard_output_path = "../outputs/Q1_Question_ingestion.json"
        q1.save_output(processed_question, standard_output_path)

        print(f"\nQ1 regenerated in inference mode: {standard_output_path}")
        print("Safe for downstream Q-Pipeline processing without data leakage")

        return True

    except Exception as e:
        print(f"Error regenerating Q1: {e}")
        return False


if __name__ == "__main__":
    success = regenerate_q1_inference_mode()

    if success:
        print("\n" + "=" * 60)
        print("Q1 REGENERATION COMPLETE - INFERENCE MODE")
        print("Ready for Q2.1 processing without data leakage")
    else:
        print("\nQ1 regeneration failed")