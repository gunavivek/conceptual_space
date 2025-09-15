"""
Q1: Question Ingestion Module
Loads questions with critical doc_id linkage for concept space alignment
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class Q1_QuestionIngestion:
    """
    Question ingestion with document association for concept space mapping.
    Critical: Each question MUST be linked to its source document for
    coordinate system alignment.
    """

    def __init__(self, data_path: str = "../../A_Concept_pipeline/data/sample_20_records.parquet", mode: str = "inference"):
        """
        Initialize question ingestion module.

        Args:
            data_path: Path to question data (parquet or json)
            mode: "inference" (no answer data) or "evaluation" (with answer data for validation)
        """
        self.data_path = data_path
        self.mode = mode  # CRITICAL: Controls answer data inclusion
        if mode not in ["inference", "evaluation"]:
            raise ValueError("Mode must be 'inference' or 'evaluation'")

        self.questions_cache = {}

        print(f"Q1 initialized in {mode} mode - {'EXCLUDING' if mode == 'inference' else 'INCLUDING'} answer data")

    def load_question(self, question_id: str) -> Dict:
        """
        Load a single question with its document association.

        Args:
            question_id: Unique question identifier (e.g., 'finqa_test_1630')

        Returns:
            Dictionary containing question data with doc_id linkage
        """
        # Check cache first
        if question_id in self.questions_cache:
            return self.questions_cache[question_id]

        # Load from data source
        if self.data_path.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(self.data_path)

            # Find question by ID - try different column names
            if 'question_id' in df.columns:
                question_row = df[df['question_id'] == question_id]
            elif 'id' in df.columns:
                question_row = df[df['id'] == question_id]
            else:
                raise ValueError(f"No question ID column found in data")

            if question_row.empty:
                raise ValueError(f"Question {question_id} not found")

            question_data = question_row.iloc[0].to_dict()

        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                question_data = next(
                    (q for q in data if q['question_id'] == question_id),
                    None
                )
                if not question_data:
                    raise ValueError(f"Question {question_id} not found")
        else:
            # Try loading from B-Pipeline outputs for compatibility
            b_output_path = f"B_Retrieval_pipeline/outputs/B1_current_question.json"
            if os.path.exists(b_output_path):
                with open(b_output_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        question_data = data[0] if data else {}
                    else:
                        question_data = data
            else:
                raise ValueError(f"No valid data source found for {question_id}")

        # Extract critical components for Q-Pipeline
        processed_question = self._process_question(question_data)

        # Cache for efficiency
        self.questions_cache[question_id] = processed_question

        return processed_question

    def _process_question(self, raw_data: Dict) -> Dict:
        """
        Process raw question data into Q-Pipeline format.

        CRITICAL: Establishes doc_id linkage for concept space alignment.

        Args:
            raw_data: Raw question data from source

        Returns:
            Processed question with Q-Pipeline structure
        """
        # Extract question_id - handle different column names
        question_id = raw_data.get('question_id') or raw_data.get('id', 'unknown')

        # Extract doc_id - CRITICAL for concept space alignment
        doc_id = raw_data.get('doc_id') or raw_data.get('document_id')

        # Handle different ID formats - for this dataset, use question ID as doc reference
        if not doc_id:
            # Use the question ID as doc_id for this dataset
            doc_id = str(question_id)

        if not doc_id:
            raise ValueError("Cannot determine doc_id for concept space alignment")

        # Base question data (always included)
        question_data = {
            "question_id": question_id,
            "question_text": raw_data.get('question', raw_data.get('question_text', '')),
            "doc_id": doc_id,  # CRITICAL: Links to A-Pipeline concept space
            "pipeline_ready": True
        }

        # CONDITIONALLY include answer data based on mode
        if self.mode == "evaluation":
            # Include answer data for validation/evaluation
            question_data["answer"] = raw_data.get('answer', raw_data.get('response', ''))
            question_data["metadata"] = {
                "source": raw_data.get('source', 'sample_data'),
                "ingestion_timestamp": datetime.now().isoformat(),
                "raw_data_keys": list(raw_data.keys()),
                "mode": "evaluation"
            }
            print(f"Q1 WARNING: Including answer data for evaluation mode")
        else:
            # inference mode - EXCLUDE answer data to prevent leakage
            question_data["metadata"] = {
                "source": raw_data.get('source', 'sample_data'),
                "ingestion_timestamp": datetime.now().isoformat(),
                "raw_data_keys": [k for k in raw_data.keys()
                                if k not in ['answer', 'response', 'generation_model_name']],
                "mode": "inference",
                "data_leakage_prevention": "answer_data_excluded"
            }

        return question_data

    def load_batch_questions(self, question_ids: List[str]) -> List[Dict]:
        """
        Load multiple questions for batch processing.

        Args:
            question_ids: List of question identifiers

        Returns:
            List of processed questions
        """
        questions = []
        for qid in question_ids:
            try:
                question = self.load_question(qid)
                questions.append(question)
            except Exception as e:
                print(f"Warning: Failed to load question {qid}: {e}")
                continue

        return questions

    def load_questions_by_doc(self, doc_id: str) -> List[Dict]:
        """
        Load all questions associated with a specific document.
        Efficient for batch processing within same concept space.

        Args:
            doc_id: Document identifier

        Returns:
            List of questions from that document
        """
        all_questions = []

        if self.data_path.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(self.data_path)

            # Filter by doc_id
            doc_questions = df[df['doc_id'] == doc_id]

            for _, row in doc_questions.iterrows():
                question_data = row.to_dict()
                processed = self._process_question(question_data)
                all_questions.append(processed)

        return all_questions

    def validate_doc_alignment(self, question: Dict) -> bool:
        """
        Validate that question has proper document alignment for concept space.

        Args:
            question: Processed question dictionary

        Returns:
            True if properly aligned, False otherwise
        """
        required_fields = ['question_id', 'question_text', 'doc_id']

        for field in required_fields:
            if field not in question or not question[field]:
                return False

        return True

    def save_output(self, question_data: Dict, output_path: str = None):
        """
        Save processed question data for downstream modules.

        Args:
            question_data: Processed question(s)
            output_path: Output file path
        """
        if output_path is None:
            output_path = "Q_Question_Pipeline/outputs/Q1_Question_ingestion.json"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(question_data, f, indent=2)

        print(f"Q1 output saved to {output_path}")


if __name__ == "__main__":
    # Test Q1 module
    q1 = Q1_QuestionIngestion()

    # Try loading a sample question
    try:
        # First, let's see what questions are available
        import pandas as pd
        if os.path.exists("data/sample_20_records.parquet"):
            df = pd.read_parquet("data/sample_20_records.parquet")
            print(f"Found {len(df)} questions in dataset")
            print(f"Sample question IDs: {df['question_id'].head().tolist()}")

            # Load first question
            first_qid = df['question_id'].iloc[0]
            question = q1.load_question(first_qid)

            print("\n" + "="*50)
            print("Q1 Output - Ingested Question:")
            print("="*50)
            print(f"Question ID: {question['question_id']}")
            print(f"Doc ID: {question['doc_id']}")
            print(f"Question: {question['question_text'][:100]}...")
            print(f"Pipeline Ready: {question['pipeline_ready']}")

            # Save output
            q1.save_output(question)

    except Exception as e:
        print(f"Error in Q1 testing: {e}")
        print("Will use fallback data from B-Pipeline outputs")