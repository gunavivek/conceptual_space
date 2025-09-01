#!/usr/bin/env python3
"""
A1.1: Document Reader
Loads documents from parquet files for pipeline processing
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime

def load_documents(data_path="data/test_mode_5_records.parquet"):
    """
    Load documents from parquet file
    
    Args:
        data_path: Path to parquet file containing documents
        
    Returns:
        dict: Loaded documents with metadata
    """
    # Resolve path relative to script location
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / data_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {full_path}")
    
    # Load parquet file
    df = pd.read_parquet(full_path)
    
    # Process documents  
    documents = []
    for idx, row in df.iterrows():
        # Handle the real FinQA data structure
        doc_array = row.get("documents", [])
        if hasattr(doc_array, 'tolist'):
            doc_array = doc_array.tolist()
        
        # Combine all document texts
        combined_text = ""
        if isinstance(doc_array, list):
            for doc_item in doc_array:
                if isinstance(doc_item, str):
                    combined_text += doc_item + " "
        
        doc = {
            "doc_id": row.get("id", f"doc_{idx}"),
            "text": combined_text.strip(),
            "question": str(row.get("question", "")),
            "metadata": {
                "source": str(full_path),
                "index": idx,
                "loaded_at": datetime.now().isoformat(),
                "dataset_name": str(row.get("dataset_name", "")),
                "original_response": str(row.get("response", ""))
            }
        }
        
        # Add numeric fields safely
        numeric_fields = ["adherence_score", "relevance_score", "utilization_score", "completeness_score"]
        for field in numeric_fields:
            value = row.get(field)
            if pd.notna(value):
                doc["metadata"][field] = float(value)
        
        documents.append(doc)
    
    return {
        "documents": documents,
        "count": len(documents),
        "source_file": str(full_path),
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/A1.1_raw_documents.json"):
    """
    Save processed documents to JSON
    
    Args:
        data: Document data to save
        output_path: Path for output file
    """
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    # Create output directory if it doesn't exist
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved {data['count']} documents to {full_path}")
    
    # Also save metadata
    meta_path = full_path.with_suffix('.meta.json')
    metadata = {
        "script": "A1.1_document_reader.py",
        "timestamp": data["processing_timestamp"],
        "document_count": data["count"],
        "output_file": str(full_path)
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main execution"""
    print("="*60)
    print("A1.1: Document Reader")
    print("="*60)
    
    try:
        # Load documents
        print("Loading documents from parquet file...")
        documents = load_documents()
        
        print(f"Loaded {documents['count']} documents")
        
        # Display sample
        if documents["documents"]:
            first_doc = documents["documents"][0]
            print(f"\nSample document:")
            print(f"  ID: {first_doc['doc_id']}")
            print(f"  Text length: {len(first_doc['text'])} characters")
            print(f"  Text preview: {first_doc['text'][:100]}...")
        
        # Save output
        save_output(documents)
        
        print("\nA1.1 Document Reader completed successfully!")
        
    except Exception as e:
        print(f"Error in A1.1 Document Reader: {str(e)}")
        raise

if __name__ == "__main__":
    main()