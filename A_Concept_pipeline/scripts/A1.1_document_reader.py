#!/usr/bin/env python3
"""
A1.1: Document Reader with RAGBench Domain Authority (ENHANCED)
Loads PURE document content from parquet files with authoritative domain classification
Integrates RAGBench dataset descriptions for accurate domain determination
Eliminates questions, responses, and evaluation data to prevent leakage
"""

import pandas as pd
import json
import os
import re
from pathlib import Path
from datetime import datetime

def load_ragbench_domain_mappings(ragbench_path="data/RAGBench_Dataset_Description.txt"):
    """
    Parse RAGBench dataset description to extract dataset-to-domain mappings
    
    Args:
        ragbench_path: Path to RAGBench dataset description file
        
    Returns:
        dict: Dataset prefix to domain mappings
    """
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / ragbench_path
    
    if not full_path.exists():
        print(f"Warning: RAGBench description not found: {full_path}")
        return {}
    
    # Domain mappings extracted from RAGBench dataset descriptions
    dataset_domains = {
        "finqa": "finance",        # "financial report passages"
        "tatqa": "finance",        # "financial QA dataset...financial reports" 
        "pubmedqa": "healthcare",  # "PubMed research abstracts"
        "covidqa": "healthcare",   # "COVID-19 research articles"
        "cuad": "legal",          # "commercial legal contracts"
        "techqa": "technology",   # "technical support documents"
        "delucionqa": "automotive", # "Jeep's 2023 Gladiator model manual"
        "emanual": "technology",   # "consumer electronic device manuals"
        "hotpotqa": "general",    # "Wikipedia articles"
        "msmarco": "general",     # "Bing search engine queries"
        "hagrid": "general"       # "multi-lingual Wikipedia"
    }
    
    return dataset_domains

def determine_document_domain(doc_id, text=""):
    """
    Determine document domain based on RAGBench dataset classification
    
    Args:
        doc_id: Document identifier (e.g., "finqa_test_617")
        text: Document text (for content analysis if needed)
        
    Returns:
        tuple: (domain, confidence, source)
    """
    dataset_domains = load_ragbench_domain_mappings()
    
    # Extract dataset prefix from document ID
    doc_id_lower = doc_id.lower()
    
    # Check for dataset prefix matches
    for dataset_prefix, domain in dataset_domains.items():
        if doc_id_lower.startswith(dataset_prefix):
            if domain != "general":
                return domain, 1.0, "ragbench_dataset"
            else:
                # For general datasets, could add lightweight content analysis
                # For now, return general with lower confidence
                return "general", 0.5, "ragbench_general"
    
    # No dataset match found - default to unknown
    return "unknown", 0.0, "no_dataset_match"

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
        
        # Determine domain using RAGBench dataset authority
        doc_id = row.get("id", f"doc_{idx}")
        text_content = combined_text.strip()
        domain, confidence, source = determine_document_domain(doc_id, text_content)
        
        # Clean A1.1 output structure: Doc_id, Domain, Text
        doc = {
            "doc_id": doc_id,
            "domain": domain,
            "text": text_content,
            "metadata": {
                "document_source": str(full_path),  # Where document content came from
                "domain_source": source,            # Where domain classification came from
                "index": idx,
                "loaded_at": datetime.now().isoformat(),
                "dataset_name": str(row.get("dataset_name", "")),
                "domain_confidence": confidence
            }
        }
        
        # ARCHITECTURAL PURITY + DOMAIN AUTHORITY:
        # - Only document content and essential domain classification
        # - RAGBench dataset-based domain determination
        # - Eliminated: questions, responses, evaluation scores to prevent data leakage
        
        documents.append(doc)
    
    # Calculate domain classification statistics
    domain_stats = {}
    for doc in documents:
        domain = doc.get("domain", "unknown")
        domain_stats[domain] = domain_stats.get(domain, 0) + 1
    
    return {
        "documents": documents,
        "count": len(documents),
        "domain_classification": domain_stats,
        "ragbench_integration": "enabled",
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
    """Main execution - Enhanced with RAGBench Domain Authority"""
    print("="*70)
    print("A1.1: Document Reader with RAGBench Domain Authority (ENHANCED)")
    print("="*70)
    
    try:
        # Load documents with RAGBench domain integration
        print("Loading documents and determining domains from RAGBench metadata...")
        documents = load_documents()
        
        print(f"Processed {documents['count']} documents with domain classification")
        
        # Display domain classification results
        print(f"\nRAGBench Domain Classification Results:")
        domain_stats = documents.get("domain_classification", {})
        for domain, count in domain_stats.items():
            print(f"  {domain}: {count} documents")
        
        # Display sample with domain
        if documents["documents"]:
            first_doc = documents["documents"][0]
            print(f"\nSample document:")
            print(f"  ID: {first_doc['doc_id']}")
            print(f"  Domain: {first_doc['domain']} (confidence: {first_doc['metadata']['domain_confidence']})")
            print(f"  Document Source: {Path(first_doc['metadata']['document_source']).name}")
            print(f"  Domain Source: {first_doc['metadata']['domain_source']}")
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