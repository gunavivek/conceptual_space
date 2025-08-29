#!/usr/bin/env python3
"""
A1.2: Document Domain Detector
Identifies the domain (finance, healthcare, technology, etc.) of documents
"""

import json
from pathlib import Path
from datetime import datetime
import re

def detect_domain(text):
    """
    Detect document domain based on keyword presence
    
    Args:
        text: Document text
        
    Returns:
        str: Detected domain
    """
    # Domain keyword mappings
    domains = {
        "finance": [
            "revenue", "income", "expense", "profit", "loss", "financial",
            "investment", "capital", "asset", "liability", "equity", "earnings",
            "cash flow", "balance sheet", "income statement", "dividend",
            "deferred", "amortization", "depreciation", "fiscal", "quarter"
        ],
        "healthcare": [
            "patient", "medical", "health", "hospital", "treatment", "diagnosis",
            "disease", "medicine", "clinical", "therapy", "doctor", "nurse",
            "symptom", "prescription", "pharmaceutical", "vaccine"
        ],
        "technology": [
            "software", "hardware", "algorithm", "data", "system", "network",
            "computer", "programming", "code", "application", "database",
            "cloud", "AI", "machine learning", "API", "framework"
        ],
        "legal": [
            "law", "legal", "court", "judge", "attorney", "contract", "agreement",
            "clause", "regulation", "compliance", "litigation", "statute",
            "jurisdiction", "plaintiff", "defendant"
        ],
        "manufacturing": [
            "production", "manufacturing", "factory", "assembly", "supply chain",
            "inventory", "quality control", "logistics", "warehouse", "distribution",
            "raw material", "product line", "operations"
        ]
    }
    
    text_lower = text.lower()
    domain_scores = {}
    
    # Calculate scores for each domain
    for domain, keywords in domains.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        domain_scores[domain] = score
    
    # Get domain with highest score
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[best_domain] > 0:
            return best_domain
    
    return "general"

def load_input(input_path="outputs/A1.1_raw_documents.json"):
    """Load documents from A1.1 output"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_documents(data):
    """
    Process documents to detect domains
    
    Args:
        data: Document data from A1.1
        
    Returns:
        dict: Documents with domain information
    """
    documents = data.get("documents", [])
    
    for doc in documents:
        text = doc.get("text", "")
        doc["domain"] = detect_domain(text)
        
        # Add domain confidence (simplified)
        doc["domain_confidence"] = 0.8 if doc["domain"] != "general" else 0.5
    
    # Calculate domain distribution
    domain_counts = {}
    for doc in documents:
        domain = doc["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    return {
        "documents": documents,
        "count": len(documents),
        "domain_distribution": domain_counts,
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/A1.2_domain_detection_output.json"):
    """Save processed documents with domain information"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved domain detection results to {full_path}")
    
    # Save metadata
    meta_path = full_path.with_suffix('.meta.json')
    metadata = {
        "script": "A1.2_document_domain_detector.py",
        "timestamp": data["processing_timestamp"],
        "document_count": data["count"],
        "domains": data["domain_distribution"],
        "output_file": str(full_path)
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main execution"""
    print("="*60)
    print("A1.2: Document Domain Detector")
    print("="*60)
    
    try:
        # Load documents from A1.1
        print("Loading documents from A1.1 output...")
        input_data = load_input()
        
        # Process documents
        print(f"Processing {input_data['count']} documents...")
        output_data = process_documents(input_data)
        
        # Display results
        print(f"\nDomain Distribution:")
        for domain, count in output_data["domain_distribution"].items():
            print(f"  {domain}: {count} documents")
        
        # Save output
        save_output(output_data)
        
        print("\nA1.2 Document Domain Detector completed successfully!")
        
    except Exception as e:
        print(f"Error in A1.2 Document Domain Detector: {str(e)}")
        raise

if __name__ == "__main__":
    main()