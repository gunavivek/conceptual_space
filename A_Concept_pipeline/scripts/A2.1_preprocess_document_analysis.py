#!/usr/bin/env python3
"""
A2.1: Document Preprocessing
Clean and standardize document text for analysis
"""

import json
import re
from pathlib import Path
from datetime import datetime
import unicodedata

def clean_text(text):
    """
    Clean and normalize text
    
    Args:
        text: Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep business-relevant ones
    text = re.sub(r'[^\w\s\-\.\,\$\%\&\(\)\/\:]', ' ', text)
    
    # Normalize spaces around punctuation
    text = re.sub(r'\s+([.,;!?])', r'\1', text)
    text = re.sub(r'([.,;!?])\s*', r'\1 ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_sentences(text):
    """
    Extract sentences from text
    
    Args:
        text: Cleaned text
        
    Returns:
        list: List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Filter out too short sentences
    sentences = [s for s in sentences if len(s.split()) > 3]
    
    return sentences

def extract_paragraphs(text):
    """
    Extract paragraphs from text
    
    Args:
        text: Cleaned text
        
    Returns:
        list: List of paragraphs
    """
    # Split on double newlines or multiple spaces that indicate paragraph breaks
    paragraphs = re.split(r'\n\n|\r\n\r\n', text)
    
    # Clean and filter paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Filter out too short paragraphs
    paragraphs = [p for p in paragraphs if len(p.split()) > 10]
    
    return paragraphs

def load_input(input_path="outputs/A1.2_domain_detection_output.json"):
    """Load documents from A1.2 output"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        # Try A1.1 output if A1.2 doesn't exist
        alt_path = script_dir / "outputs/A1.1_raw_documents.json"
        if alt_path.exists():
            full_path = alt_path
        else:
            raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_documents(data):
    """
    Process documents for preprocessing
    
    Args:
        data: Document data from previous step
        
    Returns:
        dict: Preprocessed documents
    """
    documents = data.get("documents", [])
    total_sentences = 0
    total_paragraphs = 0
    
    for doc in documents:
        raw_text = doc.get("text", "")
        
        # Clean text
        cleaned_text = clean_text(raw_text)
        doc["cleaned_text"] = cleaned_text
        
        # Extract sentences
        sentences = extract_sentences(cleaned_text)
        doc["sentences"] = sentences
        doc["sentence_count"] = len(sentences)
        total_sentences += len(sentences)
        
        # Extract paragraphs
        paragraphs = extract_paragraphs(raw_text)
        doc["paragraphs"] = paragraphs
        doc["paragraph_count"] = len(paragraphs)
        total_paragraphs += len(paragraphs)
        
        # Calculate statistics
        words = cleaned_text.split()
        doc["word_count"] = len(words)
        doc["avg_sentence_length"] = len(words) / max(len(sentences), 1)
        
        # Preserve original text
        doc["original_text"] = raw_text
    
    return {
        "documents": documents,
        "count": len(documents),
        "total_sentences": total_sentences,
        "total_paragraphs": total_paragraphs,
        "avg_sentences_per_doc": total_sentences / max(len(documents), 1),
        "avg_paragraphs_per_doc": total_paragraphs / max(len(documents), 1),
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/A2.1_preprocessed_documents.json"):
    """Save preprocessed documents"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved preprocessed documents to {full_path}")
    
    # Save metadata
    meta_path = full_path.with_suffix('.meta.json')
    metadata = {
        "script": "A2.1_preprocess_document_analysis.py",
        "timestamp": data["processing_timestamp"],
        "document_count": data["count"],
        "total_sentences": data["total_sentences"],
        "total_paragraphs": data["total_paragraphs"],
        "output_file": str(full_path)
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main execution"""
    print("="*60)
    print("A2.1: Document Preprocessing")
    print("="*60)
    
    try:
        # Load documents
        print("Loading documents from previous step...")
        input_data = load_input()
        
        # Process documents
        print(f"Preprocessing {input_data['count']} documents...")
        output_data = process_documents(input_data)
        
        # Display results
        print(f"\nPreprocessing Statistics:")
        print(f"  Documents: {output_data['count']}")
        print(f"  Total Sentences: {output_data['total_sentences']}")
        print(f"  Total Paragraphs: {output_data['total_paragraphs']}")
        print(f"  Avg Sentences/Doc: {output_data['avg_sentences_per_doc']:.1f}")
        print(f"  Avg Paragraphs/Doc: {output_data['avg_paragraphs_per_doc']:.1f}")
        
        # Save output
        save_output(output_data)
        
        print("\nA2.1 Document Preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.1 Document Preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()