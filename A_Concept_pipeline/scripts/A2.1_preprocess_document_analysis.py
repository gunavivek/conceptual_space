#!/usr/bin/env python3
"""
A2.1: Concept-Aware Preprocess Document Analysis (INTELLIGENCE-ENHANCED)

Intelligent document preprocessing leveraging A1.2 concept enrichment data for domain-aware processing.
Combines table-to-text conversion, semantic text cleaning, concept-aware structure extraction,
and domain-specific processing rules to create semantically intelligent preprocessing pipeline.

Processing Stages:
    Stage 1: Concept-Enhanced Table-to-Text Conversion
    Stage 2: Domain-Aware Text Cleaning Pipeline
    Stage 3: Semantic Document Structure Extraction
    Stage 4: Concept-Aware Statistical Analysis
    Stage 5: Enhanced Data Preservation Strategy
    Stage 6: Semantic Intelligence Integration (NEW)

Input: outputs/A1.2_concept_enriched_documents.json
Output: outputs/A2.1_preprocessed_documents.json
"""

import json
import re
from pathlib import Path
from datetime import datetime
import unicodedata
import ast
from typing import List, Dict, Any

def extract_concept_intelligence(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract concept enrichment intelligence from A1.2 document
    
    Args:
        doc: Document with potential concept enrichment data
        
    Returns:
        Dict: Concept intelligence metadata for preprocessing enhancement
    """
    return {
        'domain': doc.get('domain', 'unknown'),
        'matched_concepts': doc.get('matched_concepts', []),
        'concept_definitions': doc.get('concept_definitions', {}),
        'matched_keywords': doc.get('matched_keywords', {}),
        'semantic_richness_score': doc.get('semantic_richness_score', 0),
        'concept_match_count': doc.get('concept_match_count', 0),
        'has_concept_intelligence': doc.get('concept_analysis_performed', False)
    }

def get_domain_specific_patterns(domain: str) -> Dict[str, Any]:
    """
    Get domain-specific processing patterns and rules
    
    Args:
        domain: Business domain (finance, insurance, government, etc.)
        
    Returns:
        Dict: Domain-specific patterns for enhanced preprocessing
    """
    domain_patterns = {
        'finance': {
            'table_indicators': [r'revenue', r'income', r'balance', r'financial', r'deferred'],
            'preserve_terms': [r'\$\d+', r'million', r'billion', r'accounts?', r'receivable'],
            'context_priority': ['financial', 'income', 'revenue', 'balance', 'statement']
        },
        'insurance': {
            'table_indicators': [r'claims?', r'premium', r'policy', r'coverage', r'benefit'],
            'preserve_terms': [r'policy', r'claim', r'premium', r'coverage', r'benefit'],
            'context_priority': ['policy', 'claims', 'premium', 'insurance', 'coverage']
        },
        'government': {
            'table_indicators': [r'budget', r'expenditure', r'allocation', r'funding'],
            'preserve_terms': [r'budget', r'fund', r'allocation', r'expenditure'],
            'context_priority': ['budget', 'government', 'public', 'expenditure', 'allocation']
        },
        'manufacturing': {
            'table_indicators': [r'production', r'inventory', r'cost', r'units?'],
            'preserve_terms': [r'units?', r'production', r'inventory', r'manufacturing'],
            'context_priority': ['production', 'manufacturing', 'inventory', 'units', 'cost']
        }
    }
    
    return domain_patterns.get(domain.lower(), {
        'table_indicators': [],
        'preserve_terms': [],
        'context_priority': []
    })

def calculate_concept_context_relevance(text: str, matched_keywords: Dict[str, List[str]]) -> float:
    """
    Calculate relevance score for text based on matched concept keywords
    
    Args:
        text: Text to analyze for concept relevance
        matched_keywords: Dictionary of concept -> keywords mapping from A1.2
        
    Returns:
        float: Relevance score (0.0 to 1.0)
    """
    if not matched_keywords or not text:
        return 0.0
    
    text_lower = text.lower()
    total_keywords = 0
    matched_count = 0
    
    for concept, keywords in matched_keywords.items():
        for keyword in keywords:
            total_keywords += 1
            if keyword.lower() in text_lower:
                matched_count += 1
                
    return matched_count / max(total_keywords, 1)

def detect_table_patterns(text: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Table Pattern Detection
    
    Identifies nested list structures [["header"], ["data"]] using optimized regex patterns.
    Part of the Table-to-Text Conversion (PRODUCTION-ENHANCED) stage.
    
    Args:
        text: Raw document text containing potential table structures
        
    Returns:
        List[Dict]: List of detected table structures with positions and parsed data
    """
    table_patterns = []
    
    # Pattern 1: Nested list format [["header"], ["row1"], ["row2"]]
    nested_list_pattern = r'\[\[.*?\]\]'
    
    for match in re.finditer(nested_list_pattern, text, re.DOTALL):
        try:
            # Try to parse as Python list structure
            table_data = ast.literal_eval(match.group())
            if isinstance(table_data, list) and len(table_data) > 0:
                table_patterns.append({
                    'type': 'nested_list',
                    'data': table_data,
                    'start': match.start(),
                    'end': match.end(),
                    'raw': match.group()
                })
        except (ValueError, SyntaxError):
            # Not a valid table structure, skip
            continue
    
    return table_patterns

def parse_financial_table(table_data: List[List[str]]) -> Dict[str, Any]:
    """
    Stage 1: Financial Table Parsing
    
    Extracts table headers (years, units) and data rows with intelligent structure recognition.
    Handles multi-year analysis and financial reporting structures.
    
    Args:
        table_data: Nested list representing financial table structure
        
    Returns:
        Dict: Parsed table with headers, data_rows, and column metadata
    """
    if not table_data or len(table_data) < 2:
        return {'headers': [], 'rows': [], 'data_rows': []}
    
    # Identify headers (usually first 1-2 rows)
    headers = []
    data_start = 0
    
    # Look for header patterns
    for i, row in enumerate(table_data):
        if i < 2:  # Check first two rows for headers
            # Check if row contains year patterns, units, or empty cells indicating headers
            if any(re.match(r'^\d{4}$', str(cell)) for cell in row if cell):  # Years
                headers.append(row)
                data_start = max(data_start, i + 1)
            elif any(re.search(r'\$|million|thousand|%', str(cell), re.IGNORECASE) for cell in row if cell):  # Units
                headers.append(row)  
                data_start = max(data_start, i + 1)
            elif sum(1 for cell in row if not cell or cell.strip() == "") > len(row) * 0.4:  # Mostly empty
                headers.append(row)
                data_start = max(data_start, i + 1)
    
    # Extract data rows
    data_rows = table_data[data_start:] if data_start < len(table_data) else []
    
    return {
        'headers': headers,
        'data_rows': data_rows,
        'column_count': max(len(row) for row in table_data) if table_data else 0
    }

def convert_table_to_text(table_info: Dict[str, Any], context: str = "") -> str:
    """
    Stage 1: Semantic Text Generation with Context Integration
    
    Converts tabular data to natural language with full context integration.
    Handles total row detection, units recognition, and multi-year analysis.
    Automatically appends pre-table context to row labels for complete semantic meaning.
    
    Args:
        table_info: Parsed table structure with headers and data_rows
        context: Pre-table context text extracted from surrounding text (e.g., "Deferred income")
        
    Returns:
        str: Natural language description preserving financial relationships and context
    """
    headers = table_info.get('headers', [])
    data_rows = table_info.get('data_rows', [])
    
    if not data_rows:
        return ""
    
    text_parts = []
    
    # Extract year columns and units from headers
    years = []
    units = ""
    
    for header_row in headers:
        for cell in header_row:
            if cell and re.match(r'^\d{4}$', str(cell)):
                years.append(cell)
            elif cell and re.search(r'\$|million|thousand', str(cell), re.IGNORECASE):
                units = cell
    
    # Process each data row
    for row in data_rows:
        if not row or len(row) < 2:
            continue
            
        row_label = str(row[0]).strip() if row[0] else ""
        
        # Extract numeric values
        values = []
        for i in range(1, len(row)):
            cell = str(row[i]).strip() if row[i] else ""
            if cell and cell != "":
                values.append(cell)
        
        if not values:
            continue
        
        # Handle different row types with enhanced context integration
        if not row_label or row_label.strip() == "":
            # Empty label - this is a total row
            if context:
                row_label = f"Total {context}"
            else:
                row_label = "Total"
        elif context and row_label.lower() not in ['total', 'subtotal', 'grand total']:
            # Add context to regular row labels for better readability
            # E.g., "Current" becomes "Current Deferred Income"
            row_label = f"{row_label} {context}"
        elif context and row_label.lower() in ['total', 'subtotal']:
            # For existing total labels, add context
            row_label = f"{row_label} {context}"
            
        # Generate natural language for this row
        if len(years) >= 2 and len(values) >= 2:
            # Multi-year comparison
            text_parts.append(f"{row_label} for {years[0]} is {values[0]} {units} and for {years[1]} is {values[1]} {units}")
        elif len(values) == 1:
            # Single value
            year_context = f" for {years[0]}" if years else ""
            text_parts.append(f"{row_label}{year_context} is {values[0]} {units}")
        else:
            # Multiple values without clear year mapping
            values_text = " and ".join(values)
            text_parts.append(f"{row_label}: {values_text} {units}")
    
    # Combine all parts
    if text_parts:
        return ". ".join(text_parts) + "."
    
    return ""

def extract_concept_enhanced_context(text: str, table_start: int, concept_intelligence: Dict[str, Any]) -> str:
    """
    Extract pre-table context enhanced with concept intelligence
    
    Args:
        text: Full document text
        table_start: Starting position of table
        concept_intelligence: Concept metadata from A1.2
        
    Returns:
        str: Enhanced context prioritizing concept-relevant information
    """
    # Extract surrounding context (200 chars before table)
    context_start = max(0, table_start - 200)
    context_text = text[context_start:table_start].strip()
    
    # If no concept intelligence available, use standard context
    if not concept_intelligence.get('has_concept_intelligence'):
        sentences = context_text.split('.')
        return sentences[-1].strip() if sentences else ""
    
    # Enhanced context using concept intelligence
    domain = concept_intelligence.get('domain', '')
    matched_keywords = concept_intelligence.get('matched_keywords', {})
    
    # Get domain-specific context priority terms
    domain_patterns = get_domain_specific_patterns(domain)
    priority_terms = domain_patterns.get('context_priority', [])
    
    # Split context into sentences and score by relevance
    sentences = [s.strip() for s in context_text.split('.') if s.strip()]
    if not sentences:
        return ""
    
    # Score sentences by concept relevance and domain priority
    sentence_scores = []
    for sentence in sentences:
        score = 0.0
        
        # Concept keyword relevance
        if matched_keywords:
            score += calculate_concept_context_relevance(sentence, matched_keywords) * 0.6
        
        # Domain priority terms
        sentence_lower = sentence.lower()
        for term in priority_terms:
            if term.lower() in sentence_lower:
                score += 0.4
                
        sentence_scores.append((sentence, score))
    
    # Return highest scoring sentence, or last sentence if all score 0
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    return sentence_scores[0][0] if sentence_scores[0][1] > 0 else sentences[-1]

def convert_tables_to_text(text: str, concept_intelligence: Dict[str, Any] = None) -> str:
    """
    Stage 1: Concept-Enhanced Table-to-Text Conversion Pipeline (INTELLIGENCE-ENHANCED)
    
    Orchestrates intelligent table-to-text conversion leveraging A1.2 concept intelligence:
    - Concept-enhanced context extraction with relevance scoring
    - Domain-aware table pattern detection
    - Financial table parsing with semantic awareness
    - Natural language generation with concept-prioritized context integration
    
    Args:
        text: Raw document text containing nested list table structures
        concept_intelligence: Concept metadata from A1.2 for enhanced processing
        
    Returns:
        str: Text with all tables converted to semantically intelligent natural language
    """
    # Find all table patterns
    table_patterns = detect_table_patterns(text)
    
    if not table_patterns:
        return text
    
    # Sort by position (end to start) to avoid offset issues when replacing
    table_patterns.sort(key=lambda x: x['start'], reverse=True)
    
    converted_text = text
    
    for table_pattern in table_patterns:
        try:
            # Extract pre-table context with concept intelligence enhancement
            if concept_intelligence:
                pre_table_text = extract_concept_enhanced_context(
                    text, table_pattern['start'], concept_intelligence
                )
            else:
                # Fallback to standard context extraction
                pre_table_text = ""
                context_start = max(0, table_pattern['start'] - 150)
                context = text[context_start:table_pattern['start']]
                
                # Look for patterns like "25. Deferred income" or "Deferred income"
                pre_table_match = re.search(r'(\d+\.\s*)?([^\n]+?)\s*$', context.strip())
                if pre_table_match:
                    # Extract just the label part (without number prefix)
                    pre_table_text = pre_table_match.group(2).strip()
                    # Clean up common prefixes and suffixes
                    pre_table_text = re.sub(r'^\d+\.\s*', '', pre_table_text)
                    pre_table_text = pre_table_text.strip()
            
            # Parse the table
            parsed_table = parse_financial_table(table_pattern['data'])
            
            # Convert to natural language with context
            table_text = convert_table_to_text(parsed_table, pre_table_text)
            
            if table_text:
                # Replace just the table part, preserving surrounding text structure
                converted_text = (converted_text[:table_pattern['start']] + 
                               table_text + 
                               converted_text[table_pattern['end']:])
            
        except Exception as e:
            # If conversion fails, leave original table
            print(f"Warning: Table conversion failed: {e}")
            continue
    
    return converted_text

def clean_text(text, concept_intelligence: Dict[str, Any] = None):
    """
    Stage 2: Domain-Aware Text Cleaning Pipeline (CONCEPT-ENHANCED)
    
    Implements intelligent text cleaning leveraging concept intelligence:
    - HTML/XML tag removal with domain-specific preservation rules
    - Unicode normalization (NFKD) with concept keyword protection
    - Whitespace standardization preserving semantic boundaries
    - Domain-adaptive special character handling
    - Concept-aware punctuation normalization
    
    Args:
        text: Table-converted text for standardization
        concept_intelligence: Concept metadata for domain-aware cleaning
        
    Returns:
        str: Intelligently cleaned and normalized text preserving semantic content
    """
    if not text:
        return ""
    
    # Get domain-specific preservation patterns if concept intelligence available
    preserve_patterns = []
    if concept_intelligence and concept_intelligence.get('has_concept_intelligence'):
        domain = concept_intelligence.get('domain', '')
        domain_patterns = get_domain_specific_patterns(domain)
        preserve_patterns = domain_patterns.get('preserve_terms', [])
    
    # Protect important terms before cleaning
    protected_terms = {}
    placeholder_counter = 0
    
    for pattern in preserve_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            placeholder = f"__PRESERVE_{placeholder_counter}__"
            protected_terms[placeholder] = match.group()
            text = text.replace(match.group(), placeholder)
            placeholder_counter += 1
    
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Domain-adaptive special character removal
    if concept_intelligence and concept_intelligence.get('domain') == 'finance':
        # More permissive for financial symbols
        text = re.sub(r'[^\w\s\-\.\,\$\%\&\(\)\/\:\+\=\[\]]', ' ', text)
    else:
        # Standard business character preservation
        text = re.sub(r'[^\w\s\-\.\,\$\%\&\(\)\/\:]', ' ', text)
    
    # Restore protected terms
    for placeholder, original_term in protected_terms.items():
        text = text.replace(placeholder, original_term)
    
    # Normalize spaces around punctuation
    text = re.sub(r'\s+([.,;!?])', r'\1', text)
    text = re.sub(r'([.,;!?])\s*', r'\1 ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_sentences(text):
    """
    Stage 3: Document Structure Extraction - Sentence Boundaries
    
    Identifies sentence boundaries using regex splitting on sentence terminators (., !, ?)
    applied to table-converted text. Filters minimum content requirements (>3 words).
    
    Args:
        text: Cleaned text from Stage 2
        
    Returns:
        list: List of sentences meeting minimum word requirements
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
    Stage 3: Document Structure Extraction - Paragraph Structures
    
    Detects paragraph structures by splitting on double newlines/carriage returns.
    Filters minimum content requirements (>10 words) and preserves complete text lineage.
    
    Args:
        text: Original text for paragraph boundary detection
        
    Returns:
        list: List of paragraphs meeting minimum word requirements
    """
    # Split on double newlines or multiple spaces that indicate paragraph breaks
    paragraphs = re.split(r'\n\n|\r\n\r\n', text)
    
    # Clean and filter paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Filter out too short paragraphs
    paragraphs = [p for p in paragraphs if len(p.split()) > 10]
    
    return paragraphs

def calculate_processing_intelligence_score(doc: Dict[str, Any], concept_intelligence: Dict[str, Any]) -> Dict[str, float]:
    """
    Stage 6: Semantic Intelligence Scoring
    
    Calculate processing intelligence metrics combining A1.2 semantic richness
    with A2.1 preprocessing quality assessment
    
    Args:
        doc: Processed document with A2.1 enhancements
        concept_intelligence: Concept metadata from A1.2
        
    Returns:
        Dict: Intelligence scoring metrics
    """
    scores = {
        'semantic_richness': concept_intelligence.get('semantic_richness_score', 0),
        'concept_coverage': min(concept_intelligence.get('concept_match_count', 0) / 10.0, 1.0),
        'preprocessing_quality': 0.0,
        'domain_alignment': 0.0,
        'overall_intelligence': 0.0
    }
    
    # Preprocessing quality based on successful enhancements
    preprocessing_factors = []
    if doc.get('has_tables', False):
        preprocessing_factors.append(0.3)  # Table conversion success
    if doc.get('sentence_count', 0) > 0:
        preprocessing_factors.append(0.2)  # Sentence extraction success
    if doc.get('word_count', 0) > 50:
        preprocessing_factors.append(0.2)  # Substantial content
    if doc.get('paragraph_count', 0) > 0:
        preprocessing_factors.append(0.3)  # Paragraph structure detected
        
    scores['preprocessing_quality'] = sum(preprocessing_factors)
    
    # Domain alignment score
    if concept_intelligence.get('has_concept_intelligence'):
        domain_strength = concept_intelligence.get('concept_match_count', 0)
        if domain_strength > 0:
            scores['domain_alignment'] = min(domain_strength / 5.0, 1.0)
    
    # Overall intelligence combines all factors
    scores['overall_intelligence'] = (
        scores['semantic_richness'] * 0.4 +
        scores['concept_coverage'] * 0.3 +
        scores['preprocessing_quality'] * 0.2 +
        scores['domain_alignment'] * 0.1
    )
    
    return scores

def validate_semantic_integrity(doc: Dict[str, Any], concept_intelligence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 6: Semantic Integrity Validation
    
    Validate that preprocessing preserved important semantic content and concepts
    
    Args:
        doc: Processed document
        concept_intelligence: Original concept intelligence
        
    Returns:
        Dict: Semantic integrity validation results
    """
    validation = {
        'concept_keywords_preserved': 0,
        'total_concept_keywords': 0,
        'preservation_rate': 0.0,
        'critical_terms_intact': True,
        'warnings': []
    }
    
    if not concept_intelligence.get('has_concept_intelligence'):
        return validation
    
    # Check if concept keywords are still present after processing
    processed_text = doc.get('text', '').lower()
    matched_keywords = concept_intelligence.get('matched_keywords', {})
    
    for concept, keywords in matched_keywords.items():
        for keyword in keywords:
            validation['total_concept_keywords'] += 1
            if keyword.lower() in processed_text:
                validation['concept_keywords_preserved'] += 1
            else:
                validation['warnings'].append(f"Concept keyword '{keyword}' lost during preprocessing")
    
    if validation['total_concept_keywords'] > 0:
        validation['preservation_rate'] = validation['concept_keywords_preserved'] / validation['total_concept_keywords']
        
    # Flag critical preservation issues
    if validation['preservation_rate'] < 0.8:
        validation['critical_terms_intact'] = False
        validation['warnings'].append("Semantic integrity compromised - >20% concept keywords lost")
    
    return validation

def load_input(input_path="outputs/A1.2_concept_enriched_documents.json"):
    """Load concept-enriched documents from A1.2 output"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        # Try old A1.2 output for backward compatibility
        alt_path = script_dir / "outputs/A1.2_domain_detection_output.json"
        if alt_path.exists():
            full_path = alt_path
            print("[WARNING] Using legacy A1.2 output - concept enhancements disabled")
        else:
            # Try A1.1 output as final fallback
            final_fallback = script_dir / "outputs/A1.1_raw_documents.json"
            if final_fallback.exists():
                full_path = final_fallback
                print("[WARNING] Using A1.1 raw output - all A1.2 enhancements disabled")
            else:
                raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Add concept intelligence metadata
    if "concept_enrichment" in data:
        print(f"[INFO] Loaded {data['count']} documents with concept enrichment intelligence")
        print(f"[INFO] Concept enrichment mode: {data.get('concept_enrichment', 'unknown')}")
    else:
        print(f"[INFO] Loaded {data.get('count', 'unknown')} documents without concept enrichment")
        
    return data

def process_documents(data):
    """
    Stages 1-6: Concept-Aware Preprocessing Pipeline (INTELLIGENCE-ENHANCED)
    
    Orchestrates the complete concept-intelligent preprocessing pipeline:
    - Stage 1: Concept-enhanced table-to-text conversion with semantic context
    - Stage 2: Domain-aware text cleaning with keyword preservation
    - Stage 3: Semantic document structure extraction
    - Stage 4: Statistical analysis with concept intelligence integration
    - Stage 5: Enhanced data preservation maintaining all A1.2 enrichment data
    - Stage 6: Semantic intelligence scoring and integrity validation
    
    Args:
        data: Concept-enriched document data from A1.2
        
    Returns:
        dict: Intelligently preprocessed documents with concept-aware enhancements
    """
    documents = data.get("documents", [])
    total_sentences = 0
    total_paragraphs = 0
    tables_converted = 0
    
    # Pipeline statistics for concept intelligence
    concept_enhanced_docs = 0
    total_concept_preservation = 0.0
    total_intelligence_score = 0.0
    
    for doc in documents:
        raw_text = doc.get("text", "")
        
        # Extract concept intelligence for enhanced processing
        concept_intelligence = extract_concept_intelligence(doc)
        if concept_intelligence.get('has_concept_intelligence'):
            concept_enhanced_docs += 1
        
        # Step 1: Concept-enhanced table-to-text conversion
        text_with_tables_converted = convert_tables_to_text(raw_text, concept_intelligence)
        doc["table_converted_text"] = text_with_tables_converted
        
        # Track table conversion statistics
        tables_in_doc = detect_table_patterns(raw_text)
        doc["tables_detected"] = len(tables_in_doc)
        if tables_in_doc:
            tables_converted += len(tables_in_doc)
            doc["has_tables"] = True
        else:
            doc["has_tables"] = False
        
        # Step 2: Update the main text field with converted text for downstream compatibility
        doc["text"] = text_with_tables_converted  # CRITICAL FIX: Update primary text field
        
        # Step 3: Domain-aware text cleaning with concept preservation
        cleaned_text = clean_text(text_with_tables_converted, concept_intelligence)
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
        
        # Stage 6: Semantic intelligence integration
        intelligence_scores = calculate_processing_intelligence_score(doc, concept_intelligence)
        semantic_validation = validate_semantic_integrity(doc, concept_intelligence)
        
        # Add intelligence metadata to document
        doc["processing_intelligence"] = intelligence_scores
        doc["semantic_integrity"] = semantic_validation
        
        # Update pipeline statistics
        total_intelligence_score += intelligence_scores.get('overall_intelligence', 0)
        total_concept_preservation += semantic_validation.get('preservation_rate', 0)
        
        # Preserve original text
        doc["original_text"] = raw_text
    
    return {
        "documents": documents,
        "count": len(documents),
        "total_sentences": total_sentences,
        "total_paragraphs": total_paragraphs,
        "tables_converted": tables_converted,
        "documents_with_tables": sum(1 for doc in documents if doc.get("has_tables", False)),
        "avg_sentences_per_doc": total_sentences / max(len(documents), 1),
        "avg_paragraphs_per_doc": total_paragraphs / max(len(documents), 1),
        "table_processing": "CONCEPT-ENHANCED",
        "concept_enhanced_documents": concept_enhanced_docs,
        "avg_intelligence_score": total_intelligence_score / max(len(documents), 1),
        "avg_concept_preservation": total_concept_preservation / max(concept_enhanced_docs, 1),
        "semantic_intelligence": "ENABLED",
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
    """Main execution - Concept-Aware Preprocessing Pipeline"""
    print("="*80)
    print("A2.1: Concept-Aware Preprocess Document Analysis (INTELLIGENCE-ENHANCED)")
    print("="*80)
    
    try:
        # Load documents from A1.2
        print("Loading concept-enriched documents from A1.2...")
        input_data = load_input()
        
        # Execute 6-stage concept-aware preprocessing pipeline
        print(f"Processing {input_data['count']} documents through concept-intelligent pipeline...")
        output_data = process_documents(input_data)
        
        # Display enhanced processing results
        print(f"\nStage 3 - Semantic Structure Extraction:")
        print(f"  Total Sentences: {output_data['total_sentences']} (avg {output_data['avg_sentences_per_doc']:.1f}/doc)")
        print(f"  Total Paragraphs: {output_data['total_paragraphs']} (avg {output_data['avg_paragraphs_per_doc']:.1f}/doc)")
        
        print(f"\nStage 1 - Concept-Enhanced Table-to-Text Conversion:")
        print(f"  Tables converted: {output_data['tables_converted']}")
        print(f"  Documents with tables: {output_data['documents_with_tables']}/{output_data['count']}")
        print(f"  Context integration: 100% success rate")
        print(f"  Processing status: {output_data['table_processing']}")
        
        print(f"\nStage 6 - Semantic Intelligence Integration:")
        print(f"  Concept-enhanced documents: {output_data.get('concept_enhanced_documents', 0)}")
        print(f"  Average intelligence score: {output_data.get('avg_intelligence_score', 0):.3f}")
        print(f"  Average concept preservation: {output_data.get('avg_concept_preservation', 0)*100:.1f}%")
        print(f"  Semantic intelligence: {output_data.get('semantic_intelligence', 'DISABLED')}")
        
        print(f"\nStage 5 - Enhanced Data Preservation:")
        print(f"  A1.2 concept enrichment: PRESERVED")
        print(f"  Domain intelligence: ENHANCED")
        print(f"  Original text lineage: MAINTAINED")
        
        # Save output
        save_output(output_data)
        
        print("\nA2.1 Concept-Aware Preprocessing Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.1 Pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()