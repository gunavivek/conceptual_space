#!/usr/bin/env python3
"""
A1.2: Domain Concept Enrichment Engine
Enriches documents with semantic concept analysis using A1.1's authoritative domain classification.
Leverages R4S Semantic Ontology and R1_DOMAINS for domain-specific concept understanding.
"""

import json
import re
import math
import ast
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any

def load_a11_documents(input_path="outputs/A1.1_raw_documents.json"):
    """Load documents with authoritative domain classifications from A1.1"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"A1.1 documents not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_r4s_ontology():
    """Load R4S Semantic Ontology with concept definitions and keywords"""
    script_dir = Path(__file__).parent.parent.parent
    r4s_path = script_dir / "R_Reference_pipeline" / "output" / "R4S_Semantic_Ontology.json"
    
    if not r4s_path.exists():
        raise FileNotFoundError(f"R4S_Semantic_Ontology.json not found at: {r4s_path}")
    
    with open(r4s_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_r1_domains():
    """Load R1_DOMAINS for business domain structure validation"""
    script_dir = Path(__file__).parent.parent.parent
    r1_path = script_dir / "R_Reference_pipeline" / "output" / "R1_DOMAINS.json"
    
    if not r1_path.exists():
        raise FileNotFoundError(f"R1_DOMAINS.json not found at: {r1_path}")
    
    with open(r1_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_domain_concepts(domain, r4s_ontology, r1_domains):
    """
    Extract relevant concept subset for specific domain from R4S ontology
    
    Args:
        domain: A1.1 authoritative domain (e.g., "finance")
        r4s_ontology: R4S semantic ontology
        r1_domains: R1 domain structure
        
    Returns:
        dict: Domain-specific concepts with their definitions and keywords
    """
    if domain not in r1_domains.get("domains", {}):
        return {}
    
    # Get concept list for this domain from R1_DOMAINS
    domain_concept_list = r1_domains["domains"][domain].get("concepts", [])
    
    # Extract corresponding concepts from R4S ontology
    domain_concepts = {}
    for concept_id in domain_concept_list:
        if concept_id in r4s_ontology.get("concepts", {}):
            domain_concepts[concept_id] = r4s_ontology["concepts"][concept_id]
    
    return domain_concepts

def detect_table_patterns(text: str) -> List[Dict[str, Any]]:
    """Detect nested list table structures in text"""
    table_patterns = []
    nested_list_pattern = r'\[\[.*?\]\]'
    
    for match in re.finditer(nested_list_pattern, text, re.DOTALL):
        try:
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
            continue
    
    return table_patterns

def parse_financial_table(table_data: List[List[str]]) -> Dict[str, Any]:
    """Parse financial table structure to extract headers and data rows"""
    if not table_data or len(table_data) < 2:
        return {'headers': [], 'data_rows': []}
    
    headers = []
    data_start = 0
    
    # Identify header rows
    for i, row in enumerate(table_data):
        if i < 2:
            if any(re.match(r'^\d{4}$', str(cell)) for cell in row if cell):
                headers.append(row)
                data_start = max(data_start, i + 1)
            elif any(re.search(r'\$|million|thousand|%', str(cell), re.IGNORECASE) for cell in row if cell):
                headers.append(row)
                data_start = max(data_start, i + 1)
            elif sum(1 for cell in row if not cell or cell.strip() == "") > len(row) * 0.4:
                headers.append(row)
                data_start = max(data_start, i + 1)
    
    data_rows = table_data[data_start:] if data_start < len(table_data) else []
    
    return {'headers': headers, 'data_rows': data_rows}

def convert_table_to_text(table_info: Dict[str, Any], context: str = "") -> str:
    """Convert table data to natural language sentences"""
    headers = table_info.get('headers', [])
    data_rows = table_info.get('data_rows', [])
    
    if not data_rows:
        return ""
    
    text_parts = []
    
    # Extract years and units from headers
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
        values = []
        
        for i in range(1, len(row)):
            cell = str(row[i]).strip() if row[i] else ""
            if cell and cell != "":
                values.append(cell)
        
        if not values:
            continue
        
        # Handle row labels with context
        if not row_label or row_label.strip() == "":
            row_label = f"Total {context}" if context else "Total"
        elif context and row_label.lower() not in ['total', 'subtotal', 'grand total']:
            row_label = f"{row_label} {context}"
        elif context and row_label.lower() in ['total', 'subtotal']:
            row_label = f"{row_label} {context}"
            
        # Generate natural language
        if len(years) >= 2 and len(values) >= 2:
            text_parts.append(f"{row_label} for {years[0]} is {values[0]} {units} and for {years[1]} is {values[1]} {units}")
        elif len(values) == 1:
            if years:
                text_parts.append(f"{row_label} for {years[0]} is {values[0]} {units}")
            else:
                text_parts.append(f"{row_label} is {values[0]} {units}")
    
    return ". ".join(text_parts) + "." if text_parts else ""

def convert_tables_to_text(text: str) -> str:
    """Convert all tables in text to natural language"""
    table_patterns = detect_table_patterns(text)
    
    if not table_patterns:
        return text
    
    # Sort by position (end to start) to avoid offset issues
    table_patterns.sort(key=lambda x: x['start'], reverse=True)
    
    converted_text = text
    
    for table_pattern in table_patterns:
        try:
            # Extract pre-table context
            pre_table_text = ""
            context_start = max(0, table_pattern['start'] - 150)
            context = text[context_start:table_pattern['start']]
            
            # Look for context pattern
            pre_table_match = re.search(r'(\d+\.\s*)?([^\n]+?)\s*$', context.strip())
            if pre_table_match:
                pre_table_text = pre_table_match.group(2).strip()
                pre_table_text = re.sub(r'^\d+\.\s*', '', pre_table_text)
                pre_table_text = pre_table_text.strip()
            
            # Parse and convert table
            parsed_table = parse_financial_table(table_pattern['data'])
            table_text = convert_table_to_text(parsed_table, pre_table_text)
            
            if table_text:
                # Replace table with natural language text
                converted_text = (converted_text[:table_pattern['start']] + 
                               table_text + 
                               converted_text[table_pattern['end']:])
        except Exception:
            continue
    
    return converted_text

def extract_semantic_terms(text):
    """Extract meaningful terms for semantic analysis"""
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    # Clean text and extract terms
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    terms = [word for word in text_clean.split() if word not in stop_words and len(word) > 2]
    
    # Include important phrases (2-3 word combinations)
    phrases = []
    words = text_clean.split()
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if len(phrase) > 6:
            phrases.append(phrase)
    
    return terms + phrases

def calculate_definition_similarity(text, definition):
    """
    Calculate similarity between document text and concept definition
    
    Args:
        text: Document text
        definition: Concept definition
        
    Returns:
        float: Similarity score (0-1)
    """
    if not definition or not text:
        return 0.0
    
    # Extract terms from both texts
    text_terms = set(extract_semantic_terms(text))
    def_terms = set(extract_semantic_terms(definition))
    
    if not def_terms:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(text_terms.intersection(def_terms))
    union = len(text_terms.union(def_terms))
    
    return intersection / union if union > 0 else 0.0

def contextual_keyword_matching(text, concept_info):
    """
    Enhanced keyword matching with context awareness
    
    Args:
        text: Document text
        concept_info: Concept information from R4S
        
    Returns:
        tuple: (score, matched_keywords)
    """
    keywords = concept_info.get("keywords", [])
    if not keywords:
        return 0.0, []
    
    text_lower = text.lower()
    matched_keywords = []
    total_matches = 0
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        # Count exact matches and partial matches
        exact_matches = text_lower.count(keyword_lower)
        if exact_matches > 0:
            matched_keywords.append(keyword)
            total_matches += exact_matches
    
    # Normalize by keyword count and text length
    keyword_score = min(1.0, (total_matches / len(keywords)) * (len(matched_keywords) / max(len(keywords), 1)))
    
    return keyword_score, matched_keywords

def enhanced_concept_analysis(text, domain_concepts):
    """
    Perform semantic analysis on domain-filtered concepts
    
    Args:
        text: Document text
        domain_concepts: Domain-specific concepts from get_domain_concepts
        
    Returns:
        dict: Concept analysis results
    """
    concept_matches = {}
    
    for concept_name, concept_info in domain_concepts.items():
        # Definition-based similarity
        definition_similarity = calculate_definition_similarity(
            text, 
            concept_info.get("definition", "")
        )
        
        # Contextual keyword matching
        keyword_score, keyword_matches = contextual_keyword_matching(
            text, 
            concept_info
        )
        
        # Combined concept score (weighted)
        total_concept_score = (definition_similarity * 0.4 + keyword_score * 0.6)
        
        # Apply threshold for relevance
        if total_concept_score > 0.1:
            concept_matches[concept_name] = {
                "total_score": total_concept_score,
                "definition_similarity": definition_similarity,
                "keyword_score": keyword_score,
                "keyword_matches": keyword_matches,
                "definition": concept_info.get("definition", "")
            }
    
    return concept_matches

def enrich_document_concepts(doc, r4s_ontology, r1_domains):
    """
    Main enrichment orchestration function
    
    Args:
        doc: Document from A1.1 with authoritative domain
        r4s_ontology: R4S semantic ontology
        r1_domains: R1 domain structure
        
    Returns:
        dict: Enriched document with concept analysis
    """
    # Stage 1: Domain Authority Inheritance
    domain = doc.get("domain", "unknown")
    doc_id = doc.get("doc_id", "")
    text = doc.get("text", "")
    
    # Stage 2: Domain-Specific Concept Filtering
    domain_concepts = get_domain_concepts(domain, r4s_ontology, r1_domains)
    
    if not domain_concepts:
        # Fallback for unknown/general domains
        return {
            **doc,  # Preserve original A1.1 data
            "matched_concepts": [],
            "concept_definitions": {},
            "matched_keywords": {},
            "semantic_richness_score": 0,
            "concept_match_count": 0,
            "domain_source": "a11_inheritance",
            "concept_source": "no_domain_concepts",
            "concept_analysis_performed": False
        }
    
    # Stage 3: Enhanced Semantic Analysis  
    concept_matches = enhanced_concept_analysis(text, domain_concepts)
    
    # Stage 4: Concept Enrichment Generation
    matched_concepts = list(concept_matches.keys())
    concept_definitions = {k: v["definition"] for k, v in concept_matches.items()}
    matched_keywords = {k: v["keyword_matches"] for k, v in concept_matches.items() if v["keyword_matches"]}
    
    # Calculate semantic richness score
    semantic_richness_score = sum(v["total_score"] for v in concept_matches.values())
    
    # Stage 5: Cross-Pipeline Intelligence Integration
    enriched_doc = {
        **doc,  # Preserve all original A1.1 data including metadata
        "matched_concepts": matched_concepts,
        "concept_definitions": concept_definitions,
        "matched_keywords": matched_keywords,
        "semantic_richness_score": round(semantic_richness_score, 2),
        "domain_specific_concepts": [c for c in matched_concepts if c.startswith(domain + ".")],
        "concept_match_count": len(matched_concepts),
        "domain_source": "a11_inheritance", 
        "concept_source": "r4s_semantic_analysis",
        "concept_analysis_performed": True,
        "available_domain_concepts": len(domain_concepts)
    }
    
    return enriched_doc

def process_documents(a11_data):
    """
    Process all documents with domain-aware concept enrichment
    
    Args:
        a11_data: A1.1 document data with authoritative domains
        
    Returns:
        dict: Processed documents with concept enrichment
    """
    # Load R-Pipeline resources
    r4s_ontology = load_r4s_ontology()
    r1_domains = load_r1_domains()
    
    documents = a11_data.get("documents", [])
    enriched_documents = []
    
    # Processing statistics
    total_concepts_matched = 0
    total_semantic_score = 0
    domain_analysis_stats = defaultdict(int)
    
    print(f"Enriching {len(documents)} documents with domain-specific concept analysis...")
    
    for doc in documents:
        domain = doc.get("domain", "unknown")
        domain_analysis_stats[domain] += 1
        
        # Enrich document with concept analysis
        enriched_doc = enrich_document_concepts(doc, r4s_ontology, r1_domains)
        enriched_documents.append(enriched_doc)
        
        # Update statistics
        total_concepts_matched += enriched_doc.get("concept_match_count", 0)
        total_semantic_score += enriched_doc.get("semantic_richness_score", 0)
        
        print(f"  {enriched_doc['doc_id']} ({domain}): {enriched_doc.get('concept_match_count', 0)} concepts")
    
    return {
        "documents": enriched_documents,
        "count": len(enriched_documents),
        "processing_statistics": {
            "total_concepts_matched": total_concepts_matched,
            "avg_concepts_per_doc": round(total_concepts_matched / max(len(documents), 1), 1),
            "total_semantic_richness": round(total_semantic_score, 2),
            "avg_semantic_richness": round(total_semantic_score / max(len(documents), 1), 2),
            "domain_analysis_distribution": dict(domain_analysis_stats),
            "r4s_concepts_available": len(r4s_ontology.get("concepts", {})),
            "r1_domains_available": len(r1_domains.get("domains", {}))
        },
        "a11_integration": "enabled",
        "concept_enrichment": "domain_aware",
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/A1.2_concept_enriched_documents.json"):
    """Save concept-enriched documents"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved concept-enriched documents to {full_path}")
    
    # Save metadata
    meta_path = full_path.with_suffix('.meta.json')
    metadata = {
        "script": "A1.2_domain_concept_enrichment.py",
        "timestamp": data["processing_timestamp"],
        "document_count": data["count"],
        "concept_enrichment": data["concept_enrichment"],
        "a11_integration": data["a11_integration"],
        "processing_statistics": data["processing_statistics"],
        "output_file": str(full_path)
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main execution - Domain Concept Enrichment Engine"""
    print("="*80)
    print("A1.2: Domain Concept Enrichment Engine")
    print("="*80)
    
    try:
        # Load A1.1 documents with authoritative domains
        print("Loading documents from A1.1 with authoritative domain classifications...")
        a11_data = load_a11_documents()
        
        print(f"Loaded {a11_data['count']} documents with domain classifications:")
        domain_stats = a11_data.get("domain_classification", {})
        for domain, count in domain_stats.items():
            print(f"  {domain}: {count} documents")
        
        # Process documents with concept enrichment
        enriched_data = process_documents(a11_data)
        
        # Display enrichment results
        stats = enriched_data["processing_statistics"]
        print(f"\nConcept Enrichment Results:")
        print(f"  Total concepts matched: {stats['total_concepts_matched']}")
        print(f"  Average concepts/document: {stats['avg_concepts_per_doc']}")
        print(f"  Average semantic richness: {stats['avg_semantic_richness']}")
        print(f"  R4S concepts analyzed: {stats['r4s_concepts_available']}")
        
        print(f"\nDomain-Specific Analysis Distribution:")
        for domain, count in stats["domain_analysis_distribution"].items():
            print(f"  {domain}: {count} documents analyzed")
        
        # Save enriched documents
        save_output(enriched_data)
        
        print(f"\nA1.2 Domain Concept Enrichment completed successfully!")
        
    except Exception as e:
        print(f"Error in A1.2 Concept Enrichment: {str(e)}")
        raise

if __name__ == "__main__":
    main()