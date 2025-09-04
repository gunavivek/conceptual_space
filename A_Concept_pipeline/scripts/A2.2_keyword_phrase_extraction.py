#!/usr/bin/env python3
"""
A2.2: Ensemble Keyword Extraction (ENHANCED)

Multi-method keyword extraction using KeyBERT, YAKE, Direct extraction, and Lemma reconstruction
for domain-agnostic, business-relevant, meaningful term identification.

Processing Methods:
    Method 1: KeyBERT - Semantic keyword extraction using BERT embeddings
    Method 2: YAKE - Statistical and linguistic keyword extraction
    Method 3: Direct Extraction - Rule-based business term identification
    Method 4: Lemma Reconstruction - Rebuilding terms from normalized lemmas
    Method 5: Ensemble Fusion - Combining and scoring results

Input: outputs/A2.1_preprocessed_documents.json
Output: outputs/A2.2_keyword_extractions.json
"""

import json
import re
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set
import math

# Try to import optional libraries for ensemble extraction
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    warnings.warn("KeyBERT not available. Install with: pip install keybert")

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    warnings.warn("YAKE not available. Install with: pip install yake")

def get_business_term_dictionary() -> Dict[str, Set[str]]:
    """
    Get comprehensive business term dictionary for filtering and validation
    
    Returns:
        Dict: Categories of business terms for domain-agnostic extraction
    """
    return {
        'financial': {
            'revenue', 'income', 'expense', 'cost', 'profit', 'loss', 'asset', 'liability',
            'equity', 'balance', 'payment', 'transaction', 'account', 'deferred', 'accrued',
            'depreciation', 'amortization', 'interest', 'dividend', 'capital', 'cash', 'debt'
        },
        'business_operations': {
            'contract', 'customer', 'client', 'service', 'product', 'operation', 'business',
            'company', 'organization', 'management', 'strategy', 'process', 'procedure',
            'policy', 'compliance', 'regulation', 'audit', 'control', 'governance'
        },
        'temporal': {
            'quarter', 'annual', 'monthly', 'period', 'year', 'fiscal', 'reporting',
            'current', 'non-current', 'long-term', 'short-term', 'maturity'
        },
        'quantitative': {
            'amount', 'total', 'sum', 'average', 'percentage', 'rate', 'ratio', 'margin',
            'volume', 'quantity', 'measure', 'metric', 'indicator', 'benchmark'
        },
        'legal_regulatory': {
            'agreement', 'clause', 'provision', 'requirement', 'standard', 'framework',
            'guideline', 'principle', 'rule', 'law', 'statute', 'regulation'
        }
    }

def is_valid_business_term(term: str, business_dict: Dict[str, Set[str]]) -> bool:
    """
    Check if term is a valid business term using comprehensive filtering
    
    Args:
        term: Term to validate
        business_dict: Business term dictionary
        
    Returns:
        bool: Whether term is valid business term
    """
    term_lower = term.lower().strip()
    
    # Filter out invalid patterns
    if not term_lower or len(term_lower) < 2:
        return False
    
    # Filter out pure numbers and currency amounts
    if re.match(r'^[\d,\.\$€£¥]+$', term_lower):
        return False
    
    # Filter out single characters and punctuation
    if len(term_lower) <= 1 or re.match(r'^[^\w\s]+$', term_lower):
        return False
    
    # Filter out terms ending with punctuation
    if re.match(r'.*[\.,:;!?\)\]\}]$', term_lower):
        return False
    
    # Filter out years (4-digit numbers)
    if re.match(r'^(19|20)\d{2}$', term_lower):
        return False
    
    # Check if term contains business keywords
    for category, terms in business_dict.items():
        if any(business_term in term_lower for business_term in terms):
            return True
    
    # Check if it's a compound noun that looks business-relevant
    words = term_lower.split()
    if len(words) >= 2:
        # Multi-word terms are more likely to be business concepts
        return True
    
    # Single words need to be substantive (not too common)
    common_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'over', 'after', 'under', 'again',
        'above', 'below', 'between', 'through', 'during', 'before', 'after'
    }
    
    return term_lower not in common_words and len(term_lower) >= 3

def extract_keybert_keywords(text: str, max_keywords: int = 30) -> List[Tuple[str, float]]:
    """
    Extract keywords using KeyBERT semantic extraction
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of (keyword, score) tuples
    """
    if not KEYBERT_AVAILABLE or not text.strip():
        return []
    
    try:
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),  # 1-3 word phrases
            stop_words='english',
            use_mmr=True,  # Use Maximal Marginal Relevance for diversity
            diversity=0.5,
            top_k=max_keywords
        )
        
        # Filter for business relevance
        business_dict = get_business_term_dictionary()
        filtered_keywords = []
        
        for keyword, score in keywords:
            if is_valid_business_term(keyword, business_dict):
                filtered_keywords.append((keyword, score))
        
        return filtered_keywords[:max_keywords//2]  # Take top half after filtering
        
    except Exception as e:
        print(f"[WARNING] KeyBERT extraction failed: {str(e)}")
        return []

def extract_yake_keywords(text: str, max_keywords: int = 30) -> List[Tuple[str, float]]:
    """
    Extract keywords using YAKE statistical extraction
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of (keyword, score) tuples (note: YAKE uses inverse scoring - lower is better)
    """
    if not YAKE_AVAILABLE or not text.strip():
        return []
    
    try:
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # max n-gram size
            dedupLim=0.7,  # deduplication threshold
            top=max_keywords * 2  # Extract more to filter
        )
        
        keywords = kw_extractor.extract_keywords(text)
        
        # Filter for business relevance and convert YAKE scores (invert for consistency)
        business_dict = get_business_term_dictionary()
        filtered_keywords = []
        
        for keyword, yake_score in keywords:
            if is_valid_business_term(keyword, business_dict):
                # Convert YAKE score (lower=better) to standard score (higher=better)
                normalized_score = max(0, 1 - (yake_score / 10))
                filtered_keywords.append((keyword, normalized_score))
        
        # Sort by converted score (descending)
        filtered_keywords.sort(key=lambda x: x[1], reverse=True)
        return filtered_keywords[:max_keywords//2]
        
    except Exception as e:
        print(f"[WARNING] YAKE extraction failed: {str(e)}")
        return []

def extract_direct_terms(pos_tags: List[Dict], noun_phrases: List[Dict], business_terms: List[str]) -> List[Tuple[str, float]]:
    """
    Extract terms using direct rule-based patterns
    
    Args:
        pos_tags: POS tag data from A2.1
        noun_phrases: Noun phrase data from A2.1
        business_terms: Pre-extracted business terms from A2.1
        
    Returns:
        List of (term, score) tuples
    """
    direct_terms = []
    business_dict = get_business_term_dictionary()
    
    # Method 1: Use pre-extracted business terms from A2.1 (highest confidence)
    for term in business_terms:
        if is_valid_business_term(term, business_dict):
            direct_terms.append((term, 0.9))  # High confidence score
    
    # Method 2: Extract compound nouns from POS tags
    for i in range(len(pos_tags) - 1):
        if (pos_tags[i]['pos'] == 'NOUN' and 
            pos_tags[i+1]['pos'] == 'NOUN' and
            not pos_tags[i]['is_stop'] and 
            not pos_tags[i+1]['is_stop']):
            
            compound = f"{pos_tags[i]['token']} {pos_tags[i+1]['token']}"
            if is_valid_business_term(compound, business_dict):
                direct_terms.append((compound, 0.7))
    
    # Method 3: Extract adjective-noun patterns
    for i in range(len(pos_tags) - 1):
        if (pos_tags[i]['pos'] in ['ADJ', 'ADJECTIVE'] and 
            pos_tags[i+1]['pos'] == 'NOUN' and
            not pos_tags[i]['is_stop']):
            
            adj_noun = f"{pos_tags[i]['token']} {pos_tags[i+1]['token']}"
            if is_valid_business_term(adj_noun, business_dict):
                direct_terms.append((adj_noun, 0.6))
    
    # Method 4: Extract relevant noun phrases
    for np in noun_phrases:
        phrase = np['text']
        if is_valid_business_term(phrase, business_dict) and len(phrase.split()) <= 4:
            direct_terms.append((phrase, 0.8))
    
    # Deduplicate and return
    unique_terms = {}
    for term, score in direct_terms:
        if term not in unique_terms or unique_terms[term] < score:
            unique_terms[term] = score
    
    return [(term, score) for term, score in unique_terms.items()]

def reconstruct_from_lemmas(lemmatized_tokens: List[str], token_mapping: Dict[str, str], text: str) -> List[Tuple[str, float]]:
    """
    Reconstruct meaningful terms from lemmatized tokens
    
    Args:
        lemmatized_tokens: List of lemmatized tokens
        token_mapping: Original to lemma mapping
        text: Original text for context
        
    Returns:
        List of (term, score) tuples
    """
    if not lemmatized_tokens:
        return []
    
    reconstructed_terms = []
    business_dict = get_business_term_dictionary()
    
    # Method 1: Find business lemmas and map back to original forms
    for original, lemma in token_mapping.items():
        if any(business_term == lemma for category in business_dict.values() for business_term in category):
            # This lemma is a business term - prefer original form if different
            term = original if original != lemma else lemma
            if is_valid_business_term(term, business_dict):
                reconstructed_terms.append((term, 0.6))
    
    # Method 2: Look for adjacent business lemmas to reconstruct compound terms
    for i in range(len(lemmatized_tokens) - 1):
        lemma1, lemma2 = lemmatized_tokens[i], lemmatized_tokens[i+1]
        
        # Check if both lemmas relate to business
        if (any(business_term == lemma1 for category in business_dict.values() for business_term in category) or
            any(business_term == lemma2 for category in business_dict.values() for business_term in category)):
            
            compound_lemma = f"{lemma1} {lemma2}"
            if is_valid_business_term(compound_lemma, business_dict):
                reconstructed_terms.append((compound_lemma, 0.5))
    
    # Method 3: Count business lemma frequencies
    lemma_counts = Counter(lemmatized_tokens)
    for lemma, count in lemma_counts.most_common(20):
        if any(business_term == lemma for category in business_dict.values() for business_term in category):
            if is_valid_business_term(lemma, business_dict):
                # Score based on frequency
                score = min(0.8, 0.3 + (count * 0.1))
                reconstructed_terms.append((lemma, score))
    
    return reconstructed_terms

def ensemble_merge_keywords(keybert_kw: List[Tuple], yake_kw: List[Tuple], 
                          direct_kw: List[Tuple], reconstructed_kw: List[Tuple]) -> List[Dict]:
    """
    Merge keywords from all methods using ensemble scoring
    
    Args:
        keybert_kw: KeyBERT keywords
        yake_kw: YAKE keywords  
        direct_kw: Direct extraction keywords
        reconstructed_kw: Reconstructed keywords
        
    Returns:
        List of keyword dictionaries with ensemble scores
    """
    # Combine all keywords with method attribution
    all_keywords = defaultdict(lambda: {'scores': [], 'methods': []})
    
    # Add KeyBERT results (weight: 0.3)
    for term, score in keybert_kw:
        all_keywords[term.lower()]['scores'].append(score * 0.3)
        all_keywords[term.lower()]['methods'].append('keybert')
        all_keywords[term.lower()]['term'] = term  # Preserve original case
    
    # Add YAKE results (weight: 0.25)
    for term, score in yake_kw:
        all_keywords[term.lower()]['scores'].append(score * 0.25)
        all_keywords[term.lower()]['methods'].append('yake')
        if 'term' not in all_keywords[term.lower()]:
            all_keywords[term.lower()]['term'] = term
    
    # Add Direct results (weight: 0.3)
    for term, score in direct_kw:
        all_keywords[term.lower()]['scores'].append(score * 0.3)
        all_keywords[term.lower()]['methods'].append('direct')
        if 'term' not in all_keywords[term.lower()]:
            all_keywords[term.lower()]['term'] = term
    
    # Add Reconstructed results (weight: 0.15)
    for term, score in reconstructed_kw:
        all_keywords[term.lower()]['scores'].append(score * 0.15)
        all_keywords[term.lower()]['methods'].append('reconstructed')
        if 'term' not in all_keywords[term.lower()]:
            all_keywords[term.lower()]['term'] = term
    
    # Calculate ensemble scores
    final_keywords = []
    for term_key, data in all_keywords.items():
        if 'term' in data:
            # Ensemble score = sum of weighted scores + method diversity bonus
            base_score = sum(data['scores'])
            method_diversity = len(set(data['methods'])) * 0.05  # Bonus for multiple methods
            ensemble_score = base_score + method_diversity
            
            final_keywords.append({
                'term': data['term'],
                'score': min(1.0, ensemble_score),  # Cap at 1.0
                'methods': data['methods'],
                'method_count': len(data['methods'])
            })
    
    # Sort by ensemble score and return top keywords
    final_keywords.sort(key=lambda x: x['score'], reverse=True)
    return final_keywords[:50]  # Top 50 keywords

def process_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single document using ensemble keyword extraction
    
    Args:
        doc: Document data from A2.1
        
    Returns:
        Document with extracted keywords
    """
    doc_id = doc.get('doc_id', '')
    text = doc.get('cleaned_text', '')
    
    # Get linguistic data from A2.1
    lemmatization = doc.get('lemmatization', {})
    pos_tags = doc.get('pos_tags', [])
    noun_phrases = doc.get('noun_phrases', [])
    business_terms = doc.get('extracted_business_terms', [])
    
    lemmatized_tokens = lemmatization.get('lemmatized_tokens', [])
    token_mapping = lemmatization.get('token_mapping', {})
    
    print(f"\\nProcessing document: {doc_id}")
    print(f"  Text length: {len(text)} chars")
    print(f"  Lemmatized tokens: {len(lemmatized_tokens)}")
    print(f"  POS tags: {len(pos_tags)}")
    print(f"  Noun phrases: {len(noun_phrases)}")
    print(f"  Pre-extracted business terms: {len(business_terms)}")
    
    # Method 1: KeyBERT extraction
    keybert_keywords = extract_keybert_keywords(text)
    print(f"  KeyBERT keywords: {len(keybert_keywords)}")
    
    # Method 2: YAKE extraction  
    yake_keywords = extract_yake_keywords(text)
    print(f"  YAKE keywords: {len(yake_keywords)}")
    
    # Method 3: Direct extraction
    direct_keywords = extract_direct_terms(pos_tags, noun_phrases, business_terms)
    print(f"  Direct keywords: {len(direct_keywords)}")
    
    # Method 4: Lemma reconstruction
    reconstructed_keywords = reconstruct_from_lemmas(lemmatized_tokens, token_mapping, text)
    print(f"  Reconstructed keywords: {len(reconstructed_keywords)}")
    
    # Method 5: Ensemble fusion
    final_keywords = ensemble_merge_keywords(
        keybert_keywords, yake_keywords, direct_keywords, reconstructed_keywords
    )
    print(f"  Final ensemble keywords: {len(final_keywords)}")
    
    # Convert to A2.3-compatible format (simplified for clustering)
    keywords_for_clustering = []
    for kw in final_keywords[:30]:  # Top 30 for clustering
        keywords_for_clustering.append({
            'term': kw['term'],
            'score': kw['score']
        })
    
    # Add extraction metadata
    doc['keywords'] = keywords_for_clustering
    doc['keyword_extraction'] = {
        'method': 'ensemble',
        'methods_used': ['keybert', 'yake', 'direct', 'reconstructed'],
        'keybert_available': KEYBERT_AVAILABLE,
        'yake_available': YAKE_AVAILABLE,
        'total_extracted': len(final_keywords),
        'selected_for_clustering': len(keywords_for_clustering)
    }
    
    # Show sample results
    if keywords_for_clustering:
        sample = [kw['term'] for kw in keywords_for_clustering[:5]]
        print(f"  Sample keywords: {sample}")
    
    return doc

def load_input(input_path="outputs/A2.1_preprocessed_documents.json"):
    """Load preprocessed documents from A2.1"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("documents", [])

def save_output(data, output_path="outputs/A2.2_keyword_extractions.json"):
    """Save ensemble keyword extraction results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved ensemble keyword extractions to {full_path}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("A2.2: Ensemble Keyword Extraction (ENHANCED)")
    print("=" * 80)
    print("Loading preprocessed documents from A2.1...")
    
    # Check method availability
    print(f"KeyBERT available: {KEYBERT_AVAILABLE}")
    print(f"YAKE available: {YAKE_AVAILABLE}")
    
    if not KEYBERT_AVAILABLE and not YAKE_AVAILABLE:
        print("[WARNING] Neither KeyBERT nor YAKE available - using direct extraction only")
    
    # Load documents
    documents = load_input()
    print(f"Processing {len(documents)} documents with ensemble extraction...")
    
    # Process all documents
    for doc in documents:
        process_document(doc)
    
    # Create output structure
    results = {
        "documents": documents,
        "count": len(documents),
        "extraction_method": "ensemble",
        "methods_available": {
            "keybert": KEYBERT_AVAILABLE,
            "yake": YAKE_AVAILABLE,
            "direct": True,
            "reconstructed": True
        },
        "processing_timestamp": datetime.now().isoformat()
    }
    
    # Save results
    save_output(results)
    
    # Print summary
    total_keywords = sum(len(doc.get('keywords', [])) for doc in documents)
    avg_keywords = total_keywords / len(documents) if documents else 0
    
    print(f"\\nEnsemble Extraction Summary:")
    print(f"  Documents processed: {len(documents)}")
    print(f"  Total keywords extracted: {total_keywords}")
    print(f"  Average keywords per document: {avg_keywords:.1f}")
    print(f"  Methods used: KeyBERT={KEYBERT_AVAILABLE}, YAKE={YAKE_AVAILABLE}, Direct=True, Reconstructed=True")
    
    print(f"\\nA2.2 Ensemble Keyword Extraction completed successfully!")

if __name__ == "__main__":
    main()