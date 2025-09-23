#!/usr/bin/env python3
"""
A2.5.6: Document-Scoped Vicinity Concept Discovery
Finds concepts adjacent to A2.4 core concepts within the same document context
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import spacy
from typing import List, Dict, Tuple, Set

class VicinityConceptDiscovery:
    """
    Discovers concepts in the vicinity of core concepts within document context
    """

    def __init__(self):
        self.context_window = 40  # Words before/after core concept keywords
        self.min_concept_length = 2  # Minimum words in a vicinity concept
        self.min_frequency = 2  # Minimum occurrences to be considered
        self.max_concepts_per_core = 10  # Maximum vicinity concepts per core concept

        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            print("[WARNING] spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def find_keyword_locations(self, text: str, keywords: List[str]) -> List[Tuple[int, int, str]]:
        """
        Find all locations where core concept keywords appear in text

        Returns:
            List of (start_pos, end_pos, matched_keyword) tuples
        """
        locations = []
        text_lower = text.lower()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            start_pos = 0

            while True:
                pos = text_lower.find(keyword_lower, start_pos)
                if pos == -1:
                    break

                locations.append((pos, pos + len(keyword), keyword))
                start_pos = pos + 1

        return sorted(locations)

    def extract_context_windows(self, text: str, locations: List[Tuple[int, int, str]]) -> str:
        """
        Extract context windows around keyword locations and merge overlapping ones
        """
        if not locations:
            return ""

        words = text.split()
        word_positions = []
        current_pos = 0

        # Map character positions to word indices
        for i, word in enumerate(words):
            word_start = text.find(word, current_pos)
            word_end = word_start + len(word)
            word_positions.append((word_start, word_end, i))
            current_pos = word_end

        context_ranges = []

        for char_start, char_end, keyword in locations:
            # Find word index for this character position
            word_idx = 0
            for ws, we, wi in word_positions:
                if ws <= char_start < we:
                    word_idx = wi
                    break

            # Extract context window
            start_idx = max(0, word_idx - self.context_window)
            end_idx = min(len(words), word_idx + self.context_window)
            context_ranges.append((start_idx, end_idx))

        # Merge overlapping ranges
        merged_ranges = []
        for start, end in sorted(context_ranges):
            if merged_ranges and start <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
            else:
                merged_ranges.append((start, end))

        # Extract text from merged ranges
        context_texts = []
        for start_idx, end_idx in merged_ranges:
            context_texts.append(" ".join(words[start_idx:end_idx]))

        return " ".join(context_texts)

    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text using spaCy
        """
        if not self.nlp:
            # Fallback: simple pattern matching
            return self.extract_phrases_fallback(text)

        doc = self.nlp(text)
        phrases = []

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if len(phrase.split()) >= self.min_concept_length:
                phrases.append(phrase)

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'MONEY', 'LAW', 'PRODUCT', 'EVENT']:
                phrases.append(ent.text.strip())

        return phrases

    def extract_phrases_fallback(self, text: str) -> List[str]:
        """
        Fallback phrase extraction when spaCy is not available
        """
        phrases = []

        # Pattern for financial/business terms
        patterns = [
            r'\b[A-Z][A-Z0-9\s]{2,20}\b',  # Uppercase abbreviations (GAAP, ASC 606)
            r'\b\w+\s+(?:standard|policy|principle|method|approach|framework)\b',
            r'\b(?:according to|under|based on|pursuant to)\s+[\w\s]{2,15}\b',
            r'\b\d+[A-Z]*\s+[\w\s]{2,15}\b',  # Standards like "15 Revenue from Contracts"
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase = match.group().strip()
                if len(phrase.split()) >= self.min_concept_length:
                    phrases.append(phrase)

        return phrases

    def filter_and_score_concepts(self,
                                 vicinity_candidates: List[str],
                                 core_keywords: List[str],
                                 context_text: str) -> Dict[str, float]:
        """
        Filter vicinity candidates and score them based on relevance
        """
        concept_scores = {}
        core_keywords_lower = [k.lower() for k in core_keywords]

        # Count frequencies
        candidate_counts = Counter(vicinity_candidates)

        for candidate, frequency in candidate_counts.items():
            candidate_lower = candidate.lower()

            # Skip if frequency too low
            if frequency < self.min_frequency:
                continue

            # Skip if it's a core keyword or substring
            if any(candidate_lower in core.lower() or core.lower() in candidate_lower
                   for core in core_keywords_lower):
                continue

            # Skip generic terms
            if self.is_generic_term(candidate):
                continue

            # Calculate score
            score = self.calculate_vicinity_score(candidate, core_keywords, context_text, frequency)

            if score > 0.1:  # Minimum score threshold
                concept_scores[candidate] = score

        return concept_scores

    def is_generic_term(self, term: str) -> bool:
        """
        Check if term is too generic to be a useful vicinity concept
        """
        generic_patterns = [
            r'^\w{1,2}$',  # Very short terms
            r'^(the|and|for|with|this|that|such|other|more|most|some|any)\b',
            r'^(year|years|time|period|amount|number|total|part|way|use)\b',
            r'^(company|business|group|entity|organization)\b',  # Too generic business terms
        ]

        term_lower = term.lower()
        return any(re.match(pattern, term_lower) for pattern in generic_patterns)

    def calculate_vicinity_score(self, candidate: str, core_keywords: List[str],
                                context_text: str, frequency: int) -> float:
        """
        Calculate relevance score for a vicinity concept
        """
        score = 0.0
        candidate_lower = candidate.lower()
        context_lower = context_text.lower()

        # Frequency score (normalized)
        score += min(frequency / 10.0, 0.3)

        # Length preference (2-4 words optimal)
        word_count = len(candidate.split())
        if 2 <= word_count <= 4:
            score += 0.2
        elif word_count > 4:
            score -= 0.1

        # Domain relevance patterns
        domain_patterns = [
            r'\b(standard|policy|principle|method|framework|approach)\b',
            r'\b(revenue|income|cost|expense|liability|asset)\b',
            r'\b(contract|agreement|obligation|commitment)\b',
            r'\b(accounting|financial|reporting|disclosure)\b',
            r'\b[A-Z]{2,}\s*\d+\b',  # Standards like "AASB 15", "ASC 606"
        ]

        for pattern in domain_patterns:
            if re.search(pattern, candidate_lower):
                score += 0.3
                break

        # Proximity to core keywords (how often they appear together)
        proximity_score = 0.0
        for keyword in core_keywords:
            # Count co-occurrences in nearby text
            keyword_lower = keyword.lower()
            if keyword_lower in context_lower and candidate_lower in context_lower:
                proximity_score += 0.1

        score += min(proximity_score, 0.4)

        return min(score, 1.0)

    def discover_vicinity_concepts(self, document: Dict, core_concepts: List[Dict]) -> Dict:
        """
        Main method to discover vicinity concepts for a document
        """
        doc_id = document.get('doc_id', 'unknown')
        doc_text = document.get('text', document.get('content', ''))

        if not doc_text:
            return {'document_id': doc_id, 'vicinity_concepts': {}, 'statistics': {}}

        vicinity_results = {}
        total_vicinity_concepts = 0

        for core_concept in core_concepts:
            concept_id = core_concept.get('concept_id', '')
            concept_name = core_concept.get('canonical_name', '')
            core_keywords = core_concept.get('keywords', [])

            if not core_keywords:
                continue

            # Find keyword locations in document
            locations = self.find_keyword_locations(doc_text, core_keywords)

            if not locations:
                continue

            # Extract context windows
            context_text = self.extract_context_windows(doc_text, locations)

            # Extract vicinity candidates
            vicinity_candidates = self.extract_noun_phrases(context_text)

            # Filter and score concepts
            concept_scores = self.filter_and_score_concepts(
                vicinity_candidates, core_keywords, context_text
            )

            # Select top vicinity concepts
            top_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
            top_concepts = top_concepts[:self.max_concepts_per_core]

            if top_concepts:
                vicinity_results[concept_id] = {
                    'core_concept_name': concept_name,
                    'core_keywords': core_keywords,
                    'vicinity_concepts': [concept for concept, score in top_concepts],
                    'concept_scores': dict(top_concepts),
                    'context_locations': len(locations),
                    'context_length': len(context_text.split())
                }
                total_vicinity_concepts += len(top_concepts)

        statistics = {
            'document_id': doc_id,
            'core_concepts_processed': len([c for c in core_concepts if c.get('keywords')]),
            'vicinity_concepts_found': total_vicinity_concepts,
            'avg_vicinity_per_core': total_vicinity_concepts / max(len(vicinity_results), 1),
            'concepts_with_vicinity': len(vicinity_results)
        }

        return {
            'document_id': doc_id,
            'vicinity_concepts': vicinity_results,
            'statistics': statistics
        }

def load_input_data():
    """Load A2.4 core concepts and documents"""
    script_dir = Path(__file__).parent.parent

    # Load A2.4 core concepts
    a24_path = script_dir / "outputs/A2.4_core_concepts.json"
    with open(a24_path, 'r', encoding='utf-8') as f:
        a24_data = json.load(f)

    # Load documents (preprocessed)
    doc_path = script_dir / "outputs/A2.1_preprocessed_documents.json"
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)

    return a24_data, doc_data

def process_vicinity_discovery():
    """Main processing function"""
    print("="*60)
    print("A2.5.6: Document-Scoped Vicinity Concept Discovery")
    print("="*60)

    # Load input data
    a24_data, doc_data = load_input_data()

    # Initialize discovery engine
    discovery = VicinityConceptDiscovery()

    # Process each document
    all_results = []
    total_vicinity_concepts = 0

    for doc in doc_data['documents']:
        doc_id = doc['doc_id']

        # Find core concepts for this document
        doc_core_concepts = []
        for a24_doc in a24_data['documents']:
            if a24_doc['doc_id'] == doc_id:
                doc_core_concepts = a24_doc['core_concepts']
                break

        if not doc_core_concepts:
            continue

        print(f"\nProcessing {doc_id}: {len(doc_core_concepts)} core concepts")

        # Discover vicinity concepts
        result = discovery.discover_vicinity_concepts(doc, doc_core_concepts)
        all_results.append(result)

        stats = result['statistics']
        total_vicinity_concepts += stats['vicinity_concepts_found']

        print(f"  Found {stats['vicinity_concepts_found']} vicinity concepts")
        print(f"  Coverage: {stats['concepts_with_vicinity']}/{stats['core_concepts_processed']} core concepts")

    # Generate summary
    summary = {
        'total_documents': len(all_results),
        'total_vicinity_concepts': total_vicinity_concepts,
        'avg_vicinity_per_document': total_vicinity_concepts / max(len(all_results), 1),
        'successful_documents': len([r for r in all_results if r['statistics']['vicinity_concepts_found'] > 0])
    }

    # Save results
    output_data = {
        'strategy_name': 'vicinity_concepts',
        'results': {
            'vicinity_discoveries': all_results,
            'summary': summary
        },
        'processing_timestamp': datetime.now().isoformat()
    }

    output_path = Path(__file__).parent.parent / "outputs/A2.5.6_vicinity_concepts.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("VICINITY DISCOVERY SUMMARY:")
    print(f"Documents processed: {summary['total_documents']}")
    print(f"Total vicinity concepts: {summary['total_vicinity_concepts']}")
    print(f"Average per document: {summary['avg_vicinity_per_document']:.1f}")
    print(f"Success rate: {summary['successful_documents']}/{summary['total_documents']}")
    print(f"\nSaved to: {output_path}")

    # Show sample results
    print(f"\nSAMPLE VICINITY DISCOVERIES:")
    for result in all_results[:2]:
        if result['vicinity_concepts']:
            print(f"\n{result['document_id']}:")
            for concept_id, vicinity in list(result['vicinity_concepts'].items())[:2]:
                core_name = vicinity['core_concept_name']
                vicinity_list = vicinity['vicinity_concepts'][:3]
                print(f"  Core: '{core_name}' → Vicinity: {vicinity_list}")

def main():
    """Main execution"""
    try:
        process_vicinity_discovery()
        print("\nA2.5.6 Vicinity Concept Discovery completed successfully!")
    except Exception as e:
        print(f"Error in A2.5.6: {str(e)}")
        raise

if __name__ == "__main__":
    main()