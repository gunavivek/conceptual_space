#!/usr/bin/env python3
"""
A2.5: Semantic Vicinity Concept Discovery
Single script replacement for all A2.5 strategies
Discovers concepts semantically adjacent to A2.4 core concepts with document relevance validation
"""

import json
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from collections import defaultdict
import time

class A25_VicinityConceptDiscovery:
    """
    Single, focused A2.5 replacement for semantic vicinity concept discovery
    Replaces all previous A2.5 strategies with one effective approach
    """

    def __init__(self):
        self.similarity_threshold = 0.6
        self.relevance_threshold = 0.3  # Lowered from 0.5 to be less strict
        self.max_vicinity_per_concept = 8
        self.max_retries = 3
        self.concepts_discovered = 0

        # Initialize OpenAI client (assuming API key is set in environment)
        try:
            self.client = openai.OpenAI()
            print("OpenAI client initialized")
        except Exception as e:
            print(f"Warning: OpenAI client not available: {e}")
            print("  Will use fallback semantic discovery method")
            self.client = None

    async def generate_semantic_candidates(self,
                                         core_concept_name: str,
                                         core_keywords: List[str],
                                         domain: str = "financial_accounting") -> List[str]:
        """
        Generate semantic candidates for a core concept using LLM

        Args:
            core_concept_name: Name of the core concept
            core_keywords: Keywords associated with the core concept
            domain: Domain context for semantic discovery

        Returns:
            List of candidate vicinity concepts
        """
        if not self.client:
            return self.fallback_semantic_candidates(core_concept_name, core_keywords)

        prompt = f"""
You are an expert in {domain}. Given a core concept, find 10-12 semantically related concepts that would commonly appear in the same documents or contexts.

Core Concept: "{core_concept_name}"
Core Keywords: {', '.join(core_keywords[:5])}

Find concepts that are:
1. Semantically adjacent (related in meaning/purpose)
2. Professionally relevant in {domain}
3. Likely to co-occur in financial documents
4. NOT duplicates or simple variations of the core concept

Examples of good vicinity concepts for "revenue recognition":
- performance obligations
- contract modifications
- variable consideration
- transaction price allocation
- customer contract liabilities

Return ONLY a JSON array of 8-10 unique concept strings, no explanations:
["concept1", "concept2", ...]
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()
            # Parse JSON response
            if content.startswith('[') and content.endswith(']'):
                candidates = json.loads(content)
                return [c.strip() for c in candidates if isinstance(c, str) and len(c.strip()) > 2]
            else:
                print(f"Invalid JSON response for {core_concept_name}, using fallback")
                return self.fallback_semantic_candidates(core_concept_name, core_keywords)

        except Exception as e:
            print(f"LLM call failed for {core_concept_name}: {e}")
            return self.fallback_semantic_candidates(core_concept_name, core_keywords)

    def fallback_semantic_candidates(self, core_concept_name: str, core_keywords: List[str]) -> List[str]:
        """
        Document-driven fallback method for generating semantic candidates without LLM
        Uses the concept structure and keywords to generate related semantic terms
        """
        candidates = []

        # Generate semantic variants from the concept name itself
        concept_words = core_concept_name.lower().split()

        # Remove common stop words and connectors
        stop_words = {'&', 'and', 'or', 'the', 'a', 'an', 'of', 'to', 'for', 'with', 'in', 'on', 'at', 'by'}
        meaningful_words = [word for word in concept_words if word not in stop_words and len(word) > 2]

        # For each meaningful word in the concept, generate semantic variations
        for word in meaningful_words:
            # Generate conceptual variations (not domain-specific)
            if len(word) > 3:  # Only for substantive words
                # Add morphological variants
                candidates.extend([
                    f"{word} management",
                    f"{word} analysis",
                    f"{word} process",
                    f"{word} strategy",
                    f"{word} performance",
                    f"{word} evaluation"
                ])

        # Generate semantic candidates from keywords (document-driven)
        for keyword in core_keywords[:3]:
            keyword_words = keyword.lower().split()
            meaningful_keyword_words = [word for word in keyword_words if word not in stop_words and len(word) > 2]

            for word in meaningful_keyword_words:
                if len(word) > 3:
                    # Generate semantic neighbors based on the word's conceptual role
                    candidates.extend([
                        f"{word} framework",
                        f"{word} methodology",
                        f"{word} implementation",
                        f"{word} assessment",
                        f"{word} optimization"
                    ])

        # Generic semantic relationship patterns (domain-agnostic)
        if len(meaningful_words) > 0:
            base_word = meaningful_words[0]
            candidates.extend([
                f"related {base_word}",
                f"associated {base_word}",
                f"{base_word} standards",
                f"{base_word} guidelines",
                f"{base_word} principles",
                f"{base_word} requirements"
            ])

        # Remove duplicates and filter for reasonable length
        unique_candidates = list(set(candidates))
        filtered_candidates = [c for c in unique_candidates if 2 <= len(c.split()) <= 4 and len(c) > 5]

        return filtered_candidates[:10]

    async def evaluate_document_relevance(self,
                                        vicinity_candidates: List[str],
                                        core_concept: Dict[str, Any],
                                        document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate which vicinity candidates are relevant to the specific document

        Returns:
            List of {concept, relevance_score, reasoning} dictionaries
        """
        doc_content = document.get('content', document.get('text', ''))[:2000]  # Limit for efficiency
        core_name = core_concept.get('canonical_name', '')

        if not self.client:
            return self.fallback_document_relevance(vicinity_candidates, doc_content)

        prompt = f"""
Analyze which vicinity concepts are relevant to this financial document context.

Core Concept: "{core_name}"
Document Context: "{doc_content}"

Vicinity Candidates: {vicinity_candidates}

For each candidate, determine:
1. Is it semantically relevant to the core concept?
2. Is it contextually relevant to this document?
3. Score from 0.0 (irrelevant) to 1.0 (highly relevant)

Return ONLY a JSON array of objects:
[{{"concept": "candidate_name", "score": 0.8, "reasoning": "brief_reason"}}, ...]

Only include candidates with score >= 0.5.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )

            content = response.choices[0].message.content.strip()
            if content.startswith('[') and content.endswith(']'):
                evaluations = json.loads(content)
                return [e for e in evaluations if isinstance(e, dict) and e.get('score', 0) >= self.relevance_threshold]
            else:
                return self.fallback_document_relevance(vicinity_candidates, doc_content)

        except Exception as e:
            print(f"Document relevance evaluation failed: {e}")
            return self.fallback_document_relevance(vicinity_candidates, doc_content)

    def fallback_document_relevance(self, vicinity_candidates: List[str], doc_content: str) -> List[Dict[str, Any]]:
        """
        Document-driven fallback method for relevance evaluation without LLM
        Uses document vocabulary and semantic presence
        """
        results = []
        doc_lower = doc_content.lower()
        doc_words = set(doc_lower.split())

        for candidate in vicinity_candidates:
            candidate_lower = candidate.lower()
            candidate_words = candidate_lower.split()

            # Calculate document-driven relevance score
            score = 0.0

            # Exact phrase presence in document
            if candidate_lower in doc_lower:
                score += 0.7

            # Word-level semantic presence
            word_matches = sum(1 for word in candidate_words if word in doc_words)
            if word_matches > 0:
                coverage_ratio = word_matches / len(candidate_words)
                score += coverage_ratio * 0.5

            # Semantic word variants (stems and related forms)
            semantic_matches = 0
            for word in candidate_words:
                if len(word) > 3:
                    # Check for word stems or related forms in document
                    word_stem = word[:4]  # Simple stemming
                    if any(doc_word.startswith(word_stem) for doc_word in doc_words):
                        semantic_matches += 1

            if semantic_matches > 0:
                semantic_coverage = semantic_matches / len(candidate_words)
                score += semantic_coverage * 0.3

            # Document density bonus (if the document contains many related terms)
            if len(doc_words) > 0:
                related_terms_count = sum(1 for word in candidate_words if word in doc_words)
                if related_terms_count > 1:
                    score += 0.2

            if score >= self.relevance_threshold:
                results.append({
                    'concept': candidate,
                    'score': min(score, 1.0),
                    'reasoning': f'Document semantic matching score: {score:.2f}'
                })

        return results

    async def discover_vicinity_concepts(self,
                                       core_concept: Dict[str, Any],
                                       document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method: discover vicinity concepts for a core concept in a document

        Returns:
            Dictionary with vicinity concept results
        """
        concept_id = core_concept.get('concept_id', '')
        concept_name = core_concept.get('canonical_name', '')
        core_keywords = core_concept.get('keywords', [])

        print(f"  Discovering vicinity for: {concept_name}")

        # Step 1: Generate semantic candidates
        candidates = await self.generate_semantic_candidates(concept_name, core_keywords)

        if not candidates:
            return {
                'concept_id': concept_id,
                'core_concept_name': concept_name,
                'vicinity_concepts': [],
                'status': 'no_candidates_generated'
            }

        # Step 2: Evaluate document relevance
        relevant_concepts = await self.evaluate_document_relevance(candidates, core_concept, document)

        # Step 3: Select top vicinity concepts
        top_concepts = sorted(relevant_concepts, key=lambda x: x['score'], reverse=True)
        top_concepts = top_concepts[:self.max_vicinity_per_concept]

        vicinity_terms = [c['concept'] for c in top_concepts]
        self.concepts_discovered += len(vicinity_terms)

        return {
            'concept_id': concept_id,
            'core_concept_name': concept_name,
            'core_keywords': core_keywords,
            'vicinity_concepts': vicinity_terms,
            'vicinity_details': top_concepts,
            'candidates_generated': len(candidates),
            'concepts_selected': len(vicinity_terms)
        }

    async def process_single_document(self,
                                    document: Dict[str, Any],
                                    a24_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single document to discover vicinity concepts for all its core concepts
        """
        doc_id = document.get('doc_id', 'unknown')

        # Find core concepts for this document
        doc_core_concepts = []
        for a24_doc in a24_data.get('documents', []):
            if a24_doc.get('doc_id') == doc_id:
                doc_core_concepts = a24_doc.get('core_concepts', [])
                break

        if not doc_core_concepts:
            return {
                'doc_id': doc_id,
                'vicinity_results': {},
                'status': 'no_core_concepts_found'
            }

        print(f"\nProcessing {doc_id}: {len(doc_core_concepts)} core concepts")

        # Discover vicinity concepts for each core concept
        vicinity_results = {}

        for core_concept in doc_core_concepts:
            try:
                vicinity_result = await self.discover_vicinity_concepts(core_concept, document)
                concept_id = vicinity_result['concept_id']
                vicinity_results[concept_id] = vicinity_result

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error processing concept {core_concept.get('concept_id', 'unknown')}: {e}")
                continue

        return {
            'doc_id': doc_id,
            'vicinity_results': vicinity_results,
            'core_concepts_processed': len(doc_core_concepts),
            'vicinity_concepts_found': sum(len(r.get('vicinity_concepts', [])) for r in vicinity_results.values()),
            'status': 'completed'
        }

    async def process_all_documents(self, a24_data: Dict[str, Any], doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all documents to discover vicinity concepts
        """
        print("Starting A2.5 Semantic Vicinity Concept Discovery...")
        start_time = time.time()

        all_results = []
        total_vicinity_concepts = 0
        successful_documents = 0

        for document in doc_data.get('documents', []):
            try:
                result = await self.process_single_document(document, a24_data)
                all_results.append(result)

                if result['status'] == 'completed':
                    successful_documents += 1
                    total_vicinity_concepts += result['vicinity_concepts_found']
                    print(f"  {result['doc_id']}: {result['vicinity_concepts_found']} vicinity concepts")

            except Exception as e:
                print(f"Failed to process document: {e}")
                continue

        processing_time = time.time() - start_time

        # Generate summary statistics
        summary = {
            'total_documents': len(all_results),
            'successful_documents': successful_documents,
            'total_vicinity_concepts': total_vicinity_concepts,
            'avg_vicinity_per_document': total_vicinity_concepts / max(successful_documents, 1),
            'processing_time_seconds': processing_time,
            'concepts_discovered_total': self.concepts_discovered
        }

        return {
            'strategy_name': 'semantic_vicinity_discovery',
            'results': {
                'document_vicinity_discoveries': all_results,
                'summary': summary
            },
            'processing_timestamp': datetime.now().isoformat()
        }

def create_a3_compatible_format(vicinity_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert vicinity discovery results to A3-compatible expanded concepts format
    """
    expanded_concepts = {}

    document_results = vicinity_results['results']['document_vicinity_discoveries']

    for doc_result in document_results:
        vicinity_results_dict = doc_result.get('vicinity_results', {})

        for concept_id, vicinity_data in vicinity_results_dict.items():
            if vicinity_data.get('vicinity_concepts'):
                # Create A3-compatible structure
                expanded_concepts[concept_id] = {
                    'concept_id': concept_id,
                    'strategy_contributions': {
                        'semantic_vicinity_discovery': {
                            'terms': vicinity_data['vicinity_concepts'],
                            'count': len(vicinity_data['vicinity_concepts']),
                            'weight': 1.0,
                            'details': vicinity_data.get('vicinity_details', [])
                        }
                    },
                    'total_terms': len(vicinity_data['vicinity_concepts']),
                    'processing_metadata': {
                        'core_concept_name': vicinity_data.get('core_concept_name', ''),
                        'core_keywords': vicinity_data.get('core_keywords', []),
                        'candidates_generated': vicinity_data.get('candidates_generated', 0),
                        'concepts_selected': vicinity_data.get('concepts_selected', 0)
                    }
                }

    # Create orchestration metadata compatible with A3
    orchestration_metadata = {
        'strategy_weights': {
            'semantic_vicinity_discovery': 1.0
        },
        'processing_timestamp': vicinity_results.get('processing_timestamp', ''),
        'total_concepts': len(expanded_concepts),
        'strategy_name': 'semantic_vicinity_discovery',
        'discovery_summary': vicinity_results['results']['summary']
    }

    return {
        'orchestration_metadata': orchestration_metadata,
        'expanded_concepts': expanded_concepts
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

async def process_vicinity_discovery():
    """Main processing function"""
    print("=" * 60)
    print("A2.5: Semantic Vicinity Concept Discovery")
    print("Single script replacement for all A2.5 strategies")
    print("=" * 60)

    # Load input data
    a24_data, doc_data = load_input_data()
    print(f"Loaded {len(doc_data['documents'])} documents")
    print(f"Found {sum(len(d.get('core_concepts', [])) for d in a24_data['documents'])} core concepts")

    # Initialize discovery engine
    discovery = A25_VicinityConceptDiscovery()

    # Process all documents
    results = await discovery.process_all_documents(a24_data, doc_data)

    # Save results in both formats
    # 1. Detailed results for analysis
    detailed_output_path = Path(__file__).parent.parent / "outputs/A2.5_vicinity_concepts_detailed.json"
    with open(detailed_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 2. A3-compatible format (replaces old A2.5 output)
    a3_compatible_output = create_a3_compatible_format(results)
    output_path = Path(__file__).parent.parent / "outputs/A2.5_expanded_concepts.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(a3_compatible_output, f, indent=2, ensure_ascii=False)

    # Display summary
    summary = results['results']['summary']
    print(f"\n{'=' * 60}")
    print("A2.5 SEMANTIC VICINITY DISCOVERY SUMMARY:")
    print(f"Documents processed: {summary['successful_documents']}/{summary['total_documents']}")
    print(f"Total vicinity concepts: {summary['total_vicinity_concepts']}")
    print(f"Average per document: {summary['avg_vicinity_per_document']:.1f}")
    print(f"Processing time: {summary['processing_time_seconds']:.1f} seconds")
    print(f"Total concepts discovered: {summary['concepts_discovered_total']}")
    print(f"\nSaved to: {output_path}")

    # Show sample results
    print(f"\nSAMPLE VICINITY DISCOVERIES:")
    document_results = results['results']['document_vicinity_discoveries']
    for doc_result in document_results[:3]:
        if doc_result.get('vicinity_results'):
            print(f"\n{doc_result['doc_id']}:")
            for concept_id, vicinity in list(doc_result['vicinity_results'].items())[:2]:
                core_name = vicinity['core_concept_name']
                vicinity_list = vicinity['vicinity_concepts'][:3]
                print(f"  '{core_name}' -> {vicinity_list}")

def main():
    """Main execution"""
    try:
        asyncio.run(process_vicinity_discovery())
        print("\nA2.5 Semantic Vicinity Discovery completed successfully!")
    except Exception as e:
        print(f"Error in A2.5: {str(e)}")
        raise

if __name__ == "__main__":
    main()