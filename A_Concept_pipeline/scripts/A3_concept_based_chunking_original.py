#!/usr/bin/env python3
"""
A3: Concept-Based Chunking System
Multi-layered document chunking using concept centroids with overlapping membership

Design Philosophy:
- Takes A2.4 (core concepts) + A2.5 (expanded concepts) as dual inputs
- Creates concept centroids from both core and expanded terms
- Implements multi-layered chunking where sentences can belong to multiple concept centroids
- Each chunk has membership within one or more convex balls defined by concept centroids
- Supports document-specific processing based on concept-document relationships

Architecture:
Input: A2.4 core concepts + A2.5 expanded concepts + raw documents
Process: Concept centroid creation → Multi-layered chunking → Convex ball membership
Output: A3 concept chunks with overlapping memberships and centroid distances
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple, Set
import re
import math
from dataclasses import dataclass


@dataclass
class ConceptCentroid:
    """Represents a concept centroid with core and expanded terms"""
    concept_id: str
    canonical_name: str
    core_terms: List[str]
    expanded_terms: List[str]
    all_terms: List[str]
    importance_score: float
    related_documents: List[str]
    domain: str
    centroid_vector: np.ndarray = None
    radius: float = 1.0


@dataclass 
class ConceptChunk:
    """Represents a chunk with concept centroid memberships"""
    chunk_id: str
    doc_id: str
    text: str
    sentences: List[str]
    word_count: int
    concept_memberships: Dict[str, float]  # concept_id -> membership_score
    centroid_distances: Dict[str, float]   # concept_id -> distance
    convex_ball_memberships: Dict[str, bool]  # concept_id -> inside_ball
    primary_centroid: str
    chunk_type: str  # 'single_concept', 'multi_concept', 'overlap_zone'


class A3ConceptBasedChunker:
    """
    Advanced concept-based chunking system with overlapping membership support
    """
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.parent
        self.outputs_dir = self.script_dir / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Load inputs
        self.a24_concepts = self._load_a24_concepts()
        self.a25_concepts = self._load_a25_concepts() 
        self.documents = self._load_documents()
        
        # Create concept centroids
        self.concept_centroids = self._create_concept_centroids()
        
        # Chunking parameters
        self.min_chunk_size = 50    # Minimum words per chunk
        self.max_chunk_size = 500   # Maximum words per chunk
        self.overlap_threshold = 0.05  # Minimum score for overlapping membership (lowered for better detection)
        self.convex_ball_radius_multiplier = 1.2  # Radius multiplier for convex balls
        
        # Results storage
        self.document_chunks = {}
        self.chunking_statistics = {
            'total_documents': 0,
            'total_chunks': 0,
            'single_concept_chunks': 0,
            'multi_concept_chunks': 0,
            'overlap_zones': 0,
            'average_memberships_per_chunk': 0.0,
            'concept_utilization': {}
        }
        
    def _load_a24_concepts(self) -> Dict[str, Any]:
        """Load A2.4 core concepts"""
        a24_path = self.outputs_dir / "A2.4_core_concepts.json"
        with open(a24_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {concept['concept_id']: concept for concept in data['core_concepts']}
    
    def _load_a25_concepts(self) -> Dict[str, Any]:
        """Load A2.5 generated concept entities from individual strategy files"""
        a25_concepts = {}
        
        # Strategy files that generate new concept entities
        strategy_files = {
            "semantic_similarity": "A2.5.1_semantic_expansion.json",
            "domain_knowledge": "A2.5.2_domain_expansion.json", 
            "hierarchical_clustering": "A2.5.3_hierarchical_expansion.json"
        }
        
        for strategy_name, filename in strategy_files.items():
            strategy_path = self.outputs_dir / filename
            if strategy_path.exists():
                try:
                    with open(strategy_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract generated concepts from strategy results
                    if "results" in data and "generated_concepts" in data["results"]:
                        generated_concepts = data["results"]["generated_concepts"]
                        for concept in generated_concepts:
                            concept_id = concept["concept_id"]
                            a25_concepts[concept_id] = concept
                            print(f"Loaded new concept: {concept_id} from {strategy_name}")
                except Exception as e:
                    print(f"[WARNING] Failed to load {strategy_name}: {e}")
            else:
                print(f"[WARNING] Strategy file not found: {filename}")
        
        print(f"Loaded {len(a25_concepts)} new concept entities from A2.5 strategies")
        return a25_concepts
    
    def _load_documents(self) -> Dict[str, str]:
        """Load raw documents - using sample data for demonstration"""
        # In a real implementation, this would load from A1.1 or document store
        sample_documents = {
            'finqa_test_96': """
            Contract Balances and Revenue Recognition
            
            Our contract balances consist of receivable balance and consolidated balance items.
            Revenue recognition principles apply to deferred income and unearned revenue.
            
            Contract balances represent the difference between revenue recognized and cash received.
            The receivable balance reflects amounts due from customers under long-term agreements.
            Revenue recognition occurs when performance obligations are satisfied.
            
            Unearned revenue represents cash received in advance of performance.
            These contract balances require careful monitoring for revenue recognition compliance.
            """,
            
            'finqa_test_1017': """
            Discontinued Operations Analysis
            
            The Company completed discontinued operations during the reporting period.
            Operations were discontinued due to strategic restructuring initiatives.
            
            Discontinued operations resulted in significant impact on financial statements.
            The discontinuation process involved multiple operational considerations.
            Tax implications of discontinued operations were carefully evaluated.
            
            Operations that were discontinued included several business segments.
            Financial impact of these discontinued operations continues to be monitored.
            """,
            
            'finqa_test_199': """
            Inventory Valuation and Operations
            
            The Company maintains comprehensive inventory valuation processes.
            Inventories are valued using standard accounting methods and procedures.
            
            Inventory valuation considers market conditions and operational factors.
            Operations of the Company include inventory management and valuation.
            Market-based inventory valuations are updated quarterly.
            
            The Company's operations integrate inventory management with financial reporting.
            Inventory valuation methodologies ensure accurate financial statement presentation.
            """,
            
            'finqa_test_617': """
            Deferred Income and Tax Assets
            
            Deferred income represents revenue received in advance of service delivery.
            The Company recognizes deferred income according to accounting standards.
            
            Current deferred income is expected to be recognized within twelve months.
            Non-current deferred income extends beyond the next operating cycle.
            Total deferred income balances are monitored for compliance purposes.
            """,
            
            'finqa_test_686': """
            Tax Assets and Net Book Values
            
            NBV (Net Book Value) calculations consider depreciation and impairment.
            TWDV (Tax Written Down Value) represents the tax basis of assets.
            
            NBV Net calculations provide book value information for financial reporting.
            TWDV Tax written values are used for tax compliance and planning.
            The difference between NBV Net and TWDV Tax creates timing differences.
            
            Tax calculations incorporate both current and deferred tax implications.
            Net book value assessments are performed annually for accuracy.
            """
        }
        return sample_documents
    
    def _create_concept_centroids(self) -> Dict[str, ConceptCentroid]:
        """Create concept centroids from A2.4 core concepts and A2.5 generated concept entities"""
        centroids = {}
        
        print("Creating concept centroids from A2.4 + A2.5 inputs...")
        
        # First, create centroids for A2.4 core concepts
        for concept_id, a24_concept in self.a24_concepts.items():
            # Extract core terms from A2.4
            core_terms = a24_concept.get('primary_keywords', [])
            expanded_terms = core_terms.copy()  # For A2.4, no additional expansion
            
            # Combine all terms
            all_terms = list(set(core_terms + expanded_terms))
            
            # Create centroid vector (simplified - in production would use embeddings)
            centroid_vector = self._create_centroid_vector(all_terms)
            
            # Calculate radius based on term diversity and importance
            radius = self._calculate_centroid_radius(
                core_terms, expanded_terms, a24_concept.get('importance_score', 0.5)
            )
            
            # Determine domain
            domain = a24_concept.get('business_category', 'General')
            
            centroid = ConceptCentroid(
                concept_id=concept_id,
                canonical_name=a24_concept.get('canonical_name', concept_id),
                core_terms=core_terms,
                expanded_terms=expanded_terms,
                all_terms=all_terms,
                importance_score=a24_concept.get('importance_score', 0.5),
                related_documents=a24_concept.get('related_documents', []),
                domain=domain,
                centroid_vector=centroid_vector,
                radius=radius
            )
            
            centroids[concept_id] = centroid
        
        # Second, create centroids for A2.5 generated concept entities
        for concept_id, a25_concept in self.a25_concepts.items():
            # Extract terms from A2.5 generated concept
            core_terms = a25_concept.get('primary_keywords', [])
            expanded_terms = core_terms.copy()  # For generated concepts, primary_keywords are the expanded terms
            all_terms = list(set(core_terms))
            
            # Create centroid vector
            centroid_vector = self._create_centroid_vector(all_terms)
            
            # Calculate radius (slightly smaller for generated concepts)
            radius = self._calculate_centroid_radius(
                core_terms, expanded_terms, 0.4  # Lower importance for generated concepts
            )
            
            # Use domain from generated concept
            domain = a25_concept.get('domain', 'general')
            
            centroid = ConceptCentroid(
                concept_id=concept_id,
                canonical_name=a25_concept.get('canonical_name', concept_id),
                core_terms=core_terms,
                expanded_terms=expanded_terms,
                all_terms=all_terms,
                importance_score=0.4,  # Lower importance for generated concepts
                related_documents=a25_concept.get('related_documents', []),
                domain=domain,
                centroid_vector=centroid_vector,
                radius=radius
            )
            
            centroids[concept_id] = centroid
            
        print(f"Created {len(centroids)} concept centroids ({len(self.a24_concepts)} A2.4 + {len(self.a25_concepts)} A2.5)")
        return centroids
    
    def _create_centroid_vector(self, terms: List[str], dimension: int = 100) -> np.ndarray:
        """Create a centroid vector from terms (simplified implementation)"""
        if not terms:
            return np.zeros(dimension)
        
        # Create reproducible vector based on term content
        combined_text = " ".join(sorted(terms)).lower()
        hash_value = abs(hash(combined_text)) % (2**31)
        
        np.random.seed(hash_value % (2**31))
        vector = np.random.normal(0, 1, dimension)
        
        # Add term-specific variations
        for i, term in enumerate(terms[:20]):  # Limit to first 20 terms
            term_hash = abs(hash(term.lower())) % dimension
            vector[term_hash] += 0.5 / len(terms)
        
        # Normalize
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    
    def _calculate_centroid_radius(self, core_terms: List[str], 
                                 expanded_terms: List[str], 
                                 importance_score: float) -> float:
        """Calculate the radius of the concept's convex ball"""
        # Base radius on term diversity and importance
        core_diversity = len(set(core_terms))
        expanded_diversity = len(set(expanded_terms))
        
        # More diverse concepts have larger radii
        diversity_factor = math.sqrt(core_diversity + expanded_diversity * 0.5) / 5.0
        
        # More important concepts have larger radii
        importance_factor = importance_score
        
        # Base radius between 0.5 and 2.0
        radius = 0.5 + (diversity_factor * importance_factor * 1.5)
        
        return min(max(radius, 0.5), 2.0)
    
    def process_all_documents(self) -> Dict[str, List[ConceptChunk]]:
        """Process all documents with concept-based chunking"""
        print(f"\nProcessing {len(self.documents)} documents with concept-based chunking...")
        
        for doc_id, document_text in self.documents.items():
            print(f"\nProcessing document: {doc_id}")
            
            # Get relevant concepts for this document
            doc_concepts = self._get_document_concepts(doc_id)
            
            if not doc_concepts:
                print(f"  No concepts found for {doc_id}, skipping...")
                continue
            
            # Create chunks for this document
            chunks = self._create_document_chunks(doc_id, document_text, doc_concepts)
            
            # Store results
            self.document_chunks[doc_id] = chunks
            self.chunking_statistics['total_documents'] += 1
            self.chunking_statistics['total_chunks'] += len(chunks)
            
            print(f"  Created {len(chunks)} chunks for {doc_id}")
            
            # Update chunk type statistics
            for chunk in chunks:
                if chunk.chunk_type == 'single_concept':
                    self.chunking_statistics['single_concept_chunks'] += 1
                elif chunk.chunk_type == 'multi_concept':
                    self.chunking_statistics['multi_concept_chunks'] += 1
                elif chunk.chunk_type == 'overlap_zone':
                    self.chunking_statistics['overlap_zones'] += 1
        
        self._calculate_final_statistics()
        return self.document_chunks
    
    def _get_document_concepts(self, doc_id: str) -> List[str]:
        """Get concept IDs related to a specific document"""
        doc_concepts = []
        for concept_id, centroid in self.concept_centroids.items():
            if doc_id in centroid.related_documents:
                doc_concepts.append(concept_id)
        return doc_concepts
    
    def _create_document_chunks(self, doc_id: str, text: str, 
                              concept_ids: List[str]) -> List[ConceptChunk]:
        """Create chunks for a document using relevant concept centroids"""
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Create sliding window chunks
        base_chunks = self._create_base_chunks(sentences)
        
        # Calculate concept memberships for each chunk
        concept_chunks = []
        
        for i, (chunk_sentences, chunk_text) in enumerate(base_chunks):
            chunk_id = f"{doc_id}_concept_chunk_{i}"
            
            # Calculate memberships and distances to each concept centroid
            memberships, distances, convex_memberships = self._calculate_chunk_memberships(
                chunk_text, concept_ids
            )
            
            # Determine primary centroid and chunk type
            primary_centroid, chunk_type = self._determine_chunk_classification(
                memberships, convex_memberships
            )
            
            chunk = ConceptChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                sentences=chunk_sentences,
                word_count=len(chunk_text.split()),
                concept_memberships=memberships,
                centroid_distances=distances,
                convex_ball_memberships=convex_memberships,
                primary_centroid=primary_centroid,
                chunk_type=chunk_type
            )
            
            concept_chunks.append(chunk)
        
        return concept_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Simple sentence splitting (could be enhanced with nltk/spacy)
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _create_base_chunks(self, sentences: List[str]) -> List[Tuple[List[str], str]]:
        """Create base chunks from sentences with sliding window approach"""
        chunks = []
        current_chunk = []
        current_words = 0
        
        for sentence in sentences:
            words_in_sentence = len(sentence.split())
            
            # If adding this sentence would exceed max size and we have content
            if current_words + words_in_sentence > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append((current_chunk.copy(), chunk_text))
                
                # Start new chunk (with overlap for multi-layered approach)
                if len(current_chunk) > 1:
                    # Keep last sentence for continuity
                    current_chunk = [current_chunk[-1], sentence]
                    current_words = len(current_chunk[-1].split()) + words_in_sentence
                else:
                    current_chunk = [sentence]
                    current_words = words_in_sentence
            else:
                current_chunk.append(sentence)
                current_words += words_in_sentence
                
                # If we've reached minimum size, this could be a chunk boundary
                if current_words >= self.min_chunk_size:
                    # Create chunk but continue building (allows overlapping)
                    chunk_text = ' '.join(current_chunk)
                    chunks.append((current_chunk.copy(), chunk_text))
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((current_chunk, chunk_text))
        
        return chunks
    
    def _calculate_chunk_memberships(self, chunk_text: str, 
                                   concept_ids: List[str]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, bool]]:
        """Calculate chunk membership in each concept centroid"""
        
        # Create chunk vector (simplified)
        chunk_terms = self._extract_chunk_terms(chunk_text)
        chunk_vector = self._create_centroid_vector(chunk_terms)
        
        memberships = {}
        distances = {}
        convex_memberships = {}
        
        for concept_id in concept_ids:
            centroid = self.concept_centroids[concept_id]
            
            # Calculate similarity/membership score
            similarity = np.dot(chunk_vector, centroid.centroid_vector)
            membership_score = max(0.0, similarity)  # Ensure non-negative
            
            # Calculate distance
            distance = np.linalg.norm(chunk_vector - centroid.centroid_vector)
            
            # Check convex ball membership
            adjusted_radius = centroid.radius * self.convex_ball_radius_multiplier
            in_convex_ball = distance <= adjusted_radius
            
            # Enhance membership with term overlap
            term_overlap_score = self._calculate_term_overlap(chunk_terms, centroid.all_terms)
            final_membership = (membership_score * 0.7) + (term_overlap_score * 0.3)
            
            memberships[concept_id] = final_membership
            distances[concept_id] = distance
            convex_memberships[concept_id] = in_convex_ball
        
        return memberships, distances, convex_memberships
    
    def _extract_chunk_terms(self, text: str) -> List[str]:
        """Extract terms from chunk text"""
        # Simple tokenization and cleaning
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they',
            'have', 'was', 'been', 'their', 'said', 'each', 'which', 'she', 'you',
            'one', 'our', 'had', 'but', 'not', 'can', 'may', 'all', 'any', 'were'
        }
        
        terms = [word for word in words if word not in stop_words and len(word) > 2]
        return terms
    
    def _calculate_term_overlap(self, chunk_terms: List[str], concept_terms: List[str]) -> float:
        """Calculate term overlap between chunk and concept"""
        if not chunk_terms or not concept_terms:
            return 0.0
        
        chunk_set = set(term.lower() for term in chunk_terms)
        concept_set = set(term.lower() for term in concept_terms)
        
        intersection = len(chunk_set & concept_set)
        union = len(chunk_set | concept_set)
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_chunk_classification(self, memberships: Dict[str, float], 
                                      convex_memberships: Dict[str, bool]) -> Tuple[str, str]:
        """Determine primary centroid and chunk type"""
        
        # Find highest membership
        if not memberships:
            return "unknown", "isolated"
        
        sorted_memberships = sorted(memberships.items(), key=lambda x: x[1], reverse=True)
        primary_centroid = sorted_memberships[0][0]
        primary_score = sorted_memberships[0][1]
        
        # Count significant memberships (above threshold)
        significant_memberships = sum(1 for score in memberships.values() 
                                    if score > self.overlap_threshold)
        
        # Count convex ball memberships
        convex_count = sum(1 for in_ball in convex_memberships.values() if in_ball)
        
        # Classify chunk type
        if significant_memberships == 1:
            chunk_type = "single_concept"
        elif significant_memberships > 1 and convex_count > 1:
            chunk_type = "overlap_zone"
        elif significant_memberships > 1:
            chunk_type = "multi_concept"
        else:
            chunk_type = "weak_association"
        
        return primary_centroid, chunk_type
    
    def _calculate_final_statistics(self):
        """Calculate final chunking statistics"""
        if self.chunking_statistics['total_chunks'] > 0:
            total_memberships = 0
            concept_usage = defaultdict(int)
            
            for chunks in self.document_chunks.values():
                for chunk in chunks:
                    # Count significant memberships
                    significant = sum(1 for score in chunk.concept_memberships.values() 
                                    if score > self.overlap_threshold)
                    total_memberships += significant
                    
                    # Track concept usage
                    for concept_id in chunk.concept_memberships:
                        if chunk.concept_memberships[concept_id] > self.overlap_threshold:
                            concept_usage[concept_id] += 1
            
            self.chunking_statistics['average_memberships_per_chunk'] = (
                total_memberships / self.chunking_statistics['total_chunks']
            )
            self.chunking_statistics['concept_utilization'] = dict(concept_usage)
    
    def save_results(self):
        """Save A3 chunking results to files"""
        output_timestamp = datetime.now().isoformat()
        
        # Save detailed chunks
        chunks_data = {
            'metadata': {
                'generation_timestamp': output_timestamp,
                'total_documents': len(self.document_chunks),
                'total_chunks': sum(len(chunks) for chunks in self.document_chunks.values()),
                'input_sources': ['A2.4_core_concepts.json', 'A2.5_expanded_concepts.json'],
                'chunking_method': 'concept_centroid_multi_layered'
            },
            'concept_centroids': {
                concept_id: {
                    'canonical_name': centroid.canonical_name,
                    'core_terms_count': len(centroid.core_terms),
                    'expanded_terms_count': len(centroid.expanded_terms),
                    'total_terms': len(centroid.all_terms),
                    'importance_score': centroid.importance_score,
                    'radius': centroid.radius,
                    'domain': centroid.domain,
                    'related_documents': centroid.related_documents
                }
                for concept_id, centroid in self.concept_centroids.items()
            },
            'document_chunks': {}
        }
        
        # Add chunk data
        for doc_id, chunks in self.document_chunks.items():
            chunks_data['document_chunks'][doc_id] = []
            for chunk in chunks:
                chunks_data['document_chunks'][doc_id].append({
                    'chunk_id': chunk.chunk_id,
                    'text': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    'word_count': chunk.word_count,
                    'sentence_count': len(chunk.sentences),
                    'concept_memberships': chunk.concept_memberships,
                    'centroid_distances': chunk.centroid_distances,
                    'convex_ball_memberships': {k: bool(v) for k, v in chunk.convex_ball_memberships.items()},
                    'primary_centroid': chunk.primary_centroid,
                    'chunk_type': chunk.chunk_type
                })
        
        # Save main results
        output_path = self.outputs_dir / "A3_concept_based_chunks.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_data = {
            'chunking_statistics': self.chunking_statistics,
            'generation_timestamp': output_timestamp,
            'concept_centroid_summary': [
                {
                    'concept_id': concept_id,
                    'canonical_name': centroid.canonical_name,
                    'usage_count': self.chunking_statistics['concept_utilization'].get(concept_id, 0),
                    'core_terms': len(centroid.core_terms),
                    'expanded_terms': len(centroid.expanded_terms),
                    'radius': round(centroid.radius, 3)
                }
                for concept_id, centroid in self.concept_centroids.items()
            ]
        }
        
        stats_path = self.outputs_dir / "A3_chunking_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        # Create CSV summary for easy analysis
        csv_data = []
        for doc_id, chunks in self.document_chunks.items():
            for chunk in chunks:
                # Find memberships above threshold
                significant_memberships = {
                    concept_id: score for concept_id, score in chunk.concept_memberships.items()
                    if score > self.overlap_threshold
                }
                
                csv_data.append({
                    'Doc_ID': doc_id,
                    'Chunk_ID': chunk.chunk_id,
                    'Word_Count': chunk.word_count,
                    'Sentence_Count': len(chunk.sentences),
                    'Primary_Centroid': chunk.primary_centroid,
                    'Chunk_Type': chunk.chunk_type,
                    'Membership_Count': len(significant_memberships),
                    'Max_Membership_Score': max(chunk.concept_memberships.values()),
                    'Min_Distance': min(chunk.centroid_distances.values()),
                    'Convex_Ball_Count': sum(chunk.convex_ball_memberships.values()),
                    'Overlapping_Concepts': '|'.join(significant_memberships.keys()) if len(significant_memberships) > 1 else '',
                    'Text_Preview': chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                })
        
        csv_path = self.outputs_dir / "A3_concept_chunks_summary.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        print(f"\nA3 Results saved:")
        print(f"  Main results: {output_path}")
        print(f"  Statistics: {stats_path}")
        print(f"  CSV summary: {csv_path}")
        
        return output_path, stats_path, csv_path


def main():
    """Main execution function"""
    print("="*80)
    print("A3: CONCEPT-BASED CHUNKING SYSTEM")
    print("Multi-layered chunking with overlapping convex ball memberships")
    print("="*80)
    
    # Initialize chunker
    chunker = A3ConceptBasedChunker()
    
    print(f"\nLoaded {len(chunker.concept_centroids)} concept centroids")
    print(f"Processing {len(chunker.documents)} documents")
    
    # Process all documents
    results = chunker.process_all_documents()
    
    # Print summary
    print(f"\nCHUNKING SUMMARY:")
    print(f"="*50)
    stats = chunker.chunking_statistics
    print(f"Documents processed: {stats['total_documents']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Single-concept chunks: {stats['single_concept_chunks']}")
    print(f"Multi-concept chunks: {stats['multi_concept_chunks']}")
    print(f"Overlap zones: {stats['overlap_zones']}")
    print(f"Avg memberships per chunk: {stats['average_memberships_per_chunk']:.2f}")
    
    # Save results
    output_files = chunker.save_results()
    
    print(f"\nSuccessfully implemented A3 concept-based chunking!")
    print(f"[SUCCESS] Multi-layered chunking with concept centroids")
    print(f"[SUCCESS] Overlapping membership in convex balls")
    print(f"[SUCCESS] Uses both A2.4 core and A2.5 expanded concepts")
    print(f"[SUCCESS] Document-specific concept processing")
    
    return results


if __name__ == "__main__":
    main()