#!/usr/bin/env python3
"""
A3: Concept-Based Chunking Orchestrator
Multi-strategy document chunking system with hybrid architecture

This orchestrator coordinates multiple chunking strategies to create
multi-layered chunks with overlapping concept memberships. Each strategy
is implemented as a separate module for modularity and maintainability.

Architecture:
- Orchestrator (this file): Coordinates strategies and aggregates results
- Strategy modules: Individual chunking implementations in chunking_strategies/
- Base strategy: Common interface and utilities for all strategies

Strategies available:
1. Semantic Sentence: Sentence-level concept alignment
2. Paragraph Aware: Natural paragraph boundaries
3. Document Structure: Respects headings and sections
4. Adaptive: Dynamic sizing based on content density
5. Concept Aware: Guided by concept centroids
6. Contextual Overlap: Maintains context across boundaries
7. Quality Based: Optimized for retrieval quality
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Set
import time

# Import all chunking strategies
from chunking_strategies import (
    get_strategy,
    list_strategies,
    ConceptChunk,
    STRATEGIES
)


class A3ConceptChunkingOrchestrator:
    """
    Orchestrates multiple chunking strategies to create comprehensive
    multi-layered chunks for the conceptual space system
    """
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.parent
        self.input_dir = self.script_dir / "outputs"
        self.output_dir = self.script_dir / "outputs"
        
        # Strategy configuration
        self.enabled_strategies = list_strategies()  # All strategies by default
        self.strategy_weights = self._get_default_weights()
        
        # Deduplication and merging configuration
        self.dedup_threshold = 0.85  # Similarity threshold for deduplication
        self.merge_overlapping = True
        
        # Statistics tracking
        self.stats = defaultdict(dict)
        
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default weights for each strategy"""
        return {
            'semantic_sentence': 1.0,
            'paragraph_aware': 1.0,
            'document_structure': 1.2,  # Slightly favor structure
            'adaptive': 1.0,
            'concept_aware': 1.3,  # Favor concept-driven chunking
            'contextual_overlap': 0.9,
            'quality_based': 1.4  # Highest weight for quality-optimized
        }
    
    def load_concepts(self) -> Dict[str, Any]:
        """Load core and expanded concepts"""
        concepts = {}
        
        # Load core concepts (A2.4)
        core_path = self.input_dir / "A2.4_core_concepts.json"
        if core_path.exists():
            with open(core_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract the core_concepts list and adapt to expected format
                core_concepts = data.get('core_concepts', [])
                # Convert primary_keywords to terms for compatibility
                for concept in core_concepts:
                    if 'primary_keywords' in concept and 'terms' not in concept:
                        concept['terms'] = concept['primary_keywords']
                concepts['core'] = {'concepts': core_concepts}
                print(f"Loaded {len(core_concepts)} core concepts")
        
        # Load expanded concepts (A2.5)
        expanded_path = self.input_dir / "A2.5_expanded_concepts.json"
        if expanded_path.exists():
            with open(expanded_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract and convert expanded_concepts dict to list
                expanded_dict = data.get('expanded_concepts', {})
                expanded_list = []
                for concept_id, concept_data in expanded_dict.items():
                    # Create unified concept structure
                    expanded_concept = {
                        'concept_id': concept_id,
                        'expanded_terms': []
                    }
                    # Collect terms from all strategies
                    if 'strategy_contributions' in concept_data:
                        for strategy, contrib in concept_data['strategy_contributions'].items():
                            if 'terms' in contrib:
                                expanded_concept['expanded_terms'].extend(contrib['terms'])
                    # Remove duplicates
                    expanded_concept['expanded_terms'] = list(set(expanded_concept['expanded_terms']))
                    expanded_list.append(expanded_concept)
                concepts['expanded'] = {'expanded_concepts': expanded_list}
                print(f"Loaded {len(expanded_list)} expanded concepts")
        
        return concepts
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from A1.1 (fresh) or A2.1 (preprocessed) with fallback"""
        # Priority: Fresh A1.1 raw documents first, then A2.1 preprocessed documents
        a11_path = self.input_dir / "A1.1_raw_documents.json"
        a21_path = self.input_dir / "A2.1_preprocessed_documents.json"
        
        doc_path = None
        use_a11 = False
        
        # Check A1.1 first (fresh data from batch processing)
        if a11_path.exists():
            # Check if A1.1 is newer than A2.1 to detect fresh data
            a11_mtime = a11_path.stat().st_mtime if a11_path.exists() else 0
            a21_mtime = a21_path.stat().st_mtime if a21_path.exists() else 0
            
            if a11_mtime > a21_mtime or not a21_path.exists():
                doc_path = a11_path
                use_a11 = True
                print(f"Loading FRESH documents from: {doc_path.name}")
            else:
                doc_path = a21_path
                print(f"Loading preprocessed documents from: {doc_path.name}")
        elif a21_path.exists():
            doc_path = a21_path
            print(f"Loading preprocessed documents from: {doc_path.name}")
        else:
            print("ERROR: No document files found!")
            print("Please run A1.1 or A2.1 first.")
            return []
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            documents = doc_data.get('documents', [])
            
            # Normalize document structure - strategies expect 'content' field
            for doc in documents:
                if use_a11:
                    # A1.1 provides 'text' field
                    if 'text' in doc and 'content' not in doc:
                        doc['content'] = doc['text']
                else:
                    # A2.1 provides multiple text fields, prioritize in this order:
                    # 1. cleaned_text (fully processed)
                    # 2. table_converted_text (if tables were converted) 
                    # 3. text (basic text)
                    if 'cleaned_text' in doc:
                        doc['content'] = doc['cleaned_text']
                    elif 'table_converted_text' in doc:
                        doc['content'] = doc['table_converted_text']
                    elif 'text' in doc and 'content' not in doc:
                        doc['content'] = doc['text']
                    
            print(f"  - Loaded {len(documents)} documents with tables converted to text")
            
            return documents
    
    def run_strategy(self, 
                    strategy_name: str,
                    documents: List[Dict[str, Any]],
                    concepts: Dict[str, Any],
                    config: Optional[Dict[str, Any]] = None) -> List[ConceptChunk]:
        """
        Run a single chunking strategy on all documents
        
        Args:
            strategy_name: Name of the strategy to run
            documents: List of documents to chunk
            concepts: Concept dictionary
            config: Optional strategy-specific configuration
            
        Returns:
            List of chunks created by the strategy
        """
        print(f"\n  Running {strategy_name} strategy...")
        start_time = time.time()
        
        # Get strategy instance
        strategy = get_strategy(strategy_name, **(config or {}))
        
        # Process all documents
        all_chunks = []
        for document in documents:
            chunks = strategy.chunk_document(document, concepts)
            all_chunks.extend(chunks)
        
        elapsed = time.time() - start_time
        
        # Store statistics
        self.stats[strategy_name] = {
            'chunks_created': len(all_chunks),
            'processing_time': elapsed,
            'config': strategy.get_strategy_config()
        }
        
        print(f"    Created {len(all_chunks)} chunks in {elapsed:.2f}s")
        
        return all_chunks
    
    def deduplicate_chunks(self, chunks: List[ConceptChunk]) -> List[ConceptChunk]:
        """
        Remove duplicate or highly similar chunks
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            Deduplicated list of chunks
        """
        if not chunks:
            return []
        
        unique_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            # Create content signature
            content_sig = chunk.content[:100].lower().replace(' ', '')
            
            # Check for exact duplicates
            if content_sig in seen_content:
                continue
            
            # Check for high similarity with existing chunks
            is_duplicate = False
            for unique_chunk in unique_chunks:
                similarity = self._calculate_similarity(chunk.content, unique_chunk.content)
                if similarity > self.dedup_threshold:
                    # Merge concept memberships if highly similar
                    unique_chunk.concept_memberships = list(set(
                        unique_chunk.concept_memberships + chunk.concept_memberships
                    ))
                    unique_chunk.membership_scores.update(chunk.membership_scores)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_content.add(content_sig)
        
        return unique_chunks
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text chunks"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def aggregate_chunks(self, 
                        strategy_chunks: Dict[str, List[ConceptChunk]]) -> List[Dict[str, Any]]:
        """
        Aggregate chunks from multiple strategies
        
        Args:
            strategy_chunks: Dictionary mapping strategy names to chunk lists
            
        Returns:
            Aggregated and weighted chunk list
        """
        # Collect all chunks with strategy weights
        weighted_chunks = []
        
        for strategy_name, chunks in strategy_chunks.items():
            weight = self.strategy_weights.get(strategy_name, 1.0)
            
            for chunk in chunks:
                chunk_dict = chunk.to_dict()
                chunk_dict['strategy_weight'] = weight
                chunk_dict['source_strategies'] = [strategy_name]
                weighted_chunks.append(chunk_dict)
        
        # Save raw chunks before deduplication
        raw_chunks_path = self.output_dir / "A3_raw_chunks_no_dedup.json"
        raw_output = {
            'chunks': weighted_chunks,
            'total_raw_chunks': len(weighted_chunks),
            'chunks_by_strategy': {name: len(chunks) for name, chunks in strategy_chunks.items()},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'deduplication_applied': False
            }
        }
        with open(raw_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(raw_output, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(weighted_chunks)} raw chunks to {raw_chunks_path.name}")
        
        # Deduplicate if enabled
        if self.merge_overlapping:
            print("\n  Deduplicating chunks across strategies...")
            # Convert back to ConceptChunk for deduplication
            chunk_objects = [
                ConceptChunk(
                    chunk_id=c['chunk_id'],
                    doc_id=c['doc_id'],
                    content=c['content'],
                    chunk_type=c['chunk_type'],
                    start_index=c['start_index'],
                    end_index=c['end_index'],
                    concept_memberships=c['concept_memberships'],
                    membership_scores=c['membership_scores'],
                    metadata=c['metadata']
                ) for c in weighted_chunks
            ]
            
            unique_chunks = self.deduplicate_chunks(chunk_objects)
            weighted_chunks = [c.to_dict() for c in unique_chunks]
            print(f"    Reduced from {len(chunk_objects)} to {len(weighted_chunks)} chunks")
        
        return weighted_chunks
    
    def calculate_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the chunking results"""
        if not chunks:
            return {
                'total_chunks': 0,
                'error': 'No chunks to analyze',
                'chunks_per_document': {},
                'multi_concept_chunks': 0,
                'multi_concept_ratio': 0,
                'average_concepts_per_chunk': 0,
                'average_chunk_size': 0,
                'chunk_size_std': 0,
                'strategy_contribution': {},
                'processing_stats': self.stats
            }
        
        # Basic statistics
        total_chunks = len(chunks)
        chunks_per_doc = defaultdict(int)
        concept_memberships = []
        chunk_sizes = []
        
        for chunk in chunks:
            chunks_per_doc[chunk['doc_id']] += 1
            concept_memberships.append(len(chunk['concept_memberships']))
            chunk_sizes.append(len(chunk['content']))
        
        # Multi-concept analysis
        multi_concept_chunks = sum(1 for c in chunks if len(c['concept_memberships']) > 1)
        
        # Strategy contribution
        strategy_contribution = Counter()
        for chunk in chunks:
            strategies = chunk.get('source_strategies', [chunk['chunk_type']])
            for strategy in strategies:
                strategy_contribution[strategy] += 1
        
        return {
            'total_chunks': total_chunks,
            'chunks_per_document': dict(chunks_per_doc),
            'multi_concept_chunks': multi_concept_chunks,
            'multi_concept_ratio': multi_concept_chunks / total_chunks if total_chunks > 0 else 0,
            'average_concepts_per_chunk': np.mean(concept_memberships) if concept_memberships else 0,
            'average_chunk_size': np.mean(chunk_sizes) if chunk_sizes else 0,
            'chunk_size_std': np.std(chunk_sizes) if chunk_sizes else 0,
            'strategy_contribution': dict(strategy_contribution),
            'processing_stats': self.stats
        }
    
    def save_results(self, chunks: List[Dict[str, Any]], statistics: Dict[str, Any]):
        """Save chunking results and statistics"""
        # Save chunks
        output_path = self.output_dir / "A3_multi_strategy_chunks.json"
        output_data = {
            'chunks': chunks,
            'statistics': statistics,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'enabled_strategies': self.enabled_strategies,
                'strategy_weights': self.strategy_weights,
                'total_strategies': len(self.enabled_strategies)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved multi-strategy chunks to {output_path}")
        
        # Save statistics separately for easy access
        stats_path = self.output_dir / "A3_chunking_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        print(f"Saved statistics to {stats_path}")
    
    def orchestrate(self, 
                   strategies: Optional[List[str]] = None,
                   strategy_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Main orchestration method that coordinates all chunking strategies
        
        Args:
            strategies: List of strategy names to run (None = all)
            strategy_configs: Optional configuration for each strategy
        """
        print("=" * 80)
        print("A3: MULTI-STRATEGY CONCEPT-BASED CHUNKING ORCHESTRATOR")
        print("Hybrid Architecture with Modular Strategies")
        print("=" * 80)
        
        # Load concepts and documents
        concepts = self.load_concepts()
        documents = self.load_documents()
        
        if not documents:
            print("No documents to process. Exiting.")
            return
        
        print(f"\nProcessing {len(documents)} documents...")
        
        # Determine which strategies to run
        strategies_to_run = strategies or self.enabled_strategies
        print(f"\nEnabled strategies: {', '.join(strategies_to_run)}")
        
        # Run each strategy
        strategy_chunks = {}
        for strategy_name in strategies_to_run:
            if strategy_name not in list_strategies():
                print(f"  Warning: Unknown strategy '{strategy_name}', skipping...")
                continue
            
            config = (strategy_configs or {}).get(strategy_name, {})
            chunks = self.run_strategy(strategy_name, documents, concepts, config)
            strategy_chunks[strategy_name] = chunks
        
        # Aggregate results
        print("\nAggregating chunks from all strategies...")
        aggregated_chunks = self.aggregate_chunks(strategy_chunks)
        
        # Calculate statistics
        print("\nCalculating statistics...")
        statistics = self.calculate_statistics(aggregated_chunks)
        
        # Display summary
        print("\n" + "=" * 80)
        print("CHUNKING SUMMARY")
        print("=" * 80)
        print(f"Total chunks created: {statistics['total_chunks']}")
        print(f"Multi-concept chunks: {statistics['multi_concept_chunks']} "
              f"({statistics['multi_concept_ratio']:.1%})")
        print(f"Average concepts per chunk: {statistics['average_concepts_per_chunk']:.2f}")
        print(f"Average chunk size: {statistics['average_chunk_size']:.0f} characters")
        
        print("\nStrategy contribution:")
        for strategy, count in statistics['strategy_contribution'].items():
            print(f"  - {strategy}: {count} chunks")
        
        # Save results
        self.save_results(aggregated_chunks, statistics)
        
        print("\n" + "=" * 80)
        print("Multi-strategy chunking complete!")
        print("=" * 80)


def main():
    """Main execution function"""
    orchestrator = A3ConceptChunkingOrchestrator()
    
    # Example: Run with specific strategies and custom configuration
    # strategies = ['semantic_sentence', 'concept_aware', 'quality_based']
    # configs = {
    #     'semantic_sentence': {'alignment_threshold': 0.4},
    #     'quality_based': {'min_quality_score': 0.7}
    # }
    # orchestrator.orchestrate(strategies=strategies, strategy_configs=configs)
    
    # Run with all strategies and default configuration
    orchestrator.orchestrate()


if __name__ == "__main__":
    main()