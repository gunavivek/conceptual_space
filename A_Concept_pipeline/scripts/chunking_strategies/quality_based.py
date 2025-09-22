#!/usr/bin/env python3
"""
A3.7: Quality-Based Chunking Strategy
Chunking optimized for retrieval quality using A37 metrics
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from .base_strategy import BaseChunkingStrategy, ConceptChunk

class QualityBasedStrategy(BaseChunkingStrategy):
    """
    Implements quality-driven chunking that optimizes for retrieval performance
    using affinity, fidelity, and semantic coherence metrics
    """
    
    def __init__(self, 
                 min_quality_score: float = 0.6,
                 affinity_weight: float = 0.4,
                 fidelity_weight: float = 0.3,
                 coherence_weight: float = 0.3):
        super().__init__("quality_based")
        self.min_quality_score = min_quality_score
        self.affinity_weight = affinity_weight
        self.fidelity_weight = fidelity_weight
        self.coherence_weight = coherence_weight
        
    def get_strategy_config(self) -> Dict[str, Any]:
        """Return configuration for this strategy"""
        return {
            'min_quality_score': self.min_quality_score,
            'affinity_weight': self.affinity_weight,
            'fidelity_weight': self.fidelity_weight,
            'coherence_weight': self.coherence_weight,
            'description': 'Quality-optimized chunking using A37 metrics'
        }
    
    def calculate_affinity_score(self, text: str, concepts: Dict[str, Any]) -> float:
        """
        Calculate concept-chunk affinity score
        Measures how well the chunk aligns with its assigned concepts
        
        Returns:
            Affinity score between 0 and 1
        """
        # Note: For quality calculation, we use generic matching
        memberships, scores = self.extract_concept_memberships(text, concepts, threshold=0.2)
        
        if not scores:
            return 0.0
        
        # Average alignment across all relevant concepts
        avg_alignment = np.mean(list(scores.values()))
        
        # Bonus for multiple strong concept alignments
        strong_alignments = sum(1 for s in scores.values() if s > 0.6)
        multi_concept_bonus = min(0.2, strong_alignments * 0.05)
        
        return min(1.0, avg_alignment + multi_concept_bonus)
    
    def calculate_fidelity_score(self, text: str, original_text: str, start: int, end: int) -> float:
        """
        Calculate information fidelity score
        Measures how well the chunk preserves original information
        
        Returns:
            Fidelity score between 0 and 1
        """
        # Check if chunk boundaries respect sentence boundaries
        sentence_boundary_score = 0.0
        if start == 0 or original_text[start-1] in '.!?\n':
            sentence_boundary_score += 0.5
        if end == len(original_text) or original_text[end-1] in '.!?':
            sentence_boundary_score += 0.5
        
        # Check for completeness (no truncated words)
        completeness_score = 1.0
        if start > 0 and original_text[start-1].isalnum():
            completeness_score -= 0.25
        if end < len(original_text) and original_text[end].isalnum():
            completeness_score -= 0.25
        
        # Information density (non-whitespace ratio)
        non_whitespace = len(text.replace(' ', '').replace('\n', ''))
        density_score = non_whitespace / len(text) if text else 0
        
        # Weighted combination
        fidelity = (sentence_boundary_score * 0.4 + 
                   completeness_score * 0.3 + 
                   density_score * 0.3)
        
        return fidelity
    
    def calculate_coherence_score(self, text: str) -> float:
        """
        Calculate semantic coherence score
        Measures internal consistency and flow of the chunk
        
        Returns:
            Coherence score between 0 and 1
        """
        sentences = text.split('.')
        
        if len(sentences) <= 1:
            return 1.0  # Single sentence is maximally coherent
        
        # Check for consistency in terminology
        all_words = text.lower().split()
        unique_words = set(all_words)
        
        # Repetition indicates thematic consistency
        repetition_score = 1.0 - (len(unique_words) / len(all_words)) if all_words else 0
        
        # Check for transitional phrases indicating flow
        transitions = ['however', 'therefore', 'moreover', 'furthermore', 
                      'additionally', 'consequently', 'thus', 'hence']
        transition_count = sum(1 for t in transitions if t in text.lower())
        transition_score = min(1.0, transition_count * 0.2)
        
        # Length consistency across sentences
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sent_lengths:
            length_variance = np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 1
            consistency_score = max(0, 1.0 - length_variance)
        else:
            consistency_score = 0.5
        
        # Weighted combination
        coherence = (repetition_score * 0.3 + 
                    transition_score * 0.3 + 
                    consistency_score * 0.4)
        
        return min(1.0, coherence)
    
    def calculate_quality_score(self, 
                               text: str, 
                               concepts: Dict[str, Any],
                               original_text: str,
                               start: int,
                               end: int) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall quality score for a chunk
        
        Returns:
            Tuple of (overall_score, component_scores)
        """
        affinity = self.calculate_affinity_score(text, concepts)
        fidelity = self.calculate_fidelity_score(text, original_text, start, end)
        coherence = self.calculate_coherence_score(text)
        
        overall = (affinity * self.affinity_weight + 
                  fidelity * self.fidelity_weight + 
                  coherence * self.coherence_weight)
        
        components = {
            'affinity': affinity,
            'fidelity': fidelity,
            'coherence': coherence,
            'overall': overall
        }
        
        return overall, components
    
    def optimize_chunk_boundaries(self, 
                                 text: str, 
                                 concepts: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Find optimal chunk boundaries based on quality metrics
        
        Returns:
            List of (start, end) tuples for optimal chunks
        """
        boundaries = []
        sentences = self.split_sentences(text)
        
        if not sentences:
            return [(0, len(text))]
        
        # Dynamic programming approach to find optimal chunking
        current_chunk_start = 0
        current_chunk_sentences = []
        
        for sent_text, sent_start, sent_end in sentences:
            # Try adding sentence to current chunk
            test_chunk_sentences = current_chunk_sentences + [sent_text]
            test_chunk_text = ' '.join(test_chunk_sentences)
            
            # Calculate quality if we include this sentence
            quality, _ = self.calculate_quality_score(
                test_chunk_text, 
                concepts,
                text,
                current_chunk_start,
                sent_end
            )
            
            # Decide whether to include or start new chunk
            if quality >= self.min_quality_score and len(test_chunk_text) < 500:
                # Include in current chunk
                current_chunk_sentences.append(sent_text)
            else:
                # Save current chunk and start new one
                if current_chunk_sentences:
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunk_end = current_chunk_start + len(chunk_text)
                    boundaries.append((current_chunk_start, chunk_end))
                
                # Start new chunk with current sentence
                current_chunk_start = sent_start
                current_chunk_sentences = [sent_text]
        
        # Save final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_end = current_chunk_start + len(chunk_text)
            boundaries.append((current_chunk_start, chunk_end))
        
        return boundaries
    
    def chunk_document(self, 
                      document: Dict[str, Any], 
                      concepts: Dict[str, Any],
                      **kwargs) -> List[ConceptChunk]:
        """
        Create quality-optimized chunks from document
        
        Args:
            document: Document with 'doc_id' and 'content'
            concepts: Core and expanded concepts
            **kwargs: Additional parameters
            
        Returns:
            List of ConceptChunk objects
        """
        chunks = []
        doc_id = document.get('doc_id', 'unknown')
        content = document.get('content', '')
        
        # Find optimal chunk boundaries
        boundaries = self.optimize_chunk_boundaries(content, concepts)
        
        chunk_index = 0
        for start_idx, end_idx in boundaries:
            chunk_text = content[start_idx:end_idx]
            
            # Calculate quality scores
            quality_score, quality_components = self.calculate_quality_score(
                chunk_text, concepts, content, start_idx, end_idx
            )
            
            # Skip low-quality chunks
            if quality_score < self.min_quality_score:
                continue
            
            # Extract concept memberships
            # DOCUMENT-AWARE: Only match concepts from same document

            memberships, scores = self.extract_concept_memberships(chunk_text, concepts, doc_id=doc_id)
            
            # Skip if no concept memberships
            if not memberships:
                continue
            
            metadata = {
                'quality_score': quality_score,
                'affinity_score': quality_components['affinity'],
                'fidelity_score': quality_components['fidelity'],
                'coherence_score': quality_components['coherence'],
                'chunk_size': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'quality_tier': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.6 else 'low',
                'retrieval_weight': quality_score * len(memberships)  # Combined metric for retrieval
            }
            
            chunk = self.create_chunk(
                doc_id=doc_id,
                content=chunk_text,
                chunk_index=chunk_index,
                start_index=start_idx,
                end_index=end_idx,
                concepts=concepts,
                metadata=metadata
            )
            
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks