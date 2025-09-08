#!/usr/bin/env python3
"""
A3.6: Contextual Overlap Chunking Strategy
Chunking with controlled overlap to maintain context across boundaries
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from .base_strategy import BaseChunkingStrategy, ConceptChunk

class ContextualOverlapStrategy(BaseChunkingStrategy):
    """
    Implements chunking with intentional overlap between adjacent chunks
    to preserve context and improve retrieval continuity
    """
    
    def __init__(self, 
                 base_chunk_size: int = 200,
                 overlap_size: int = 50,
                 overlap_ratio: float = 0.25):
        super().__init__("contextual_overlap")
        self.base_chunk_size = base_chunk_size
        self.overlap_size = overlap_size
        self.overlap_ratio = overlap_ratio  # Percentage of chunk to overlap
        
    def get_strategy_config(self) -> Dict[str, Any]:
        """Return configuration for this strategy"""
        return {
            'base_chunk_size': self.base_chunk_size,
            'overlap_size': self.overlap_size,
            'overlap_ratio': self.overlap_ratio,
            'description': 'Contextual chunking with controlled overlap for continuity'
        }
    
    def calculate_context_similarity(self, chunk1: str, chunk2: str) -> float:
        """
        Calculate contextual similarity between two chunks
        
        Returns:
            Similarity score between 0 and 1
        """
        # Use word overlap as a simple similarity metric
        words1 = set(chunk1.lower().split())
        words2 = set(chunk2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard
    
    def create_overlapping_segments(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Create text segments with controlled overlap
        
        Returns:
            List of (segment_text, start_idx, end_idx) tuples
        """
        segments = []
        text_length = len(text)
        
        # Calculate actual overlap based on ratio
        actual_overlap = min(self.overlap_size, int(self.base_chunk_size * self.overlap_ratio))
        
        # Calculate step size (how much to advance for each chunk)
        step_size = self.base_chunk_size - actual_overlap
        
        current_pos = 0
        while current_pos < text_length:
            # Calculate end position
            end_pos = min(current_pos + self.base_chunk_size, text_length)
            
            # Try to extend to sentence boundary
            if end_pos < text_length:
                # Look for sentence end within next 50 characters
                for i in range(end_pos, min(end_pos + 50, text_length)):
                    if text[i] in '.!?':
                        end_pos = i + 1
                        break
            
            # Extract segment
            segment_text = text[current_pos:end_pos]
            
            # Only add if segment is substantial
            if len(segment_text.strip()) > 20:
                segments.append((segment_text, current_pos, end_pos))
            
            # Move to next position with overlap
            current_pos += step_size
            
            # Break if we've processed the entire text
            if current_pos >= text_length - 20:
                break
        
        return segments
    
    def analyze_overlap_quality(self, 
                               prev_chunk: str, 
                               curr_chunk: str,
                               concepts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the quality of overlap between two chunks
        
        Returns:
            Dictionary with overlap quality metrics
        """
        # Find the overlapping region
        overlap_start = max(0, len(prev_chunk) - self.overlap_size)
        overlap_text_prev = prev_chunk[overlap_start:]
        overlap_text_curr = curr_chunk[:self.overlap_size]
        
        # Calculate concept preservation in overlap
        prev_concepts, prev_scores = self.extract_concept_memberships(overlap_text_prev, concepts)
        curr_concepts, curr_scores = self.extract_concept_memberships(overlap_text_curr, concepts)
        
        # Concept continuity
        shared_concepts = set(prev_concepts).intersection(set(curr_concepts))
        concept_continuity = len(shared_concepts) / max(len(prev_concepts), len(curr_concepts), 1)
        
        # Context similarity
        context_sim = self.calculate_context_similarity(overlap_text_prev, overlap_text_curr)
        
        return {
            'concept_continuity': concept_continuity,
            'context_similarity': context_sim,
            'shared_concepts': list(shared_concepts),
            'overlap_length': min(len(overlap_text_prev), len(overlap_text_curr))
        }
    
    def chunk_document(self, 
                      document: Dict[str, Any], 
                      concepts: Dict[str, Any],
                      **kwargs) -> List[ConceptChunk]:
        """
        Create overlapping chunks from document
        
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
        
        # Create overlapping segments
        segments = self.create_overlapping_segments(content)
        
        chunk_index = 0
        prev_chunk_text = None
        
        for segment_text, start_idx, end_idx in segments:
            # Extract concept memberships
            memberships, scores = self.extract_concept_memberships(segment_text, concepts)
            
            # Skip if no concept memberships
            if not memberships:
                continue
            
            # Analyze overlap quality if not first chunk
            overlap_metrics = {}
            if prev_chunk_text:
                overlap_metrics = self.analyze_overlap_quality(
                    prev_chunk_text, segment_text, concepts
                )
            
            metadata = {
                'has_overlap': prev_chunk_text is not None,
                'overlap_size': self.overlap_size if prev_chunk_text else 0,
                'overlap_ratio': self.overlap_ratio,
                'chunk_size': len(segment_text),
                'word_count': len(segment_text.split()),
                'concept_density': len(memberships) / len(segment_text.split()) if segment_text.split() else 0
            }
            
            # Add overlap quality metrics if available
            if overlap_metrics:
                metadata.update({
                    'concept_continuity': overlap_metrics['concept_continuity'],
                    'context_similarity': overlap_metrics['context_similarity'],
                    'shared_concepts_count': len(overlap_metrics['shared_concepts'])
                })
            
            chunk = self.create_chunk(
                doc_id=doc_id,
                content=segment_text,
                chunk_index=chunk_index,
                start_index=start_idx,
                end_index=end_idx,
                concepts=concepts,
                metadata=metadata
            )
            
            chunks.append(chunk)
            prev_chunk_text = segment_text
            chunk_index += 1
        
        return chunks