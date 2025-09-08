#!/usr/bin/env python3
"""
A3.4: Adaptive Chunking Strategy
Dynamic chunk sizing based on content density and concept distribution
"""

from typing import Dict, List, Any
import numpy as np
from .base_strategy import BaseChunkingStrategy, ConceptChunk

class AdaptiveChunkingStrategy(BaseChunkingStrategy):
    """
    Implements adaptive chunking that dynamically adjusts chunk size
    based on concept density, content complexity, and semantic coherence
    """
    
    def __init__(self, 
                 min_chunk_size: int = 50,
                 max_chunk_size: int = 500,
                 target_concepts_per_chunk: int = 3):
        super().__init__("adaptive_chunking")
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_concepts_per_chunk = target_concepts_per_chunk
        
    def get_strategy_config(self) -> Dict[str, Any]:
        """Return configuration for this strategy"""
        return {
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'target_concepts_per_chunk': self.target_concepts_per_chunk,
            'description': 'Adaptive chunking with dynamic size based on concept density'
        }
    
    def calculate_concept_density(self, text: str, concepts: Dict[str, Any]) -> float:
        """
        Calculate the concept density of a text segment
        
        Returns:
            Density score (concepts per word)
        """
        memberships, scores = self.extract_concept_memberships(text, concepts, threshold=0.2)
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
            
        return len(memberships) / word_count
    
    def find_optimal_chunk_boundary(self, 
                                   text: str, 
                                   start_pos: int,
                                   concepts: Dict[str, Any]) -> int:
        """
        Find the optimal position to end a chunk based on concept distribution
        
        Args:
            text: Full text to chunk
            start_pos: Starting position of current chunk
            concepts: Concept dictionary
            
        Returns:
            Optimal end position for the chunk
        """
        # Start with minimum chunk size
        current_pos = start_pos + self.min_chunk_size
        
        if current_pos >= len(text):
            return len(text)
        
        best_pos = current_pos
        best_score = 0.0
        
        # Evaluate positions between min and max chunk size
        while current_pos < min(start_pos + self.max_chunk_size, len(text)):
            # Get chunk candidate
            chunk_text = text[start_pos:current_pos]
            
            # Calculate metrics for this chunk size
            density = self.calculate_concept_density(chunk_text, concepts)
            memberships, scores = self.extract_concept_memberships(chunk_text, concepts)
            
            # Score based on how close we are to target concepts
            concept_count = len(memberships)
            concept_score = 1.0 - abs(concept_count - self.target_concepts_per_chunk) / self.target_concepts_per_chunk
            
            # Score based on ending at sentence boundary
            boundary_score = 1.0 if text[current_pos-1:current_pos] in '.!?' else 0.5
            
            # Combined score
            total_score = (concept_score * 0.6 + boundary_score * 0.2 + density * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_pos = current_pos
            
            # Move to next potential boundary
            next_sentence = text.find('.', current_pos)
            if next_sentence == -1 or next_sentence > start_pos + self.max_chunk_size:
                break
            current_pos = next_sentence + 1
        
        return best_pos
    
    def chunk_document(self, 
                      document: Dict[str, Any], 
                      concepts: Dict[str, Any],
                      **kwargs) -> List[ConceptChunk]:
        """
        Create adaptive chunks from document
        
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
        
        # Process document adaptively
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(content):
            # Find optimal chunk boundary
            end_pos = self.find_optimal_chunk_boundary(content, current_pos, concepts)
            
            # Extract chunk text
            chunk_text = content[current_pos:end_pos].strip()
            
            if len(chunk_text) < self.min_chunk_size and current_pos + self.min_chunk_size < len(content):
                # Too small, extend to minimum size
                end_pos = min(current_pos + self.min_chunk_size, len(content))
                chunk_text = content[current_pos:end_pos].strip()
            
            # Extract concept memberships
            memberships, scores = self.extract_concept_memberships(chunk_text, concepts)
            
            # Create chunk if it has concept memberships
            if memberships:
                density = self.calculate_concept_density(chunk_text, concepts)
                
                metadata = {
                    'chunk_size': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'concept_density': density,
                    'concepts_found': len(memberships),
                    'size_category': 'small' if len(chunk_text) < 150 else 'medium' if len(chunk_text) < 350 else 'large',
                    'avg_concept_score': np.mean(list(scores.values())) if scores else 0
                }
                
                chunk = self.create_chunk(
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_index=current_pos,
                    end_index=end_pos,
                    concepts=concepts,
                    metadata=metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            current_pos = end_pos
        
        return chunks