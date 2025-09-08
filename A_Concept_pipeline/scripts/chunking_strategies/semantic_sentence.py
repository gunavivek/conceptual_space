#!/usr/bin/env python3
"""
A3.1: Semantic Sentence Chunking Strategy
Sentence-level semantic chunking based on concept alignment
"""

from typing import Dict, List, Any
from .base_strategy import BaseChunkingStrategy, ConceptChunk

class SemanticSentenceStrategy(BaseChunkingStrategy):
    """
    Implements sentence-level chunking where each sentence is evaluated
    for its semantic alignment with concept entities
    """
    
    def __init__(self, alignment_threshold: float = 0.3, min_sentence_length: int = 10):
        super().__init__("semantic_sentence")
        self.alignment_threshold = alignment_threshold
        self.min_sentence_length = min_sentence_length
        
    def get_strategy_config(self) -> Dict[str, Any]:
        """Return configuration for this strategy"""
        return {
            'alignment_threshold': self.alignment_threshold,
            'min_sentence_length': self.min_sentence_length,
            'description': 'Sentence-level semantic chunking based on concept alignment'
        }
    
    def chunk_document(self, 
                      document: Dict[str, Any], 
                      concepts: Dict[str, Any],
                      **kwargs) -> List[ConceptChunk]:
        """
        Create semantic chunks from document based on sentence-level concept alignment
        
        Args:
            document: Document with 'doc_id' and 'content'
            concepts: Core and expanded concepts
            **kwargs: Additional parameters (can override threshold)
            
        Returns:
            List of ConceptChunk objects
        """
        chunks = []
        doc_id = document.get('doc_id', 'unknown')
        content = document.get('content', '')
        
        # Allow threshold override
        threshold = kwargs.get('alignment_threshold', self.alignment_threshold)
        
        # Split into sentences with position tracking
        sentences = self.split_sentences(content)
        
        chunk_index = 0
        for sentence_text, start_idx, end_idx in sentences:
            # Skip short sentences
            if len(sentence_text) < self.min_sentence_length:
                continue
            
            # Extract concept memberships for this sentence
            memberships, scores = self.extract_concept_memberships(
                sentence_text, concepts, threshold
            )
            
            # Only create chunk if it has concept memberships
            if memberships:
                metadata = {
                    'sentence_length': len(sentence_text),
                    'word_count': len(sentence_text.split()),
                    'concept_count': len(memberships),
                    'avg_alignment_score': sum(scores.values()) / len(scores) if scores else 0
                }
                
                chunk = self.create_chunk(
                    doc_id=doc_id,
                    content=sentence_text,
                    chunk_index=chunk_index,
                    start_index=start_idx,
                    end_index=end_idx,
                    concepts=concepts,
                    metadata=metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks