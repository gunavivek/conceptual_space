#!/usr/bin/env python3
"""
A3.2: Paragraph-Aware Chunking Strategy
Paragraph-level chunking that preserves document structure
"""

from typing import Dict, List, Any
from .base_strategy import BaseChunkingStrategy, ConceptChunk

class ParagraphAwareStrategy(BaseChunkingStrategy):
    """
    Implements paragraph-aware chunking that maintains natural document boundaries
    while ensuring concept alignment
    """
    
    def __init__(self, min_paragraph_length: int = 20, merge_threshold: float = 0.7):
        super().__init__("paragraph_aware")
        self.min_paragraph_length = min_paragraph_length
        self.merge_threshold = merge_threshold  # For merging similar adjacent paragraphs
        
    def get_strategy_config(self) -> Dict[str, Any]:
        """Return configuration for this strategy"""
        return {
            'min_paragraph_length': self.min_paragraph_length,
            'merge_threshold': self.merge_threshold,
            'description': 'Paragraph-level chunking preserving document structure'
        }
    
    def should_merge_paragraphs(self, para1: str, para2: str, concepts: Dict[str, Any]) -> bool:
        """
        Determine if two adjacent paragraphs should be merged based on concept similarity
        """
        # Get concept memberships for each paragraph
        # Note: For similarity calculation, we use generic matching
        memberships1, scores1 = self.extract_concept_memberships(para1, concepts)
        memberships2, scores2 = self.extract_concept_memberships(para2, concepts)
        
        # Calculate Jaccard similarity of concept memberships
        if not memberships1 or not memberships2:
            return False
            
        set1 = set(memberships1)
        set2 = set(memberships2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= self.merge_threshold
    
    def chunk_document(self, 
                      document: Dict[str, Any], 
                      concepts: Dict[str, Any],
                      **kwargs) -> List[ConceptChunk]:
        """
        Create paragraph-aware chunks from document
        
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
        
        # Split into paragraphs with position tracking
        paragraphs = self.split_paragraphs(content)
        
        # Process paragraphs, potentially merging adjacent ones
        chunk_index = 0
        i = 0
        
        while i < len(paragraphs):
            para_text, start_idx, end_idx = paragraphs[i]
            
            # Skip short paragraphs
            if len(para_text) < self.min_paragraph_length:
                i += 1
                continue
            
            # Check if we should merge with next paragraph
            merged_text = para_text
            merged_end = end_idx
            
            while i + 1 < len(paragraphs):
                next_para_text, next_start, next_end = paragraphs[i + 1]
                
                if self.should_merge_paragraphs(merged_text, next_para_text, concepts):
                    merged_text = content[start_idx:next_end]
                    merged_end = next_end
                    i += 1
                else:
                    break
            
            # Extract concept memberships
            # DOCUMENT-AWARE: Only match concepts from same document
            memberships, scores = self.extract_concept_memberships(merged_text, concepts, doc_id=doc_id)
            
            # Create chunk if it has concept memberships
            if memberships:
                metadata = {
                    'paragraph_count': (merged_end - start_idx) // len(para_text) if len(para_text) > 0 else 1,
                    'char_count': len(merged_text),
                    'word_count': len(merged_text.split()),
                    'concept_density': len(memberships) / len(merged_text.split()) if merged_text.split() else 0
                }
                
                chunk = self.create_chunk_detailed(
                    doc_id=doc_id,
                    content=merged_text,
                    chunk_index=chunk_index,
                    start_index=start_idx,
                    end_index=merged_end,
                    concepts=concepts,
                    metadata=metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            i += 1
        
        return chunks