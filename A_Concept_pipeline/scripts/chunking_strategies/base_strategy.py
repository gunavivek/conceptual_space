#!/usr/bin/env python3
"""
Base Chunking Strategy Interface
Defines the common interface and utilities for all chunking strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import re

@dataclass
class ConceptChunk:
    """Unified chunk representation across all strategies"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_type: str
    start_index: int
    end_index: int
    concept_memberships: List[str]
    membership_scores: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for JSON serialization"""
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'content': self.content,
            'chunk_type': self.chunk_type,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'concept_memberships': self.concept_memberships,
            'membership_scores': self.membership_scores,
            'metadata': self.metadata
        }


class BaseChunkingStrategy(ABC):
    """
    Abstract base class for all chunking strategies
    Provides common utilities and enforces interface consistency
    """
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.chunks_created = 0
        
    @abstractmethod
    def chunk_document(self, 
                      document: Dict[str, Any], 
                      concepts: Dict[str, Any],
                      **kwargs) -> List[ConceptChunk]:
        """
        Main chunking method that each strategy must implement
        
        Args:
            document: Document dictionary with 'doc_id' and 'content'
            concepts: Dictionary containing core and expanded concepts
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of ConceptChunk objects
        """
        pass
    
    @abstractmethod
    def get_strategy_config(self) -> Dict[str, Any]:
        """
        Return the configuration parameters for this strategy
        """
        pass
    
    def calculate_concept_alignment(self, 
                                   text: str, 
                                   concept_terms: List[str],
                                   use_embeddings: bool = False) -> float:
        """
        Calculate alignment score between text and concept terms
        
        Args:
            text: Text to evaluate
            concept_terms: List of terms representing the concept
            use_embeddings: Whether to use embedding similarity (if available)
            
        Returns:
            Alignment score between 0 and 1
        """
        text_lower = text.lower()
        
        if not concept_terms:
            return 0.0
            
        # Basic keyword matching
        matched_terms = 0
        total_score = 0.0
        
        for term in concept_terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                matched_terms += 1
                # Weight by frequency and position
                count = text_lower.count(term_lower)
                position_weight = 1.0 / (1 + text_lower.index(term_lower) * 0.01)
                total_score += count * position_weight
        
        if matched_terms == 0:
            return 0.0
            
        # Normalize score
        max_possible = len(concept_terms) * 2  # Arbitrary max
        normalized_score = min(1.0, total_score / max_possible)
        
        return normalized_score
    
    def extract_concept_memberships(self,
                                   text: str,
                                   concepts: Dict[str, Any],
                                   threshold: float = 0.3) -> Tuple[List[str], Dict[str, float]]:
        """
        Determine which concepts a text chunk belongs to
        
        Args:
            text: Text to analyze
            concepts: Dictionary of concepts with their terms
            threshold: Minimum alignment score for membership
            
        Returns:
            Tuple of (concept_ids, membership_scores)
        """
        memberships = []
        scores = {}
        
        # Process core concepts
        if 'core' in concepts and 'concepts' in concepts['core']:
            for concept in concepts['core']['concepts']:
                concept_id = concept.get('concept_id', '')
                terms = concept.get('terms', [])
                
                score = self.calculate_concept_alignment(text, terms)
                if score >= threshold:
                    memberships.append(concept_id)
                    scores[concept_id] = score
        
        # Process expanded concepts
        if 'expanded' in concepts and 'expanded_concepts' in concepts['expanded']:
            for concept in concepts['expanded']['expanded_concepts']:
                concept_id = concept.get('concept_id', '')
                terms = concept.get('expanded_terms', [])
                
                score = self.calculate_concept_alignment(text, terms)
                if score >= threshold:
                    memberships.append(concept_id)
                    scores[concept_id] = score
        
        return memberships, scores
    
    def split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with position tracking
        
        Returns:
            List of (sentence, start_index, end_index) tuples
        """
        sentences = []
        
        # Use regex to find sentence boundaries
        sentence_pattern = r'([^.!?]+[.!?])'
        matches = re.finditer(sentence_pattern, text)
        
        for match in matches:
            sentence = match.group(1).strip()
            if len(sentence) > 10:  # Filter very short fragments
                sentences.append((sentence, match.start(), match.end()))
        
        return sentences
    
    def split_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into paragraphs with position tracking
        
        Returns:
            List of (paragraph, start_index, end_index) tuples
        """
        paragraphs = []
        
        # Split by double newlines or indentation
        parts = re.split(r'\n\n+|\n\t+', text)
        
        current_pos = 0
        for part in parts:
            part = part.strip()
            if len(part) > 20:  # Filter very short paragraphs
                start_pos = text.find(part, current_pos)
                end_pos = start_pos + len(part)
                paragraphs.append((part, start_pos, end_pos))
                current_pos = end_pos
        
        return paragraphs
    
    def calculate_overlap_score(self, chunk1: str, chunk2: str) -> float:
        """
        Calculate overlap between two text chunks
        
        Returns:
            Overlap score between 0 and 1
        """
        words1 = set(chunk1.lower().split())
        words2 = set(chunk2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        return f"{doc_id}_{self.strategy_name}_{chunk_index}"
    
    def create_chunk(self,
                    doc_id: str,
                    content: str,
                    chunk_index: int,
                    start_index: int,
                    end_index: int,
                    concepts: Dict[str, Any],
                    metadata: Optional[Dict[str, Any]] = None) -> ConceptChunk:
        """
        Create a ConceptChunk with automatic concept membership detection
        """
        chunk_id = self.generate_chunk_id(doc_id, chunk_index)
        memberships, scores = self.extract_concept_memberships(content, concepts)
        
        if metadata is None:
            metadata = {}
            
        metadata.update({
            'strategy': self.strategy_name,
            'created_at': datetime.now().isoformat()
        })
        
        self.chunks_created += 1
        
        return ConceptChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=content,
            chunk_type=self.strategy_name,
            start_index=start_index,
            end_index=end_index,
            concept_memberships=memberships,
            membership_scores=scores,
            metadata=metadata
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about chunks created by this strategy"""
        return {
            'strategy_name': self.strategy_name,
            'chunks_created': self.chunks_created,
            'config': self.get_strategy_config()
        }