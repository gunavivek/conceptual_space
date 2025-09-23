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
    concept_details: Optional[Dict[str, Any]] = None  # Enhanced: detailed keyword breakdown

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for JSON serialization"""
        base_dict = {
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

        # Always include concept_details field for transparency
        if self.concept_details is not None:
            base_dict['concept_details'] = self.concept_details
        else:
            base_dict['concept_details'] = {}  # Empty dict when no details available

        return base_dict


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
    
    def extract_detailed_keyword_matches(self, text: str, terms: List[str]) -> Dict[str, Any]:
        """
        Extract detailed information about which keywords matched in the text

        Args:
            text: Text to analyze
            terms: List of terms to match

        Returns:
            Dictionary with matched terms and their details
        """
        text_lower = text.lower()
        matched_terms = []
        term_scores = {}

        for term in terms:
            term_lower = term.lower()
            # Count occurrences and calculate score
            if term_lower in text_lower:
                count = text_lower.count(term_lower)
                # Simple scoring: presence + frequency
                score = min(1.0, 0.5 + (count * 0.1))
                matched_terms.append(term)
                term_scores[term] = score

        return {
            'matched_terms': matched_terms,
            'term_scores': term_scores,
            'match_count': len(matched_terms),
            'total_terms': len(terms),
            'coverage_ratio': len(matched_terms) / max(len(terms), 1)
        }

    def extract_concept_memberships_detailed(self,
                                           text: str,
                                           concepts: Dict[str, Any],
                                           threshold: float = 0.3,
                                           doc_id: str = None) -> Tuple[List[str], Dict[str, float], Dict[str, Any]]:
        """
        Enhanced version that provides detailed keyword matching information

        Args:
            text: Text to analyze
            concepts: Dictionary of concepts with their terms
            threshold: Minimum alignment score for membership
            doc_id: Document ID to filter concepts

        Returns:
            Tuple of (concept_ids, membership_scores, concept_details)
        """
        memberships = []
        scores = {}
        concept_details = {}

        # Process core concepts - ONLY from the same document
        if 'core' in concepts and 'concepts' in concepts['core']:
            for concept in concepts['core']['concepts']:
                concept_id = concept.get('concept_id', '')

                # FIXED: Check if concept is related to this document using related_documents list
                if doc_id:
                    related_docs = concept.get('related_documents', [])
                    if doc_id not in related_docs:
                        continue

                core_terms = concept.get('terms', [])
                core_score = self.calculate_concept_alignment(text, core_terms)
                core_details = self.extract_detailed_keyword_matches(text, core_terms)

                # Check for corresponding expanded concept
                expanded_score = 0.0
                expanded_details = {'matched_terms': [], 'term_scores': {}, 'match_count': 0, 'total_terms': 0, 'coverage_ratio': 0.0}
                a25_strategy_contributions = {}

                if 'expanded' in concepts and 'expanded_concepts' in concepts['expanded']:
                    for exp_concept in concepts['expanded']['expanded_concepts']:
                        if exp_concept.get('concept_id') == concept_id:
                            expanded_terms = exp_concept.get('expanded_terms', [])
                            expanded_score = self.calculate_concept_alignment(text, expanded_terms)
                            expanded_details = self.extract_detailed_keyword_matches(text, expanded_terms)

                            # Extract A2.5 strategy contributions if available (disabled for performance)
                            # TODO: Re-enable after optimizing the strategy breakdown extraction
                            a25_strategy_contributions = {}  # Temporarily disabled
                            # a25_strategy_contributions = self.extract_a25_strategy_breakdown(
                            #     text, exp_concept, concepts
                            # )
                            break

                # Calculate total score
                total_score = max(core_score, expanded_score, (core_score + expanded_score) / 2)

                if total_score >= threshold:
                    memberships.append(concept_id)
                    scores[concept_id] = total_score

                    # Store detailed breakdown
                    concept_details[concept_id] = {
                        'core_keywords_matched': core_details['matched_terms'],
                        'core_keyword_scores': core_details['term_scores'],
                        'core_alignment_score': core_score,
                        'core_coverage_ratio': core_details['coverage_ratio'],

                        'expanded_keywords_matched': expanded_details['matched_terms'],
                        'expanded_keyword_scores': expanded_details['term_scores'],
                        'expanded_alignment_score': expanded_score,
                        'expanded_coverage_ratio': expanded_details['coverage_ratio'],

                        'a25_strategy_contributions': a25_strategy_contributions,
                        'total_alignment_score': total_score,

                        'summary': {
                            'core_terms_total': core_details['total_terms'],
                            'core_terms_matched': core_details['match_count'],
                            'expanded_terms_total': expanded_details['total_terms'],
                            'expanded_terms_matched': expanded_details['match_count'],
                            'enhancement_ratio': expanded_details['total_terms'] / max(core_details['total_terms'], 1)
                        }
                    }

        return memberships, scores, concept_details

    def extract_a25_strategy_breakdown(self, text: str, expanded_concept: Dict[str, Any], concepts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract A2.5 strategy-specific contributions to keyword matching

        Args:
            text: Text being analyzed
            expanded_concept: Expanded concept data from A2.5
            concepts: Full concepts dictionary

        Returns:
            Dictionary showing which A2.5 strategies contributed matched keywords
        """
        strategy_contributions = {}

        try:
            # Try to get the original A2.5 data structure with strategy breakdown
            if 'expanded' in concepts and 'expanded_concepts_detailed' in concepts['expanded']:
                # Look for detailed A2.5 data
                concept_id = expanded_concept.get('concept_id', '')
                detailed_data = concepts['expanded']['expanded_concepts_detailed'].get(concept_id, {})

                if 'strategy_contributions' in detailed_data:
                    for strategy_name, strategy_data in detailed_data['strategy_contributions'].items():
                        strategy_terms = strategy_data.get('terms', [])
                        if strategy_terms:  # Only process if there are terms
                            matched_details = self.extract_detailed_keyword_matches(text, strategy_terms)

                            if matched_details['matched_terms']:
                                strategy_contributions[strategy_name] = {
                                    'matched_terms': matched_details['matched_terms'],
                                    'term_scores': matched_details['term_scores'],
                                    'strategy_weight': strategy_data.get('weight', 0.0),
                                    'match_count': matched_details['match_count'],
                                    'total_strategy_terms': len(strategy_terms)
                                }
        except Exception as e:
            # If there's any error with strategy breakdown, just return empty dict
            # This ensures the enhanced method doesn't fail even if A2.5 data is malformed
            pass

        return strategy_contributions

    def extract_concept_memberships(self,
                                   text: str,
                                   concepts: Dict[str, Any],
                                   threshold: float = 0.3,
                                   doc_id: str = None) -> Tuple[List[str], Dict[str, float]]:
        """
        Determine which concepts a text chunk belongs to
        WITH DOCUMENT-AWARE FILTERING to prevent cross-document contamination

        NOTE: This is the original method for backward compatibility.
        Use extract_concept_memberships_detailed() for enhanced output with keyword visibility.

        Args:
            text: Text to analyze
            concepts: Dictionary of concepts with their terms
            threshold: Minimum alignment score for membership
            doc_id: Document ID to filter concepts (only match concepts from same document)

        Returns:
            Tuple of (concept_ids, membership_scores)
        """
        memberships, scores, _ = self.extract_concept_memberships_detailed(text, concepts, threshold, doc_id)
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
    
    def create_chunk_detailed(self,
                            doc_id: str,
                            content: str,
                            chunk_index: int,
                            start_index: int,
                            end_index: int,
                            concepts: Dict[str, Any],
                            metadata: Optional[Dict[str, Any]] = None,
                            concept_details: Optional[Dict[str, Any]] = None) -> ConceptChunk:
        """
        Create a ConceptChunk with detailed keyword visibility (enhanced version)
        """
        chunk_id = self.generate_chunk_id(doc_id, chunk_index)

        # Use provided concept_details or extract them
        if concept_details is None:
            memberships, scores, concept_details = self.extract_concept_memberships_detailed(
                content, concepts, doc_id=doc_id
            )
        else:
            # Extract basic memberships from detailed data
            memberships = list(concept_details.keys())
            scores = {cid: details['total_alignment_score'] for cid, details in concept_details.items()}

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
            metadata=metadata,
            concept_details=concept_details
        )

    def create_chunk(self,
                    doc_id: str,
                    content: str,
                    chunk_index: int,
                    start_index: int,
                    end_index: int,
                    concepts: Dict[str, Any],
                    metadata: Optional[Dict[str, Any]] = None) -> ConceptChunk:
        """
        Create a ConceptChunk with automatic concept membership detection (backward compatible)

        NOTE: For enhanced output with keyword visibility, use create_chunk_detailed()
        """
        chunk_id = self.generate_chunk_id(doc_id, chunk_index)
        # DOCUMENT-AWARE FILTERING: Only match concepts from the same document
        memberships, scores = self.extract_concept_memberships(content, concepts, doc_id=doc_id)

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