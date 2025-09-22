#!/usr/bin/env python3
"""
A3.5: Concept-Aware Chunking Strategy
Chunking guided by concept boundaries and centroid distances
"""

from typing import Dict, List, Any, Set
import numpy as np
from collections import defaultdict
from .base_strategy import BaseChunkingStrategy, ConceptChunk

class ConceptAwareStrategy(BaseChunkingStrategy):
    """
    Implements concept-guided chunking where chunk boundaries are determined
    by concept centroids and their influence regions (convex balls)
    """
    
    def __init__(self, 
                 centroid_threshold: float = 0.5,
                 max_concepts_per_chunk: int = 5,
                 overlap_allowed: bool = True):
        super().__init__("concept_aware")
        self.centroid_threshold = centroid_threshold
        self.max_concepts_per_chunk = max_concepts_per_chunk
        self.overlap_allowed = overlap_allowed
        
    def get_strategy_config(self) -> Dict[str, Any]:
        """Return configuration for this strategy"""
        return {
            'centroid_threshold': self.centroid_threshold,
            'max_concepts_per_chunk': self.max_concepts_per_chunk,
            'overlap_allowed': self.overlap_allowed,
            'description': 'Concept-guided chunking based on centroid distances'
        }
    
    def calculate_centroid_distance(self, text: str, concept_terms: List[str]) -> float:
        """
        Calculate distance from text to concept centroid
        
        Args:
            text: Text segment
            concept_terms: Terms defining the concept centroid
            
        Returns:
            Distance score (0 = at centroid, 1 = far from centroid)
        """
        if not concept_terms:
            return 1.0
            
        alignment = self.calculate_concept_alignment(text, concept_terms)
        # Convert alignment to distance
        distance = 1.0 - alignment
        return distance
    
    def identify_concept_regions(self, 
                                text: str, 
                                concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify regions in text that belong to different concept centroids
        
        Returns:
            List of regions with their dominant concepts
        """
        regions = []
        sentences = self.split_sentences(text)
        
        # Build concept influence map
        concept_influences = defaultdict(list)
        
        for sent_text, start_idx, end_idx in sentences:
            # Calculate distance to each concept centroid
            concept_distances = {}
            
            # Process core concepts
            if 'core' in concepts and 'concepts' in concepts['core']:
                for concept in concepts['core']['concepts']:
                    concept_id = concept.get('concept_id', '')
                    terms = concept.get('terms', [])
                    distance = self.calculate_centroid_distance(sent_text, terms)
                    
                    if distance <= self.centroid_threshold:
                        concept_distances[concept_id] = distance
            
            # Process expanded concepts
            if 'expanded' in concepts and 'expanded_concepts' in concepts['expanded']:
                for concept in concepts['expanded']['expanded_concepts']:
                    concept_id = concept.get('concept_id', '')
                    terms = concept.get('expanded_terms', [])
                    distance = self.calculate_centroid_distance(sent_text, terms)
                    
                    if distance <= self.centroid_threshold:
                        concept_distances[concept_id] = distance
            
            # Assign sentence to concepts within threshold
            for concept_id, distance in concept_distances.items():
                concept_influences[concept_id].append({
                    'text': sent_text,
                    'start': start_idx,
                    'end': end_idx,
                    'distance': distance
                })
        
        # Group adjacent sentences with same dominant concepts
        for concept_id, influenced_sentences in concept_influences.items():
            if not influenced_sentences:
                continue
                
            # Sort by position
            influenced_sentences.sort(key=lambda x: x['start'])
            
            # Group adjacent sentences
            current_region = {
                'concept_id': concept_id,
                'sentences': [influenced_sentences[0]],
                'start': influenced_sentences[0]['start'],
                'end': influenced_sentences[0]['end']
            }
            
            for sent in influenced_sentences[1:]:
                # Check if adjacent (within reasonable gap)
                if sent['start'] - current_region['end'] < 100:
                    current_region['sentences'].append(sent)
                    current_region['end'] = sent['end']
                else:
                    # Save current region and start new one
                    if len(current_region['sentences']) > 0:
                        regions.append(current_region)
                    current_region = {
                        'concept_id': concept_id,
                        'sentences': [sent],
                        'start': sent['start'],
                        'end': sent['end']
                    }
            
            # Save final region
            if len(current_region['sentences']) > 0:
                regions.append(current_region)
        
        return regions
    
    def chunk_document(self, 
                      document: Dict[str, Any], 
                      concepts: Dict[str, Any],
                      **kwargs) -> List[ConceptChunk]:
        """
        Create concept-aware chunks from document
        
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
        
        # Identify concept regions
        regions = self.identify_concept_regions(content, concepts)
        
        # Sort regions by start position
        regions.sort(key=lambda x: x['start'])
        
        # Process regions, potentially merging overlapping ones
        chunk_index = 0
        processed_regions = []
        
        for region in regions:
            # Check for overlap with existing chunks if not allowed
            if not self.overlap_allowed:
                overlaps = False
                for proc_region in processed_regions:
                    if (region['start'] < proc_region['end'] and 
                        region['end'] > proc_region['start']):
                        overlaps = True
                        break
                
                if overlaps:
                    continue
            
            # Extract region text
            region_text = content[region['start']:region['end']]
            
            # Get all concepts for this region
            # DOCUMENT-AWARE: Only match concepts from same document
            memberships, scores = self.extract_concept_memberships(region_text, concepts, doc_id=doc_id)
            
            # Limit concepts per chunk if specified
            if len(memberships) > self.max_concepts_per_chunk:
                # Keep only top concepts by score
                sorted_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                memberships = [c[0] for c in sorted_concepts[:self.max_concepts_per_chunk]]
                scores = {c[0]: c[1] for c in sorted_concepts[:self.max_concepts_per_chunk]}
            
            # Calculate average distance to centroids
            avg_distance = np.mean([1.0 - scores.get(m, 0) for m in memberships]) if memberships else 1.0
            
            metadata = {
                'primary_concept': region['concept_id'],
                'concept_count': len(memberships),
                'avg_centroid_distance': avg_distance,
                'region_sentences': len(region['sentences']),
                'is_multi_concept': len(memberships) > 1,
                'concept_overlap_ratio': len(set(memberships)) / len(memberships) if memberships else 0
            }
            
            chunk = self.create_chunk(
                doc_id=doc_id,
                content=region_text,
                chunk_index=chunk_index,
                start_index=region['start'],
                end_index=region['end'],
                concepts=concepts,
                metadata=metadata
            )
            
            chunks.append(chunk)
            processed_regions.append(region)
            chunk_index += 1
        
        return chunks