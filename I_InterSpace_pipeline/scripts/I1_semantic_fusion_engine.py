#!/usr/bin/env python3
"""
I1 Semantic Fusion Engine
Advanced tri-semantic integration system that fuses insights from:
- Ontology Space (Authoritative BIZBOK knowledge)
- Document Space (Real-world empirical evidence)  
- Question Space (User intent and behavior patterns)

Creates unified semantic understanding through multiple fusion strategies
PhD Research Contribution: Novel semantic fusion algorithms for multi-modal AI
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
import math

class SemanticFusionEngine:
    """
    Advanced semantic fusion engine that combines insights from three semantic spaces
    Uses multiple fusion strategies to create comprehensive understanding
    """
    
    def __init__(self, ontology_space, document_space, question_space, cross_space_bridges):
        self.ontology_space = ontology_space
        self.document_space = document_space
        self.question_space = question_space
        self.cross_space_bridges = cross_space_bridges
        
        # Fusion strategy configurations
        self.fusion_strategies = {
            'consensus_weighting': {
                'weight_authority': 0.4,    # BIZBOK ontology weight
                'weight_empirical': 0.35,   # Document evidence weight
                'weight_user_intent': 0.25, # Question pattern weight
                'consensus_threshold': 0.7
            },
            'authority_preference': {
                'ontology_boost': 1.2,      # Boost authoritative sources
                'evidence_requirement': 0.3, # Minimum empirical evidence
                'user_validation': 0.2      # Minimum user intent validation
            },
            'evidence_strength': {
                'document_frequency_weight': 0.3,
                'coherence_weight': 0.25,
                'intent_alignment_weight': 0.25,
                'authority_weight': 0.2
            },
            'adaptive_learning': {
                'success_reinforcement': 1.1,  # Boost successful patterns
                'failure_dampening': 0.9,      # Reduce failed patterns
                'novelty_exploration': 0.15    # Explore new relationships
            }
        }
        
        # Fusion quality metrics
        self.fusion_quality_metrics = {
            'consensus_relationships': [],
            'high_confidence_fusions': [],
            'novel_discoveries': [],
            'validation_conflicts': []
        }
    
    def fuse_tri_semantic_perspectives(self, concept: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Create unified semantic understanding by fusing all three perspective spaces
        
        Args:
            concept: The concept to analyze
            context: Optional context for context-aware fusion
            
        Returns:
            Comprehensive fused semantic understanding
        """
        
        # Step 1: Gather perspectives from each space
        perspectives = self._gather_tri_space_perspectives(concept)
        
        # Step 2: Apply fusion strategies
        fusion_results = {}
        
        # Strategy 1: Consensus-based fusion
        fusion_results['consensus_fusion'] = self._consensus_based_fusion(concept, perspectives)
        
        # Strategy 2: Authority-guided fusion
        fusion_results['authority_fusion'] = self._authority_guided_fusion(concept, perspectives)
        
        # Strategy 3: Evidence-strength fusion
        fusion_results['evidence_fusion'] = self._evidence_based_fusion(concept, perspectives)
        
        # Strategy 4: Context-aware fusion (if context provided)
        if context:
            fusion_results['context_fusion'] = self._context_aware_fusion(concept, perspectives, context)
        
        # Step 3: Meta-fusion - combine all fusion strategies
        unified_understanding = self._meta_fusion(concept, fusion_results, perspectives)
        
        # Step 4: Calculate fusion quality metrics
        fusion_quality = self._calculate_fusion_quality(unified_understanding, perspectives)
        unified_understanding['fusion_quality'] = fusion_quality
        
        return unified_understanding
    
    def _gather_tri_space_perspectives(self, concept: str) -> Dict[str, Any]:
        """Gather perspectives from all three semantic spaces"""
        
        perspectives = {
            'ontology': {'available': False, 'data': {}},
            'document': {'available': False, 'data': {}},
            'question': {'available': False, 'data': {}}
        }
        
        # Ontology space perspective
        try:
            if concept in self.ontology_space.get_all_concepts():
                perspectives['ontology'] = {
                    'available': True,
                    'data': self.ontology_space.get_concept_authority_view(concept)
                }
        except Exception as e:
            print(f"Warning: Could not get ontology perspective for {concept}: {e}")
        
        # Document space perspective  
        try:
            if concept in self.document_space.get_all_concepts():
                perspectives['document'] = {
                    'available': True,
                    'data': self.document_space.get_concept_reality_view(concept)
                }
        except Exception as e:
            print(f"Warning: Could not get document perspective for {concept}: {e}")
        
        # Question space perspective
        try:
            if concept in self.question_space.get_all_concepts():
                perspectives['question'] = {
                    'available': True,
                    'data': self.question_space.get_concept_intent_view(concept)
                }
        except Exception as e:
            print(f"Warning: Could not get question perspective for {concept}: {e}")
        
        return perspectives
    
    def _consensus_based_fusion(self, concept: str, perspectives: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse based on consensus across multiple spaces"""
        
        fusion_config = self.fusion_strategies['consensus_weighting']
        
        # Collect relationships from each space
        all_relationships = defaultdict(list)
        confidence_sources = defaultdict(list)
        
        # Ontology relationships
        if perspectives['ontology']['available']:
            ont_data = perspectives['ontology']['data']
            ont_relationships = ont_data.get('authoritative_relationships', [])
            for rel in ont_relationships:
                all_relationships[rel].append('ontology')
                confidence_sources[rel].append(ont_data.get('authority_confidence', 0.9))
        
        # Document relationships  
        if perspectives['document']['available']:
            doc_data = perspectives['document']['data']
            doc_relationships = self.document_space.get_concept_relationships(concept)
            for rel in doc_relationships:
                all_relationships[rel].append('document')
                confidence_sources[rel].append(doc_data.get('empirical_confidence', 0.7))
        
        # Question relationships
        if perspectives['question']['available']:
            q_data = perspectives['question']['data']
            q_relationships = self.question_space.get_concept_relationships(concept)
            for rel in q_relationships:
                all_relationships[rel].append('question')
                confidence_sources[rel].append(q_data.get('intent_confidence', 0.6))
        
        # Calculate consensus relationships
        consensus_relationships = {}
        for relationship, sources in all_relationships.items():
            source_count = len(sources)
            unique_sources = len(set(sources))
            
            # Calculate weighted confidence
            weighted_confidence = 0.0
            if 'ontology' in sources:
                weighted_confidence += fusion_config['weight_authority']
            if 'document' in sources:
                weighted_confidence += fusion_config['weight_empirical']
            if 'question' in sources:
                weighted_confidence += fusion_config['weight_user_intent']
            
            # Boost for multi-source consensus
            consensus_boost = 1.0 + (unique_sources - 1) * 0.2
            final_confidence = min(1.0, weighted_confidence * consensus_boost)
            
            if final_confidence >= fusion_config['consensus_threshold']:
                consensus_relationships[relationship] = {
                    'confidence': final_confidence,
                    'sources': sources,
                    'unique_source_count': unique_sources,
                    'consensus_level': 'high' if unique_sources >= 2 else 'medium'
                }
        
        return {
            'fusion_method': 'consensus_weighting',
            'consensus_relationships': consensus_relationships,
            'total_consensus_count': len(consensus_relationships),
            'multi_source_count': len([r for r in consensus_relationships.values() if r['unique_source_count'] > 1])
        }
    
    def _authority_guided_fusion(self, concept: str, perspectives: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse with preference for authoritative sources, validated by evidence"""
        
        fusion_config = self.fusion_strategies['authority_preference']
        authority_relationships = {}
        
        if not perspectives['ontology']['available']:
            return {'fusion_method': 'authority_guided', 'authority_relationships': {}}
        
        ont_data = perspectives['ontology']['data']
        ont_relationships = ont_data.get('authoritative_relationships', [])
        
        for relationship in ont_relationships:
            # Start with ontology authority
            base_confidence = ont_data.get('authority_confidence', 0.9) * fusion_config['ontology_boost']
            
            # Check for empirical validation
            empirical_validation = 0.0
            if perspectives['document']['available']:
                doc_relationships = self.document_space.get_concept_relationships(concept)
                if relationship in doc_relationships:
                    doc_data = perspectives['document']['data']
                    empirical_validation = doc_data.get('empirical_confidence', 0.7)
            
            # Check for user intent validation
            intent_validation = 0.0
            if perspectives['question']['available']:
                q_relationships = self.question_space.get_concept_relationships(concept)
                if relationship in q_relationships:
                    q_data = perspectives['question']['data']
                    intent_validation = q_data.get('intent_confidence', 0.6)
            
            # Calculate authority-guided confidence
            if (empirical_validation >= fusion_config['evidence_requirement'] and 
                intent_validation >= fusion_config['user_validation']):
                
                # Full validation: authority + evidence + user intent
                final_confidence = min(1.0, base_confidence + empirical_validation * 0.3 + intent_validation * 0.2)
                validation_level = 'full_validation'
                
            elif empirical_validation >= fusion_config['evidence_requirement']:
                # Empirical validation only
                final_confidence = min(1.0, base_confidence + empirical_validation * 0.2)
                validation_level = 'empirical_validation'
                
            elif intent_validation >= fusion_config['user_validation']:
                # User intent validation only
                final_confidence = min(1.0, base_confidence + intent_validation * 0.15)
                validation_level = 'intent_validation'
                
            else:
                # Authority only
                final_confidence = min(1.0, base_confidence)
                validation_level = 'authority_only'
            
            authority_relationships[relationship] = {
                'confidence': final_confidence,
                'validation_level': validation_level,
                'authority_base': ont_data.get('authority_confidence', 0.9),
                'empirical_support': empirical_validation,
                'intent_support': intent_validation
            }
        
        return {
            'fusion_method': 'authority_guided',
            'authority_relationships': authority_relationships,
            'fully_validated_count': len([r for r in authority_relationships.values() 
                                        if r['validation_level'] == 'full_validation'])
        }
    
    def _evidence_based_fusion(self, concept: str, perspectives: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse based on strength of empirical evidence across spaces"""
        
        fusion_config = self.fusion_strategies['evidence_strength']
        evidence_relationships = {}
        
        # Collect all relationships with evidence scores
        relationship_evidence = defaultdict(lambda: {
            'document_frequency': 0.0,
            'coherence_score': 0.0,
            'intent_alignment': 0.0,
            'authority_weight': 0.0,
            'sources': []
        })
        
        # Document evidence
        if perspectives['document']['available']:
            doc_data = perspectives['document']['data']
            doc_relationships = self.document_space.get_concept_relationships(concept)
            
            for rel in doc_relationships:
                relationship_evidence[rel]['document_frequency'] = min(1.0, doc_data.get('document_frequency', 0) * 0.1)
                relationship_evidence[rel]['coherence_score'] = doc_data.get('average_coherence', 0.5)
                relationship_evidence[rel]['sources'].append('document')
        
        # Question intent evidence
        if perspectives['question']['available']:
            q_data = perspectives['question']['data']
            q_relationships = self.question_space.get_concept_relationships(concept)
            
            for rel in q_relationships:
                relationship_evidence[rel]['intent_alignment'] = q_data.get('intent_confidence', 0.6)
                relationship_evidence[rel]['sources'].append('question')
        
        # Authority evidence
        if perspectives['ontology']['available']:
            ont_data = perspectives['ontology']['data']
            ont_relationships = ont_data.get('authoritative_relationships', [])
            
            for rel in ont_relationships:
                relationship_evidence[rel]['authority_weight'] = ont_data.get('authority_confidence', 0.9)
                relationship_evidence[rel]['sources'].append('ontology')
        
        # Calculate evidence-based confidence
        for relationship, evidence in relationship_evidence.items():
            evidence_score = (
                evidence['document_frequency'] * fusion_config['document_frequency_weight'] +
                evidence['coherence_score'] * fusion_config['coherence_weight'] +
                evidence['intent_alignment'] * fusion_config['intent_alignment_weight'] +
                evidence['authority_weight'] * fusion_config['authority_weight']
            )
            
            # Boost for multiple evidence sources
            source_boost = 1.0 + (len(set(evidence['sources'])) - 1) * 0.15
            final_confidence = min(1.0, evidence_score * source_boost)
            
            if final_confidence > 0.3:  # Only include relationships with meaningful evidence
                evidence_relationships[relationship] = {
                    'confidence': final_confidence,
                    'evidence_breakdown': evidence,
                    'evidence_strength': 'strong' if final_confidence > 0.7 else 
                                       'medium' if final_confidence > 0.5 else 'weak'
                }
        
        return {
            'fusion_method': 'evidence_based',
            'evidence_relationships': evidence_relationships,
            'strong_evidence_count': len([r for r in evidence_relationships.values() 
                                        if r['evidence_strength'] == 'strong'])
        }
    
    def _context_aware_fusion(self, concept: str, perspectives: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Perform context-aware fusion based on specific use context"""
        
        context_relationships = {}
        
        # Simple context-aware logic (can be enhanced with NLP)
        context_lower = context.lower()
        
        # Context keywords that boost certain relationship types
        context_boosts = {
            'financial': {'Tax': 0.2, 'Financial Transaction': 0.2, 'Asset': 0.2, 'Payment': 0.2},
            'legal': {'Agreement': 0.2, 'Legal Proceeding': 0.2, 'Partner': 0.2},
            'customer': {'Customer': 0.15, 'Partner': 0.15, 'Agreement': 0.15},
            'business': {'Business Entity': 0.15, 'Strategy': 0.15, 'Plan': 0.15}
        }
        
        # Apply context boosts
        base_relationships = {}
        
        # Gather relationships from all spaces
        for space_name, space_data in perspectives.items():
            if space_data['available']:
                if space_name == 'ontology':
                    relationships = space_data['data'].get('authoritative_relationships', [])
                elif space_name == 'document':
                    relationships = self.document_space.get_concept_relationships(concept)
                elif space_name == 'question':
                    relationships = self.question_space.get_concept_relationships(concept)
                else:
                    continue
                    
                for rel in relationships:
                    if rel not in base_relationships:
                        base_relationships[rel] = {'confidence': 0.5, 'sources': []}
                    base_relationships[rel]['sources'].append(space_name)
        
        # Apply context boosts
        for relationship, data in base_relationships.items():
            final_confidence = data['confidence']
            
            # Check for context relevance
            for context_key, boosts in context_boosts.items():
                if context_key in context_lower:
                    for boost_concept, boost_value in boosts.items():
                        if boost_concept.lower() in relationship.lower():
                            final_confidence += boost_value
                            break
            
            context_relationships[relationship] = {
                'confidence': min(1.0, final_confidence),
                'context_relevance': self._calculate_context_relevance(relationship, context),
                'sources': data['sources']
            }
        
        return {
            'fusion_method': 'context_aware',
            'context': context,
            'context_relationships': context_relationships,
            'high_relevance_count': len([r for r in context_relationships.values() 
                                       if r['context_relevance'] > 0.7])
        }
    
    def _meta_fusion(self, concept: str, fusion_results: Dict[str, Any], perspectives: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from all fusion strategies into unified understanding"""
        
        # Collect all relationships from different fusion strategies
        all_fused_relationships = defaultdict(lambda: {
            'fusion_sources': [],
            'confidence_scores': [],
            'evidence_types': []
        })
        
        # Aggregate relationships from each fusion strategy
        for fusion_type, fusion_data in fusion_results.items():
            relationships_key = None
            
            if fusion_type == 'consensus_fusion':
                relationships_key = 'consensus_relationships'
            elif fusion_type == 'authority_fusion':
                relationships_key = 'authority_relationships'
            elif fusion_type == 'evidence_fusion':
                relationships_key = 'evidence_relationships'
            elif fusion_type == 'context_fusion':
                relationships_key = 'context_relationships'
            
            if relationships_key and relationships_key in fusion_data:
                for rel, rel_data in fusion_data[relationships_key].items():
                    all_fused_relationships[rel]['fusion_sources'].append(fusion_type)
                    all_fused_relationships[rel]['confidence_scores'].append(rel_data.get('confidence', 0.5))
                    
                    # Add evidence type
                    if fusion_type == 'consensus_fusion':
                        all_fused_relationships[rel]['evidence_types'].append('multi_space_consensus')
                    elif fusion_type == 'authority_fusion':
                        all_fused_relationships[rel]['evidence_types'].append('authoritative_validation')
                    elif fusion_type == 'evidence_fusion':
                        all_fused_relationships[rel]['evidence_types'].append('empirical_evidence')
                    elif fusion_type == 'context_fusion':
                        all_fused_relationships[rel]['evidence_types'].append('contextual_relevance')
        
        # Calculate unified confidence for each relationship
        unified_relationships = {}
        for relationship, agg_data in all_fused_relationships.items():
            # Calculate weighted average confidence
            confidence_scores = agg_data['confidence_scores']
            fusion_sources = agg_data['fusion_sources']
            
            if not confidence_scores:
                continue
            
            # Weight fusion strategies
            strategy_weights = {
                'consensus_fusion': 0.3,
                'authority_fusion': 0.25,
                'evidence_fusion': 0.25,
                'context_fusion': 0.2
            }
            
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for i, source in enumerate(fusion_sources):
                weight = strategy_weights.get(source, 0.2)
                weighted_confidence += confidence_scores[i] * weight
                total_weight += weight
            
            if total_weight > 0:
                final_confidence = weighted_confidence / total_weight
                
                # Boost for multiple fusion strategy agreement
                strategy_boost = 1.0 + (len(set(fusion_sources)) - 1) * 0.1
                final_confidence = min(1.0, final_confidence * strategy_boost)
                
                unified_relationships[relationship] = {
                    'unified_confidence': final_confidence,
                    'fusion_sources': list(set(fusion_sources)),
                    'evidence_types': list(set(agg_data['evidence_types'])),
                    'fusion_agreement_level': len(set(fusion_sources)),
                    'relationship_quality': 'high' if final_confidence > 0.8 else 
                                          'medium' if final_confidence > 0.6 else 'low'
                }
        
        # Create comprehensive unified understanding
        unified_understanding = {
            'concept': concept,
            'fusion_timestamp': datetime.now().isoformat(),
            'source_spaces': [space for space, data in perspectives.items() if data['available']],
            'unified_relationships': unified_relationships,
            'fusion_strategies_used': list(fusion_results.keys()),
            'total_relationships': len(unified_relationships),
            'high_quality_relationships': len([r for r in unified_relationships.values() 
                                             if r['relationship_quality'] == 'high']),
            'fusion_completeness': len(perspectives) / 3.0,  # How many spaces contributed
            'strategy_fusion_results': fusion_results
        }
        
        return unified_understanding
    
    def _calculate_fusion_quality(self, unified_understanding: Dict[str, Any], perspectives: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the fusion process"""
        
        relationships = unified_understanding.get('unified_relationships', {})
        
        # Basic quality metrics
        total_relationships = len(relationships)
        high_quality_count = len([r for r in relationships.values() if r['relationship_quality'] == 'high'])
        multi_source_count = len([r for r in relationships.values() if r['fusion_agreement_level'] > 1])
        
        # Coverage metrics
        source_spaces = unified_understanding.get('source_spaces', [])
        space_coverage = len(source_spaces) / 3.0  # Out of 3 possible spaces
        
        # Consensus metrics
        consensus_relationships = len([r for r in relationships.values() 
                                     if 'multi_space_consensus' in r.get('evidence_types', [])])
        
        # Confidence distribution
        confidence_scores = [r['unified_confidence'] for r in relationships.values()]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        confidence_std = np.std(confidence_scores) if confidence_scores else 0.0
        
        return {
            'total_relationships': total_relationships,
            'high_quality_ratio': high_quality_count / max(1, total_relationships),
            'multi_source_ratio': multi_source_count / max(1, total_relationships),
            'space_coverage': space_coverage,
            'consensus_ratio': consensus_relationships / max(1, total_relationships),
            'average_confidence': avg_confidence,
            'confidence_stability': 1.0 - min(1.0, confidence_std),
            'fusion_quality_score': (
                (high_quality_count / max(1, total_relationships)) * 0.3 +
                (multi_source_count / max(1, total_relationships)) * 0.25 +
                space_coverage * 0.2 +
                (consensus_relationships / max(1, total_relationships)) * 0.15 +
                avg_confidence * 0.1
            )
        }
    
    def _calculate_context_relevance(self, relationship: str, context: str) -> float:
        """Calculate how relevant a relationship is to the given context"""
        
        # Simple keyword-based relevance (can be enhanced with semantic similarity)
        relationship_lower = relationship.lower()
        context_lower = context.lower()
        
        # Count common words
        rel_words = set(relationship_lower.split())
        context_words = set(context_lower.split())
        
        if not rel_words or not context_words:
            return 0.5  # Default relevance
        
        # Calculate Jaccard similarity
        intersection = len(rel_words & context_words)
        union = len(rel_words | context_words)
        
        relevance = intersection / union if union > 0 else 0.0
        
        # Boost for exact matches
        if any(word in context_lower for word in rel_words):
            relevance += 0.2
        
        return min(1.0, relevance)
    
    def batch_fuse_concepts(self, concepts: List[str], context: Optional[str] = None) -> Dict[str, Any]:
        """Perform fusion for multiple concepts efficiently"""
        
        batch_results = {}
        batch_start_time = datetime.now()
        
        for i, concept in enumerate(concepts):
            try:
                fusion_result = self.fuse_tri_semantic_perspectives(concept, context)
                batch_results[concept] = fusion_result
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(concepts)} concepts...")
                    
            except Exception as e:
                print(f"Warning: Failed to fuse concept '{concept}': {e}")
                batch_results[concept] = {
                    'concept': concept,
                    'error': str(e),
                    'fusion_status': 'failed'
                }
        
        batch_end_time = datetime.now()
        processing_time = (batch_end_time - batch_start_time).total_seconds()
        
        return {
            'batch_results': batch_results,
            'batch_statistics': {
                'total_concepts': len(concepts),
                'successful_fusions': len([r for r in batch_results.values() if 'error' not in r]),
                'failed_fusions': len([r for r in batch_results.values() if 'error' in r]),
                'processing_time_seconds': processing_time,
                'concepts_per_second': len(concepts) / max(0.1, processing_time)
            }
        }
    
    def analyze_fusion_patterns(self, fusion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across multiple fusion results"""
        
        if not fusion_results or 'batch_results' not in fusion_results:
            return {}
        
        batch_results = fusion_results['batch_results']
        successful_results = [r for r in batch_results.values() if 'error' not in r]
        
        if not successful_results:
            return {'message': 'No successful fusion results to analyze'}
        
        # Analyze relationship patterns
        all_relationships = defaultdict(int)
        relationship_qualities = defaultdict(list)
        fusion_strategy_usage = defaultdict(int)
        
        for result in successful_results:
            relationships = result.get('unified_relationships', {})
            
            for rel, rel_data in relationships.items():
                all_relationships[rel] += 1
                relationship_qualities[rel].append(rel_data.get('unified_confidence', 0.0))
                
                for strategy in rel_data.get('fusion_sources', []):
                    fusion_strategy_usage[strategy] += 1
        
        # Find common patterns
        common_relationships = sorted(all_relationships.items(), key=lambda x: x[1], reverse=True)[:20]
        high_quality_relationships = [
            (rel, np.mean(qualities)) for rel, qualities in relationship_qualities.items()
            if np.mean(qualities) > 0.8
        ]
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'concepts_analyzed': len(successful_results),
            'common_relationships': common_relationships,
            'high_quality_relationships': high_quality_relationships,
            'fusion_strategy_usage': dict(fusion_strategy_usage),
            'average_relationships_per_concept': np.mean([
                len(r.get('unified_relationships', {})) for r in successful_results
            ]),
            'average_fusion_quality': np.mean([
                r.get('fusion_quality', {}).get('fusion_quality_score', 0.0) 
                for r in successful_results
            ])
        }


if __name__ == "__main__":
    print("I1 Semantic Fusion Engine - Test Mode")
    print("This module is designed to be imported by I1_cross_pipeline_semantic_integrator.py")
    print("Run the main I1 system to see the fusion engine in action.")