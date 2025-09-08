#!/usr/bin/env python3
"""
A3 Chunking Strategies Module
Provides all chunking strategies for the A-Pipeline
"""

from .base_strategy import BaseChunkingStrategy, ConceptChunk
from .semantic_sentence import SemanticSentenceStrategy
from .paragraph_aware import ParagraphAwareStrategy
from .document_structure import DocumentStructureStrategy
from .adaptive_chunking import AdaptiveChunkingStrategy
from .concept_aware import ConceptAwareStrategy
from .contextual_overlap import ContextualOverlapStrategy
from .quality_based import QualityBasedStrategy

# Strategy registry for easy access
STRATEGIES = {
    'semantic_sentence': SemanticSentenceStrategy,
    'paragraph_aware': ParagraphAwareStrategy,
    'document_structure': DocumentStructureStrategy,
    'adaptive': AdaptiveChunkingStrategy,
    'concept_aware': ConceptAwareStrategy,
    'contextual_overlap': ContextualOverlapStrategy,
    'quality_based': QualityBasedStrategy
}

def get_strategy(strategy_name: str, **kwargs) -> BaseChunkingStrategy:
    """
    Factory function to get a chunking strategy by name
    
    Args:
        strategy_name: Name of the strategy
        **kwargs: Strategy-specific configuration parameters
        
    Returns:
        Instance of the requested strategy
        
    Raises:
        ValueError: If strategy_name is not recognized
    """
    if strategy_name not in STRATEGIES:
        available = ', '.join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available}")
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(**kwargs)

def list_strategies() -> list:
    """
    Get list of available strategy names
    
    Returns:
        List of strategy names
    """
    return list(STRATEGIES.keys())

__all__ = [
    'BaseChunkingStrategy',
    'ConceptChunk',
    'SemanticSentenceStrategy',
    'ParagraphAwareStrategy',
    'DocumentStructureStrategy',
    'AdaptiveChunkingStrategy',
    'ConceptAwareStrategy',
    'ContextualOverlapStrategy',
    'QualityBasedStrategy',
    'STRATEGIES',
    'get_strategy',
    'list_strategies'
]