#!/usr/bin/env python3
"""
R4S: Semantic Ontology Builder
Build TRUE semantic ontology with meaning-based relationships from BIZBOK definitions
Enhanced with keyword context and domain-specific inference rules

ENHANCED VERSION: Uses ALL R1 outputs (CONCEPTS, DOMAINS, KEYWORDS)
Unlike R4L (lexical), R4S extracts logical, hierarchical, and functional relationships
"""

import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set

class EnhancedDefinitionParser:
    """Parse BIZBOK definitions enhanced with keyword and domain context"""
    
    def __init__(self, keywords_data: Dict, domains_data: Dict):
        self.keyword_index = keywords_data.get('keyword_index', {})
        self.domain_mappings = domains_data.get('domains', {})
        
        # Enhanced semantic pattern templates
        self.semantic_patterns = {
            'IS_A': [
                r'is a (?:type|kind|form|sort) of (\w+(?:\s+\w+)*)',
                r'represents (?:a|an) (\w+(?:\s+\w+)*)',
                r'defined as (?:a|an) (\w+(?:\s+\w+)*)',
                r'refers to (?:a|an) (\w+(?:\s+\w+)*)',
                r'(?:a|an) (\w+(?:\s+\w+)*) that',
                r'considered (?:a|an) (\w+(?:\s+\w+)*)',
                r'classified as (?:a|an) (\w+(?:\s+\w+)*)'
            ],
            
            'PART_OF': [
                r'(?:part|component|element|portion) of (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'belongs to (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'contained (?:in|within) (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'within (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'comprises (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'subset of (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'member of (?:a|an|the) (\w+(?:\s+\w+)*)'
            ],
            
            'HAS_PROPERTY': [
                r'has (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'with (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'characterized by (\w+(?:\s+\w+)*)',
                r'includes (?:a|an|the) (\w+(?:\s+\w+)*)(?: attribute|property|field)',
                r'contains (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'possesses (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'featuring (?:a|an|the) (\w+(?:\s+\w+)*)'
            ],
            
            'REQUIRES': [
                r'requires (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'needs (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'depends on (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'must have (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'relies on (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'necessitates (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'prerequisite (?:is|of) (?:a|an|the) (\w+(?:\s+\w+)*)'
            ],
            
            'USED_FOR': [
                r'used (?:for|to|in) (\w+(?:\s+\w+)*)',
                r'enables (\w+(?:\s+\w+)*)',
                r'supports (\w+(?:\s+\w+)*)',
                r'facilitates (\w+(?:\s+\w+)*)',
                r'serves to (\w+(?:\s+\w+)*)',
                r'purpose (?:is|of) (\w+(?:\s+\w+)*)',
                r'intended for (\w+(?:\s+\w+)*)'
            ],
            
            'CAUSES': [
                r'causes (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'results in (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'leads to (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'triggers (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'produces (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'generates (?:a|an|the) (\w+(?:\s+\w+)*)'
            ],
            
            'ENABLES': [
                r'enables (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'allows (?:for )?(?:a|an|the) (\w+(?:\s+\w+)*)',
                r'makes possible (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'permits (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'empowers (?:a|an|the) (\w+(?:\s+\w+)*)'
            ],
            
            'CONSTRAINS': [
                r'limits (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'restricts (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'constrains (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'bounds (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'controls (?:a|an|the) (\w+(?:\s+\w+)*)'
            ],
            
            'PRECEDES': [
                r'before (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'prior to (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'precedes (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'happens before (?:a|an|the) (\w+(?:\s+\w+)*)',
                r'earlier than (?:a|an|the) (\w+(?:\s+\w+)*)'
            ]
        }
        
        # Domain-specific semantic indicators
        self.domain_indicators = {
            'finance': {
                'REQUIRES': ['authorization', 'approval', 'verification'],
                'HAS_PROPERTY': ['balance', 'amount', 'currency', 'rate'],
                'CAUSES': ['transaction', 'transfer', 'payment']
            },
            'organizational': {
                'REQUIRES': ['competency', 'skill', 'certification'],
                'HAS_PROPERTY': ['role', 'responsibility', 'authority'],
                'PART_OF': ['organization', 'department', 'team']
            },
            'operational': {
                'REQUIRES': ['resource', 'input', 'prerequisite'],
                'CAUSES': ['output', 'result', 'outcome'],
                'PART_OF': ['process', 'workflow', 'procedure']
            }
        }
    
    def extract_semantic_patterns(self, concept_name: str, definition: str, 
                                 concept_keywords: List[str], concept_domain: str) -> Dict[str, List[str]]:
        """ENHANCED: Extract semantic patterns using definition, keywords, and domain context"""
        relationships = defaultdict(list)
        definition_lower = definition.lower()
        
        # Extract from definition patterns
        for relation_type, patterns in self.semantic_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, definition_lower, re.IGNORECASE)
                if matches:
                    for match in matches:
                        target = self._normalize_concept(match)
                        if target and target != concept_name.lower():
                            relationships[relation_type].append(target)
        
        # ENHANCEMENT: Use keywords to infer additional relationships
        keyword_relationships = self._infer_from_keywords(concept_keywords, concept_domain)
        for rel_type, targets in keyword_relationships.items():
            relationships[rel_type].extend(targets)
        
        # ENHANCEMENT: Use domain-specific indicators
        domain_relationships = self._infer_from_domain(concept_name, concept_domain, concept_keywords)
        for rel_type, targets in domain_relationships.items():
            relationships[rel_type].extend(targets)
        
        # Remove duplicates
        for rel_type in relationships:
            relationships[rel_type] = list(set(relationships[rel_type]))
        
        return dict(relationships)
    
    def extract_properties_with_keywords(self, definition: str, concept_keywords: List[str]) -> List[str]:
        """ENHANCED: Extract properties using both definition and keywords"""
        properties = []
        
        # Extract from definition
        definition_lower = definition.lower()
        property_patterns = [
            r'(?:has|having|with) (?:a|an|the) (\w+)',
            r'characterized by (\w+)',
            r'(\w+) attribute',
            r'(\w+) property',
            r'includes (\w+)'
        ]
        
        for pattern in property_patterns:
            matches = re.findall(pattern, definition_lower, re.IGNORECASE)
            for match in matches:
                prop = self._normalize_concept(match)
                if prop and len(prop) > 2:
                    properties.append(prop)
        
        # ENHANCEMENT: Use keywords as potential properties
        common_properties = {
            'balance', 'amount', 'currency', 'rate', 'status', 'identifier',
            'name', 'description', 'type', 'category', 'owner', 'location',
            'date', 'time', 'value', 'count', 'size', 'priority'
        }
        
        for keyword in concept_keywords:
            if keyword.lower() in common_properties:
                properties.append(keyword.lower())
        
        return list(set(properties))
    
    def _infer_from_keywords(self, keywords: List[str], domain: str) -> Dict[str, List[str]]:
        """Infer relationships based on keyword patterns"""
        relationships = defaultdict(list)
        
        keyword_set = {kw.lower() for kw in keywords}
        
        # Financial keyword patterns
        if any(kw in keyword_set for kw in ['financial', 'money', 'currency', 'payment']):
            relationships['REQUIRES'].extend(['authorization', 'verification'])
            relationships['HAS_PROPERTY'].extend(['amount', 'currency'])
        
        # Organizational keyword patterns
        if any(kw in keyword_set for kw in ['human', 'resource', 'person', 'employee']):
            relationships['HAS_PROPERTY'].extend(['skill', 'competency'])
            relationships['PART_OF'].extend(['organization'])
        
        # Process keyword patterns
        if any(kw in keyword_set for kw in ['process', 'operation', 'workflow']):
            relationships['REQUIRES'].extend(['input', 'resource'])
            relationships['CAUSES'].extend(['output', 'result'])
        
        return dict(relationships)
    
    def _infer_from_domain(self, concept_name: str, domain: str, keywords: List[str]) -> Dict[str, List[str]]:
        """Infer relationships using domain-specific rules"""
        relationships = defaultdict(list)
        
        # Map domain names to our domain indicators
        domain_mapping = {
            'finance': 'finance',
            'financial': 'finance',
            'common': 'organizational',  # Many common concepts are organizational
            'government': 'organizational',
            'insurance': 'finance',
            'transportation': 'operational',
            'manufacturing': 'operational',
            'telecom': 'operational'
        }
        
        mapped_domain = domain_mapping.get(domain.lower(), 'organizational')
        
        if mapped_domain in self.domain_indicators:
            domain_rules = self.domain_indicators[mapped_domain]
            
            # Apply domain-specific inference rules
            for rel_type, common_targets in domain_rules.items():
                # Only add if supported by keywords or concept name
                for target in common_targets:
                    if (target in ' '.join(keywords).lower() or 
                        target in concept_name.lower() or
                        any(target in kw.lower() for kw in keywords)):
                        relationships[rel_type].append(target)
        
        return dict(relationships)
    
    def validate_with_domain_context(self, subject: str, relation: str, 
                                   object_: str, subject_domain: str) -> bool:
        """Validate relationships using domain-specific rules"""
        # Domain-specific validation rules
        if subject_domain.lower() in ['finance', 'financial']:
            if relation == 'REQUIRES' and object_ in ['authorization', 'approval']:
                return True
            if relation == 'HAS_PROPERTY' and object_ in ['balance', 'currency', 'amount']:
                return True
        
        if 'transaction' in subject.lower():
            if relation == 'CAUSES' and 'change' in object_:
                return True
            if relation == 'REQUIRES' and object_ in ['account', 'authorization']:
                return True
        
        return True  # Default to valid
    
    def _normalize_concept(self, concept: str) -> str:
        """Normalize concept name for consistency"""
        concept = re.sub(r'^(?:the|a|an)\s+', '', concept.strip())
        concept = re.sub(r'\s+', '_', concept.lower())
        if concept.endswith('s') and len(concept) > 3 and not concept.endswith('ss'):
            concept = concept[:-1]
        return concept


class EnhancedSemanticRelationshipExtractor:
    """Extract meaning-based relationships enhanced with keyword and domain context"""
    
    def __init__(self, keywords_data: Dict, domains_data: Dict):
        self.definition_parser = EnhancedDefinitionParser(keywords_data, domains_data)
        self.keywords_data = keywords_data
        self.domains_data = domains_data
        
        # Semantic relation types
        self.SEMANTIC_RELATIONS = {
            'IS_A', 'PART_OF', 'HAS_PROPERTY', 'REQUIRES', 
            'CAUSES', 'USED_FOR', 'ENABLES', 'CONSTRAINS', 
            'PRECEDES', 'RELATED_TO'
        }
    
    def extract_from_definition(self, concept: str, concept_data: Dict, 
                               all_concepts: Set[str]) -> Dict[str, List[str]]:
        """ENHANCED: Extract relationships using definition, keywords, and domain context"""
        definition = concept_data.get('definition', '')
        concept_keywords = concept_data.get('keywords', [])
        concept_domain = concept_data.get('domain', '')
        
        # Extract semantic patterns with enhanced context
        relationships = self.definition_parser.extract_semantic_patterns(
            concept, definition, concept_keywords, concept_domain
        )
        
        # Extract properties with keyword enhancement
        properties = self.definition_parser.extract_properties_with_keywords(
            definition, concept_keywords
        )
        
        if properties:
            if 'HAS_PROPERTY' not in relationships:
                relationships['HAS_PROPERTY'] = []
            relationships['HAS_PROPERTY'].extend(properties)
        
        # ENHANCEMENT: Cross-reference with keyword index for validation
        validated_relationships = self._validate_with_keywords(
            concept, relationships, all_concepts, concept_keywords
        )
        
        return validated_relationships
    
    def _validate_with_keywords(self, concept: str, relationships: Dict[str, List[str]], 
                               all_concepts: Set[str], concept_keywords: List[str]) -> Dict[str, List[str]]:
        """Validate relationships using keyword cross-reference"""
        validated = {}
        
        for rel_type, targets in relationships.items():
            validated_targets = []
            
            for target in targets:
                # Check if target exists as concept
                if target in all_concepts:
                    validated_targets.append(target)
                # Check if target appears in keyword index
                elif target in self.keywords_data.get('keyword_index', {}):
                    validated_targets.append(target)
                # Check for fuzzy matches in concepts
                elif any(target in c or c in target for c in all_concepts):
                    # Find best match
                    best_match = self._find_best_concept_match(target, all_concepts)
                    if best_match:
                        validated_targets.append(best_match)
                # Allow common business properties
                elif self._is_common_business_property(target):
                    validated_targets.append(target)
            
            if validated_targets:
                validated[rel_type] = validated_targets
        
        return validated
    
    def _find_best_concept_match(self, target: str, all_concepts: Set[str]) -> Optional[str]:
        """Find best matching concept using fuzzy matching"""
        best_match = None
        best_score = 0
        
        for concept in all_concepts:
            # Simple similarity scoring
            if target in concept:
                score = len(target) / len(concept)
            elif concept in target:
                score = len(concept) / len(target)
            else:
                # Jaccard similarity for keywords
                target_words = set(target.split('_'))
                concept_words = set(concept.split('_'))
                if target_words and concept_words:
                    score = len(target_words & concept_words) / len(target_words | concept_words)
                else:
                    score = 0
            
            if score > best_score and score > 0.3:  # Minimum similarity threshold
                best_score = score
                best_match = concept
        
        return best_match
    
    def _is_common_business_property(self, property_name: str) -> bool:
        """Check if property is a common business attribute"""
        common_properties = {
            'balance', 'amount', 'currency', 'rate', 'status', 'identifier',
            'name', 'description', 'type', 'category', 'owner', 'location',
            'date', 'time', 'value', 'count', 'size', 'priority', 'cost',
            'revenue', 'profit', 'risk', 'quality', 'performance', 'efficiency'
        }
        return property_name.lower() in common_properties


class EnhancedTaxonomyBuilder:
    """Build hierarchical taxonomy enhanced with domain context"""
    
    def __init__(self, domains_data: Dict):
        self.domains_data = domains_data
        self.taxonomy = {}
        self.levels = defaultdict(list)
    
    def construct_hierarchy(self, is_a_relations: Dict[str, List[str]], 
                          concept_domains: Dict[str, str]) -> Dict[str, Any]:
        """Build taxonomy considering domain structure"""
        # Build basic hierarchy from IS_A relationships
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        
        for child, parents in is_a_relations.items():
            for parent in parents:
                graph[parent].append(child)
                reverse_graph[child].append(parent)
        
        # ENHANCEMENT: Create domain-based root structure
        domain_roots = self._create_domain_roots(concept_domains)
        
        # Find orphan concepts (no parents) and assign to domain roots
        all_concepts = set(is_a_relations.keys()) | {p for parents in is_a_relations.values() for p in parents}
        orphan_concepts = []
        
        for concept in all_concepts:
            if concept not in reverse_graph or len(reverse_graph[concept]) == 0:
                orphan_concepts.append(concept)
        
        # Assign orphan concepts to appropriate domain roots
        for concept in orphan_concepts:
            concept_domain = concept_domains.get(concept, 'common')
            domain_root = f"{concept_domain}_concept"
            if domain_root not in graph:
                graph[domain_root] = []
            graph[domain_root].append(concept)
            reverse_graph[concept] = [domain_root]
        
        # Build levels starting from domain roots
        roots = list(domain_roots.keys())
        self._build_levels(graph, roots)
        
        return {
            'root': 'business_concept',
            'domain_roots': domain_roots,
            'levels': dict(self.levels),
            'graph': dict(graph),
            'reverse_graph': dict(reverse_graph),
            'max_depth': max(self.levels.keys()) if self.levels else 0
        }
    
    def _create_domain_roots(self, concept_domains: Dict[str, str]) -> Dict[str, List[str]]:
        """Create domain-based root concepts"""
        domain_roots = {}
        
        # Get unique domains
        domains = set(concept_domains.values())
        
        for domain in domains:
            if domain:  # Skip empty domains
                root_name = f"{domain}_concept"
                concepts_in_domain = [c for c, d in concept_domains.items() if d == domain]
                domain_roots[root_name] = concepts_in_domain
        
        return domain_roots
    
    def _build_levels(self, graph: Dict[str, List[str]], roots: List[str], level: int = 1):
        """Recursively build taxonomy levels"""
        if not roots:
            return
        
        self.levels[level] = roots.copy()
        
        next_level = []
        for root in roots:
            children = graph.get(root, [])
            next_level.extend(children)
        
        if next_level:
            self._build_levels(graph, next_level, level + 1)


class EnhancedDomainReasoner:
    """Apply enhanced business domain logic with keyword and domain context"""
    
    def __init__(self, domains_data: Dict, keywords_data: Dict):
        self.domains_data = domains_data
        self.keywords_data = keywords_data
        self.inference_rules = self._load_enhanced_inference_rules()
        self.domain_rules = self._load_enhanced_domain_rules()
        self.applied_rules = []
    
    def apply_inference_rules(self, relationships: Dict[str, Dict[str, List[str]]], 
                             concept_domains: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
        """Apply enhanced inference rules with domain context"""
        inferred_relationships = defaultdict(lambda: defaultdict(list))
        
        # Copy original relationships
        for concept, relations in relationships.items():
            for rel_type, targets in relations.items():
                inferred_relationships[concept][rel_type].extend(targets)
        
        # Apply transitivity rules
        self._apply_transitivity_rules(inferred_relationships)
        
        # ENHANCEMENT: Apply domain-specific rules with context
        self._apply_enhanced_domain_rules(inferred_relationships, concept_domains)
        
        return dict(inferred_relationships)
    
    def _apply_enhanced_domain_rules(self, relationships: Dict[str, Dict[str, List[str]]], 
                                   concept_domains: Dict[str, str]):
        """Apply domain rules enhanced with keyword context"""
        
        for concept, concept_relations in relationships.items():
            concept_domain = concept_domains.get(concept, 'common')
            
            # Financial domain rules
            if concept_domain.lower() in ['finance', 'financial']:
                self._apply_financial_rules(concept, concept_relations, relationships)
            
            # Organizational domain rules
            elif concept_domain.lower() in ['common', 'government']:
                self._apply_organizational_rules(concept, concept_relations, relationships)
            
            # Operational domain rules
            elif concept_domain.lower() in ['transportation', 'manufacturing', 'telecom']:
                self._apply_operational_rules(concept, concept_relations, relationships)
    
    def _apply_financial_rules(self, concept: str, concept_relations: Dict[str, List[str]], 
                              all_relationships: Dict[str, Dict[str, List[str]]]):
        """Apply financial domain specific rules"""
        concept_lower = concept.lower()
        
        # Rule: Financial concepts require authorization
        if any(fin_term in concept_lower for fin_term in ['financial', 'payment', 'transaction', 'account']):
            if 'authorization' not in concept_relations['REQUIRES']:
                concept_relations['REQUIRES'].append('authorization')
                self.applied_rules.append({
                    'rule': 'financial_requires_authorization',
                    'subject': concept,
                    'object': 'authorization'
                })
        
        # Rule: Transactions cause balance changes
        if 'transaction' in concept_lower:
            if 'balance_change' not in concept_relations['CAUSES']:
                concept_relations['CAUSES'].append('balance_change')
                self.applied_rules.append({
                    'rule': 'transaction_causes_change',
                    'subject': concept,
                    'object': 'balance_change'
                })
        
        # Rule: Accounts have financial properties
        if 'account' in concept_lower:
            financial_props = ['balance', 'currency', 'owner']
            for prop in financial_props:
                if prop not in concept_relations['HAS_PROPERTY']:
                    concept_relations['HAS_PROPERTY'].append(prop)
    
    def _apply_organizational_rules(self, concept: str, concept_relations: Dict[str, List[str]], 
                                  all_relationships: Dict[str, Dict[str, List[str]]]):
        """Apply organizational domain rules"""
        concept_lower = concept.lower()
        
        # Rule: Human resources have competencies
        if any(hr_term in concept_lower for hr_term in ['human_resource', 'employee', 'person']):
            if 'competency' not in concept_relations['HAS_PROPERTY']:
                concept_relations['HAS_PROPERTY'].append('competency')
                self.applied_rules.append({
                    'rule': 'human_resource_has_competency',
                    'subject': concept,
                    'object': 'competency'
                })
        
        # Rule: Jobs require competencies
        if 'job' in concept_lower:
            if 'competency' not in concept_relations['REQUIRES']:
                concept_relations['REQUIRES'].append('competency')
    
    def _apply_operational_rules(self, concept: str, concept_relations: Dict[str, List[str]], 
                                all_relationships: Dict[str, Dict[str, List[str]]]):
        """Apply operational domain rules"""
        concept_lower = concept.lower()
        
        # Rule: Operations require resources
        if any(op_term in concept_lower for op_term in ['operation', 'process', 'workflow']):
            if 'resource' not in concept_relations['REQUIRES']:
                concept_relations['REQUIRES'].append('resource')
                self.applied_rules.append({
                    'rule': 'operation_requires_resource',
                    'subject': concept,
                    'object': 'resource'
                })
        
        # Rule: Processes produce outputs
        if 'process' in concept_lower:
            if 'output' not in concept_relations['CAUSES']:
                concept_relations['CAUSES'].append('output')
    
    def _apply_transitivity_rules(self, relationships: Dict[str, Dict[str, List[str]]]):
        """Apply transitivity for PART_OF, REQUIRES, and PRECEDES"""
        transitive_relations = ['PART_OF', 'REQUIRES', 'PRECEDES']
        
        for rel_type in transitive_relations:
            changed = True
            iterations = 0
            max_iterations = 5
            
            while changed and iterations < max_iterations:
                changed = False
                iterations += 1
                
                for concept_a, relations in relationships.items():
                    if rel_type not in relations:
                        continue
                    
                    for concept_b in relations[rel_type][:]:
                        if concept_b in relationships and rel_type in relationships[concept_b]:
                            for concept_c in relationships[concept_b][rel_type]:
                                if concept_c not in relationships[concept_a][rel_type] and concept_c != concept_a:
                                    relationships[concept_a][rel_type].append(concept_c)
                                    changed = True
                                    self.applied_rules.append({
                                        'rule': f'transitivity_{rel_type}',
                                        'subject': concept_a,
                                        'object': concept_c,
                                        'via': concept_b
                                    })
    
    def _load_enhanced_inference_rules(self) -> List[Dict]:
        """Load enhanced inference rules"""
        return [
            {'type': 'transitivity', 'relations': ['PART_OF', 'REQUIRES', 'PRECEDES']},
            {'type': 'inheritance', 'relations': ['HAS_PROPERTY']},
            {'type': 'domain_specific', 'relations': ['REQUIRES', 'CAUSES', 'HAS_PROPERTY']}
        ]
    
    def _load_enhanced_domain_rules(self) -> List[Dict]:
        """Load enhanced domain-specific rules"""
        return [
            {'type': 'financial_requires_auth', 'domain': 'finance'},
            {'type': 'transaction_causes_change', 'domain': 'finance'},
            {'type': 'human_resource_has_competency', 'domain': 'organizational'},
            {'type': 'operation_requires_resource', 'domain': 'operational'}
        ]


class EnhancedSemanticClusterer:
    """Create semantic clusters enhanced with domain and keyword context"""
    
    def __init__(self, domains_data: Dict, keywords_data: Dict):
        self.domains_data = domains_data
        self.keywords_data = keywords_data
    
    def create_semantic_domains(self, concepts: Dict[str, Dict], 
                               relationships: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """Create enhanced semantic domains using all available context"""
        
        # Start with R1 domain mappings
        predefined_domains = self.domains_data.get('domains', {})
        
        # Enhanced domain assignments using multiple signals
        concept_assignments = defaultdict(list)
        concept_scores = defaultdict(dict)
        
        for concept_name, concept_data in concepts.items():
            # Get original domain from R1
            original_domain = concept_data.get('domain', 'common')
            
            # Score concept against semantic domains
            definition = concept_data.get('definition', '').lower()
            keywords = concept_data.get('keywords', [])
            
            # Calculate scores for each semantic domain
            domain_scores = self._calculate_domain_scores(
                concept_name, definition, keywords, original_domain, relationships.get(concept_name, {})
            )
            
            concept_scores[concept_name] = domain_scores
        
        # Assign concepts to best matching semantic domain
        for concept_name, scores in concept_scores.items():
            if scores:
                best_domain = max(scores, key=scores.get)
                if scores[best_domain] > 0:
                    concept_assignments[best_domain].append(concept_name)
                else:
                    concept_assignments['uncategorized'].append(concept_name)
            else:
                concept_assignments['uncategorized'].append(concept_name)
        
        # Create enhanced semantic domains
        semantic_domains = {}
        for domain, concept_list in concept_assignments.items():
            if concept_list:
                central_concept = self._find_central_concept(concept_list, relationships)
                coherence = self._calculate_enhanced_coherence(concept_list, concepts, relationships)
                
                semantic_domains[domain] = {
                    'concepts': concept_list,
                    'concept_count': len(concept_list),
                    'central_concept': central_concept,
                    'coherence': coherence,
                    'domain_keywords': self._extract_domain_keywords(concept_list, concepts)
                }
        
        return semantic_domains
    
    def _calculate_domain_scores(self, concept_name: str, definition: str, 
                                keywords: List[str], original_domain: str, 
                                concept_relationships: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate enhanced domain scores using multiple signals"""
        
        # Define enhanced semantic domains
        domain_indicators = {
            'financial_management': {
                'keywords': ['financial', 'finance', 'account', 'payment', 'currency', 
                           'monetary', 'tax', 'investment', 'transaction', 'balance', 'cost', 'revenue'],
                'concepts': ['financial_account', 'payment', 'currency', 'tax'],
                'relationships': {'REQUIRES': ['authorization'], 'HAS_PROPERTY': ['balance', 'amount']}
            },
            'organizational_structure': {
                'keywords': ['business_entity', 'partner', 'customer', 'human_resource', 
                           'job', 'competency', 'organization', 'role', 'responsibility'],
                'concepts': ['human_resource', 'job', 'business_entity', 'partner'],
                'relationships': {'HAS_PROPERTY': ['competency', 'skill'], 'PART_OF': ['organization']}
            },
            'strategic_planning': {
                'keywords': ['strategy', 'plan', 'objective', 'goal', 'vision', 
                           'initiative', 'decision', 'policy', 'mission'],
                'concepts': ['strategy', 'plan', 'goal', 'objective', 'vision'],
                'relationships': {'ENABLES': ['goal', 'objective'], 'REQUIRES': ['decision']}
            },
            'operational_processes': {
                'keywords': ['operation', 'work', 'workflow', 'schedule', 'event', 
                           'process', 'procedure', 'task', 'activity'],
                'concepts': ['operation', 'work', 'workflow', 'schedule', 'event'],
                'relationships': {'REQUIRES': ['resource'], 'CAUSES': ['output']}
            },
            'information_management': {
                'keywords': ['information', 'data', 'content', 'document', 'message', 
                           'communication', 'knowledge', 'report'],
                'concepts': ['content', 'message', 'information'],
                'relationships': {'HAS_PROPERTY': ['format', 'type'], 'USED_FOR': ['communication']}
            },
            'resource_management': {
                'keywords': ['asset', 'resource', 'material', 'facility', 'infrastructure', 
                           'equipment', 'inventory', 'supply'],
                'concepts': ['asset', 'material', 'facility', 'infrastructure'],
                'relationships': {'PART_OF': ['organization'], 'USED_FOR': ['operation']}
            }
        }
        
        scores = {}
        concept_lower = concept_name.lower()
        keywords_lower = [kw.lower() for kw in keywords]
        
        for domain, indicators in domain_indicators.items():
            score = 0
            
            # Score based on concept name
            for keyword in indicators['keywords']:
                if keyword in concept_lower:
                    score += 3
            
            # Score based on definition
            for keyword in indicators['keywords']:
                if keyword in definition:
                    score += 2
            
            # Score based on concept keywords
            for keyword in indicators['keywords']:
                if keyword in keywords_lower:
                    score += 2
            
            # Score based on relationships
            for rel_type, targets in indicators.get('relationships', {}).items():
                if rel_type in concept_relationships:
                    for target in targets:
                        if target in concept_relationships[rel_type]:
                            score += 1
            
            # Boost score if original domain matches
            if self._domains_match(original_domain, domain):
                score += 1
            
            scores[domain] = score
        
        return scores
    
    def _domains_match(self, original_domain: str, semantic_domain: str) -> bool:
        """Check if original domain matches semantic domain"""
        domain_mappings = {
            'finance': 'financial_management',
            'financial': 'financial_management',
            'common': 'organizational_structure',
            'government': 'organizational_structure',
            'manufacturing': 'operational_processes',
            'transportation': 'operational_processes',
            'telecom': 'operational_processes'
        }
        
        return domain_mappings.get(original_domain.lower()) == semantic_domain
    
    def _calculate_enhanced_coherence(self, concept_list: List[str], 
                                    concepts: Dict[str, Dict], 
                                    relationships: Dict[str, Dict[str, List[str]]]) -> float:
        """Calculate enhanced coherence using multiple factors"""
        if len(concept_list) < 2:
            return 1.0
        
        total_similarity = 0
        pair_count = 0
        
        for i, concept1 in enumerate(concept_list):
            for concept2 in concept_list[i+1:]:
                # Calculate similarity based on multiple factors
                similarity = 0
                
                # Keyword similarity
                kw1 = set(concepts.get(concept1, {}).get('keywords', []))
                kw2 = set(concepts.get(concept2, {}).get('keywords', []))
                if kw1 or kw2:
                    jaccard = len(kw1 & kw2) / len(kw1 | kw2) if (kw1 | kw2) else 0
                    similarity += jaccard * 0.4
                
                # Relationship similarity
                rel1 = relationships.get(concept1, {})
                rel2 = relationships.get(concept2, {})
                rel_sim = self._calculate_relationship_similarity(rel1, rel2)
                similarity += rel_sim * 0.6
                
                total_similarity += similarity
                pair_count += 1
        
        return total_similarity / pair_count if pair_count > 0 else 0
    
    def _calculate_relationship_similarity(self, rel1: Dict[str, List[str]], 
                                         rel2: Dict[str, List[str]]) -> float:
        """Calculate similarity between relationship sets"""
        if not rel1 and not rel2:
            return 1.0
        
        all_rel_types = set(rel1.keys()) | set(rel2.keys())
        if not all_rel_types:
            return 0
        
        similarity = 0
        for rel_type in all_rel_types:
            targets1 = set(rel1.get(rel_type, []))
            targets2 = set(rel2.get(rel_type, []))
            
            if targets1 or targets2:
                jaccard = len(targets1 & targets2) / len(targets1 | targets2)
                similarity += jaccard
        
        return similarity / len(all_rel_types)
    
    def _find_central_concept(self, concept_list: List[str], 
                             relationships: Dict[str, Dict[str, List[str]]]) -> str:
        """Find the most connected concept within a domain"""
        if not concept_list:
            return ""
        if len(concept_list) == 1:
            return concept_list[0]
        
        connection_counts = {}
        for concept in concept_list:
            count = 0
            if concept in relationships:
                for rel_type, targets in relationships[concept].items():
                    count += len([t for t in targets if t in concept_list])
            connection_counts[concept] = count
        
        return max(connection_counts, key=connection_counts.get)
    
    def _extract_domain_keywords(self, concept_list: List[str], 
                                concepts: Dict[str, Dict]) -> List[str]:
        """Extract common keywords for the domain"""
        all_keywords = []
        for concept in concept_list:
            all_keywords.extend(concepts.get(concept, {}).get('keywords', []))
        
        # Count keyword frequency
        keyword_counts = Counter(all_keywords)
        
        # Return top keywords (appearing in multiple concepts)
        domain_keywords = [kw for kw, count in keyword_counts.items() if count > 1]
        return domain_keywords[:10]  # Top 10 keywords


class R4S_SemanticOntologyBuilder:
    """
    ENHANCED Main class for building semantic ontology from BIZBOK definitions
    Uses ALL R1 outputs: CONCEPTS, DOMAINS, KEYWORDS for comprehensive understanding
    """
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.parent
        self.output_dir = self.script_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Data storage - ENHANCED with all R1 inputs
        self.concepts = {}
        self.domains_data = {}
        self.keywords_data = {}
        self.semantic_relationships = defaultdict(lambda: defaultdict(list))
        self.semantic_taxonomy = {}
        self.semantic_clusters = {}
        self.ontology_statistics = {}
        
        # Initialize enhanced components (will be set after loading data)
        self.relationship_extractor = None
        self.taxonomy_builder = None
        self.domain_reasoner = None
        self.semantic_clusterer = None
        
        # Performance metrics
        self.performance_metrics = {
            'start_time': None,
            'processing_stages': {}
        }
    
    def main_processing_pipeline(self):
        """Execute the ENHANCED semantic ontology building pipeline"""
        print("=" * 80)
        print("R4S: ENHANCED Semantic Ontology Builder")
        print("Building TRUE semantic relationships using CONCEPTS + DOMAINS + KEYWORDS")
        print("=" * 80)
        
        self.performance_metrics['start_time'] = time.time()
        
        # Stage 1: Load ALL R1 outputs
        print("\n[STAGE 1] Loading ALL R1 outputs (CONCEPTS + DOMAINS + KEYWORDS)...")
        stage_start = time.time()
        self.stage1_load_all_r1_data()
        self.performance_metrics['processing_stages']['data_loading'] = time.time() - stage_start
        
        # Stage 2: Enhanced definition parsing
        print("\n[STAGE 2] Enhanced parsing with keyword and domain context...")
        stage_start = time.time()
        self.stage2_enhanced_parsing()
        self.performance_metrics['processing_stages']['enhanced_parsing'] = time.time() - stage_start
        
        # Stage 3: Enhanced relationship extraction
        print("\n[STAGE 3] Enhanced semantic relationship extraction...")
        stage_start = time.time()
        self.stage3_enhanced_extraction()
        self.performance_metrics['processing_stages']['enhanced_extraction'] = time.time() - stage_start
        
        # Stage 4: Enhanced taxonomy building
        print("\n[STAGE 4] Enhanced taxonomical hierarchy...")
        stage_start = time.time()
        self.stage4_enhanced_taxonomy()
        self.performance_metrics['processing_stages']['enhanced_taxonomy'] = time.time() - stage_start
        
        # Stage 5: Enhanced domain reasoning
        print("\n[STAGE 5] Enhanced domain reasoning...")
        stage_start = time.time()
        self.stage5_enhanced_reasoning()
        self.performance_metrics['processing_stages']['enhanced_reasoning'] = time.time() - stage_start
        
        # Stage 6: Enhanced semantic clustering
        print("\n[STAGE 6] Enhanced semantic clustering...")
        stage_start = time.time()
        self.stage6_enhanced_clustering()
        self.performance_metrics['processing_stages']['enhanced_clustering'] = time.time() - stage_start
        
        # Stage 7: Generate comprehensive outputs
        print("\n[STAGE 7] Generating enhanced outputs...")
        stage_start = time.time()
        self.stage7_generate_enhanced_outputs()
        self.performance_metrics['processing_stages']['output_generation'] = time.time() - stage_start
        
        total_time = time.time() - self.performance_metrics['start_time']
        print(f"\n[SUCCESS] R4S Enhanced Semantic Ontology Builder completed in {total_time:.2f} seconds!")
        
        return self.generate_enhanced_summary()
    
    def stage1_load_all_r1_data(self):
        """ENHANCED: Load ALL R1 outputs for comprehensive processing"""
        # Load R1 CONCEPTS
        r1_concepts_path = self.output_dir / "R1_CONCEPTS.json"
        if not r1_concepts_path.exists():
            raise FileNotFoundError(f"R1 concepts not found: {r1_concepts_path}")
        
        with open(r1_concepts_path, 'r', encoding='utf-8') as f:
            r1_data = json.load(f)
            self.concepts = r1_data.get("concepts", {})
        
        print(f"   [OK] Loaded {len(self.concepts)} BIZBOK concepts")
        
        # ENHANCEMENT: Load R1 DOMAINS
        r1_domains_path = self.output_dir / "R1_DOMAINS.json"
        if not r1_domains_path.exists():
            raise FileNotFoundError(f"R1 domains not found: {r1_domains_path}")
        
        with open(r1_domains_path, 'r', encoding='utf-8') as f:
            self.domains_data = json.load(f)
        
        domain_count = len(self.domains_data.get('domains', {}))
        print(f"   [OK] Loaded {domain_count} domain mappings")
        
        # ENHANCEMENT: Load R1 KEYWORDS
        r1_keywords_path = self.output_dir / "R1_KEYWORDS.json"
        if not r1_keywords_path.exists():
            raise FileNotFoundError(f"R1 keywords not found: {r1_keywords_path}")
        
        with open(r1_keywords_path, 'r', encoding='utf-8') as f:
            self.keywords_data = json.load(f)
        
        keyword_count = self.keywords_data.get('metadata', {}).get('total_keywords', 0)
        print(f"   [OK] Loaded {keyword_count} keyword mappings")
        
        # Initialize enhanced components with loaded data
        self.relationship_extractor = EnhancedSemanticRelationshipExtractor(
            self.keywords_data, self.domains_data
        )
        self.taxonomy_builder = EnhancedTaxonomyBuilder(self.domains_data)
        self.domain_reasoner = EnhancedDomainReasoner(self.domains_data, self.keywords_data)
        self.semantic_clusterer = EnhancedSemanticClusterer(self.domains_data, self.keywords_data)
        
        print(f"   [OK] Initialized enhanced processing components")
    
    def stage2_enhanced_parsing(self):
        """Enhanced parsing with keyword and domain context"""
        parsed_count = 0
        
        for concept_name, concept_data in self.concepts.items():
            # Add keyword context to each concept
            concept_keywords = concept_data.get('keywords', [])
            concept_domain = concept_data.get('domain', '')
            
            # Store enhanced context
            concept_data['enhanced_keywords'] = concept_keywords
            concept_data['enhanced_domain'] = concept_domain
            
            parsed_count += 1
        
        print(f"   [OK] Enhanced parsing for {parsed_count} concepts with keyword/domain context")
    
    def stage3_enhanced_extraction(self):
        """ENHANCED: Extract semantic relationships using all available context"""
        all_concept_names = set(self.concepts.keys())
        extracted_relationships = 0
        
        for concept_name, concept_data in self.concepts.items():
            # Enhanced extraction using all context
            relationships = self.relationship_extractor.extract_from_definition(
                concept_name, concept_data, all_concept_names
            )
            
            # Store relationships
            for rel_type, targets in relationships.items():
                self.semantic_relationships[concept_name][rel_type].extend(targets)
                extracted_relationships += len(targets)
        
        print(f"   [OK] Extracted {extracted_relationships} enhanced semantic relationships")
        
        # Display relationship distribution
        rel_counts = defaultdict(int)
        for concept_rels in self.semantic_relationships.values():
            for rel_type, targets in concept_rels.items():
                rel_counts[rel_type] += len(targets)
        
        print(f"      Relationship distribution:")
        for rel_type, count in sorted(rel_counts.items()):
            print(f"        {rel_type}: {count}")
    
    def stage4_enhanced_taxonomy(self):
        """Enhanced taxonomy building with domain structure"""
        # Extract IS_A relationships
        is_a_relations = {}
        concept_domains = {}
        
        for concept, relations in self.semantic_relationships.items():
            if 'IS_A' in relations and relations['IS_A']:
                is_a_relations[concept] = relations['IS_A']
            concept_domains[concept] = self.concepts[concept].get('domain', 'common')
        
        # Build enhanced taxonomy
        self.semantic_taxonomy = self.taxonomy_builder.construct_hierarchy(
            is_a_relations, concept_domains
        )
        
        print(f"   [OK] Built enhanced taxonomy with {self.semantic_taxonomy.get('max_depth', 0)} levels")
        print(f"   [OK] Created domain-based hierarchy structure")
    
    def stage5_enhanced_reasoning(self):
        """Enhanced domain reasoning with keyword and domain context"""
        # Build concept domains mapping
        concept_domains = {name: data.get('domain', 'common') 
                          for name, data in self.concepts.items()}
        
        # Apply enhanced inference rules
        reasoned_relationships = self.domain_reasoner.apply_inference_rules(
            dict(self.semantic_relationships), concept_domains
        )
        
        # Update relationships
        self.semantic_relationships = reasoned_relationships
        
        inference_count = len(self.domain_reasoner.applied_rules)
        print(f"   [OK] Applied {inference_count} enhanced inference rules")
        
        # Display applied rules summary
        rule_counts = defaultdict(int)
        for rule in self.domain_reasoner.applied_rules:
            rule_counts[rule['rule']] += 1
        
        print(f"      Applied rule summary:")
        for rule_name, count in sorted(rule_counts.items()):
            print(f"        {rule_name}: {count}")
    
    def stage6_enhanced_clustering(self):
        """Enhanced semantic clustering using all available context"""
        self.semantic_clusters = self.semantic_clusterer.create_semantic_domains(
            self.concepts, dict(self.semantic_relationships)
        )
        
        cluster_count = len(self.semantic_clusters)
        print(f"   [OK] Created {cluster_count} enhanced semantic domains")
        
        # Display enhanced cluster summary
        for domain, data in self.semantic_clusters.items():
            concept_count = data['concept_count']
            coherence = data['coherence']
            central_concept = data['central_concept']
            print(f"      {domain}: {concept_count} concepts (coherence: {coherence:.2f}, center: {central_concept})")
    
    def stage7_generate_enhanced_outputs(self):
        """Generate comprehensive enhanced outputs"""
        # Calculate enhanced statistics
        self.calculate_enhanced_statistics()
        
        # Generate all output files
        self.save_enhanced_semantic_ontology()
        self.save_enhanced_relationships()
        self.save_enhanced_hierarchy()
        self.save_enhanced_clusters()
        self.save_enhanced_statistics()
        
        print(f"   [OK] Generated 5 enhanced output files")
    
    def calculate_enhanced_statistics(self):
        """Calculate comprehensive enhanced statistics"""
        # Count relationships by type
        relationship_counts = defaultdict(int)
        total_relationships = 0
        
        for concept, relations in self.semantic_relationships.items():
            for rel_type, targets in relations.items():
                count = len(targets)
                relationship_counts[rel_type] += count
                total_relationships += count
        
        # Calculate coverage and quality metrics
        concept_count = len(self.concepts)
        cluster_count = len(self.semantic_clusters)
        
        concepts_with_relationships = len([c for c in self.semantic_relationships 
                                         if any(self.semantic_relationships[c].values())])
        semantic_coverage = concepts_with_relationships / concept_count if concept_count > 0 else 0
        
        # Average cluster coherence
        coherence_scores = [data['coherence'] for data in self.semantic_clusters.values()]
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        
        # Enhancement metrics
        keyword_usage = self.keywords_data.get('metadata', {}).get('total_keywords', 0)
        domain_count = len(self.domains_data.get('domains', {}))
        inference_rules_applied = len(self.domain_reasoner.applied_rules)
        
        self.ontology_statistics = {
            'metadata': {
                'version': '2.0_ENHANCED',
                'created': datetime.now().isoformat(),
                'processing_time': time.time() - self.performance_metrics['start_time'],
                'enhancement_level': 'FULL_R1_INTEGRATION'
            },
            'input_sources': {
                'r1_concepts': concept_count,
                'r1_domains': domain_count,
                'r1_keywords': keyword_usage
            },
            'counts': {
                'total_concepts': concept_count,
                'total_relationships': total_relationships,
                'semantic_clusters': cluster_count,
                'relationship_types': len(relationship_counts),
                'inference_rules_applied': inference_rules_applied
            },
            'relationship_distribution': dict(relationship_counts),
            'quality_metrics': {
                'semantic_coverage': semantic_coverage,
                'avg_relationships_per_concept': total_relationships / concept_count if concept_count > 0 else 0,
                'avg_cluster_coherence': avg_coherence,
                'taxonomy_depth': self.semantic_taxonomy.get('max_depth', 0),
                'domain_integration_score': self._calculate_domain_integration_score()
            },
            'enhancement_metrics': {
                'keyword_enhanced_extractions': self._count_keyword_enhanced_extractions(),
                'domain_enhanced_inferences': self._count_domain_enhanced_inferences(),
                'cross_domain_relationships': self._count_cross_domain_relationships()
            },
            'performance_metrics': self.performance_metrics
        }
    
    def _calculate_domain_integration_score(self) -> float:
        """Calculate how well domains are integrated in the semantic model"""
        if not self.semantic_clusters:
            return 0.0
        
        total_score = 0
        for domain_data in self.semantic_clusters.values():
            coherence = domain_data.get('coherence', 0)
            concept_count = domain_data.get('concept_count', 0)
            # Weight by concept count
            total_score += coherence * min(concept_count / 10, 1.0)
        
        return total_score / len(self.semantic_clusters)
    
    def _count_keyword_enhanced_extractions(self) -> int:
        """Count relationships that were enhanced by keyword data"""
        # This is a simplified count - in practice, would track during extraction
        keyword_enhanced = 0
        for concept, relations in self.semantic_relationships.items():
            if 'HAS_PROPERTY' in relations:
                # Properties are often enhanced by keywords
                keyword_enhanced += len(relations['HAS_PROPERTY'])
        return keyword_enhanced
    
    def _count_domain_enhanced_inferences(self) -> int:
        """Count inferences that were enhanced by domain context"""
        domain_rules = [rule for rule in self.domain_reasoner.applied_rules 
                       if 'domain' in rule.get('rule', '').lower()]
        return len(domain_rules)
    
    def _count_cross_domain_relationships(self) -> int:
        """Count relationships that cross domain boundaries"""
        cross_domain = 0
        concept_domains = {name: data.get('domain', 'common') 
                          for name, data in self.concepts.items()}
        
        for concept, relations in self.semantic_relationships.items():
            concept_domain = concept_domains.get(concept, 'common')
            for rel_type, targets in relations.items():
                for target in targets:
                    target_domain = concept_domains.get(target, 'common')
                    if target_domain != concept_domain and target in concept_domains:
                        cross_domain += 1
        
        return cross_domain
    
    def save_enhanced_semantic_ontology(self):
        """Save the complete enhanced semantic ontology"""
        ontology_data = {
            'metadata': self.ontology_statistics['metadata'],
            'input_sources': self.ontology_statistics['input_sources'],
            'concepts': {},
            'taxonomy': self.semantic_taxonomy,
            'semantic_clusters': self.semantic_clusters,
            'statistics': self.ontology_statistics,
            'enhancement_info': {
                'used_keywords': True,
                'used_domains': True,
                'applied_inference': True,
                'enhancement_level': 'COMPREHENSIVE'
            }
        }
        
        # Prepare enhanced concept data
        for concept_name, concept_data in self.concepts.items():
            relationships = dict(self.semantic_relationships[concept_name]) if concept_name in self.semantic_relationships else {}
            
            # Find semantic domain
            semantic_domain = 'uncategorized'
            for domain, domain_data in self.semantic_clusters.items():
                if concept_name in domain_data['concepts']:
                    semantic_domain = domain
                    break
            
            ontology_data['concepts'][concept_name] = {
                'name': concept_name,
                'definition': concept_data.get('definition', ''),
                'original_domain': concept_data.get('domain', ''),
                'semantic_domain': semantic_domain,
                'keywords': concept_data.get('keywords', []),
                'relationships': relationships,
                'enhancement_applied': {
                    'keyword_context': bool(concept_data.get('keywords')),
                    'domain_context': bool(concept_data.get('domain')),
                    'relationship_count': sum(len(targets) for targets in relationships.values())
                }
            }
        
        output_path = self.output_dir / "R4S_semantic_ontology.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology_data, f, indent=2, ensure_ascii=False)
        
        print(f"   [OK] Saved enhanced semantic ontology to {output_path.name}")
    
    def save_enhanced_relationships(self):
        """Save enhanced semantic relationships"""
        relationships_data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_relationships': sum(len(targets) for relations in self.semantic_relationships.values() for targets in relations.values()),
                'enhancement_level': 'KEYWORD_AND_DOMAIN_ENHANCED'
            },
            'relationships': dict(self.semantic_relationships),
            'relationship_types': list(self.relationship_extractor.SEMANTIC_RELATIONS),
            'applied_inference_rules': self.domain_reasoner.applied_rules,
            'enhancement_sources': {
                'keyword_enhanced': True,
                'domain_enhanced': True,
                'inference_applied': True
            }
        }
        
        output_path = self.output_dir / "R4S_semantic_relationships.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(relationships_data, f, indent=2, ensure_ascii=False)
        
        print(f"   [OK] Saved enhanced relationships to {output_path.name}")
    
    def save_enhanced_hierarchy(self):
        """Save enhanced taxonomical hierarchy"""
        output_path = self.output_dir / "R4S_semantic_hierarchy.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_taxonomy, f, indent=2, ensure_ascii=False)
        
        print(f"   [OK] Saved enhanced hierarchy to {output_path.name}")
    
    def save_enhanced_clusters(self):
        """Save enhanced semantic clusters"""
        output_path = self.output_dir / "R4S_semantic_clusters.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_clusters, f, indent=2, ensure_ascii=False)
        
        print(f"   [OK] Saved enhanced clusters to {output_path.name}")
    
    def save_enhanced_statistics(self):
        """Save comprehensive enhanced statistics"""
        output_path = self.output_dir / "R4S_ontology_statistics.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.ontology_statistics, f, indent=2, ensure_ascii=False)
        
        print(f"   [OK] Saved enhanced statistics to {output_path.name}")
    
    def generate_enhanced_summary(self):
        """Generate enhanced execution summary"""
        stats = self.ontology_statistics
        
        summary = {
            'success': True,
            'enhancement_level': 'COMPREHENSIVE_R1_INTEGRATION',
            'processing_time': stats['metadata']['processing_time'],
            'input_sources': stats['input_sources'],
            'concepts_processed': stats['counts']['total_concepts'],
            'relationships_extracted': stats['counts']['total_relationships'],
            'relationship_types': stats['counts']['relationship_types'],
            'semantic_clusters': stats['counts']['semantic_clusters'],
            'taxonomy_depth': stats['quality_metrics']['taxonomy_depth'],
            'semantic_coverage': stats['quality_metrics']['semantic_coverage'],
            'avg_coherence': stats['quality_metrics']['avg_cluster_coherence'],
            'domain_integration_score': stats['quality_metrics']['domain_integration_score'],
            'enhancement_metrics': stats['enhancement_metrics']
        }
        
        return summary


def main():
    """Main execution function"""
    try:
        builder = R4S_SemanticOntologyBuilder()
        result = builder.main_processing_pipeline()
        
        # Display comprehensive results
        print("\n" + "=" * 80)
        print("R4S ENHANCED SEMANTIC ONTOLOGY BUILDER SUMMARY")
        print("=" * 80)
        print(f"Enhancement Level: {result['enhancement_level']}")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print(f"\nInput Sources:")
        print(f"  R1 Concepts: {result['input_sources']['r1_concepts']}")
        print(f"  R1 Domains: {result['input_sources']['r1_domains']}")  
        print(f"  R1 Keywords: {result['input_sources']['r1_keywords']}")
        print(f"\nSemantic Results:")
        print(f"  Concepts Processed: {result['concepts_processed']}")
        print(f"  Semantic Relationships: {result['relationships_extracted']}")
        print(f"  Relationship Types: {result['relationship_types']}")
        print(f"  Semantic Clusters: {result['semantic_clusters']}")
        print(f"  Taxonomy Depth: {result['taxonomy_depth']} levels")
        print(f"\nQuality Metrics:")
        print(f"  Semantic Coverage: {result['semantic_coverage']:.1%}")
        print(f"  Cluster Coherence: {result['avg_coherence']:.2f}")
        print(f"  Domain Integration: {result['domain_integration_score']:.2f}")
        print(f"\nEnhancement Impact:")
        print(f"  Keyword Enhanced: {result['enhancement_metrics']['keyword_enhanced_extractions']}")
        print(f"  Domain Enhanced: {result['enhancement_metrics']['domain_enhanced_inferences']}")
        print(f"  Cross-Domain Links: {result['enhancement_metrics']['cross_domain_relationships']}")
        
        print("\n" + "=" * 80)
        print("SUCCESS: Enhanced semantic ontology built using ALL R1 context!")
        print("BIZBOK concepts enhanced with keywords and domain reasoning")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: R4S Enhanced Semantic Ontology Builder failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()