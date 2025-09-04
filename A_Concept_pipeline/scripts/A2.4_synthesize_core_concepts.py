#!/usr/bin/env python3
"""
A2.4: Synthesize Core Concepts
Identifies and synthesizes the most important core concepts from intra-document business themes
with explicit keyword_id tracking and cross-document concept aggregation
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import math
import re

def calculate_concept_importance(theme_group, total_docs):
    """
    Calculate importance score for a business theme aggregated across documents
    
    Args:
        theme_group: Aggregated theme with cross-document instances
        total_docs: Total number of documents
        
    Returns:
        float: Importance score
    """
    # Factors for business theme importance:
    # 1. Cross-document coverage (how many documents have this theme)
    # 2. Aggregate keyword scores (strength of business concepts)
    # 3. Keyword diversity (richness of business terminology)
    # 4. Business semantic coherence
    
    doc_coverage = len(theme_group["document_instances"]) / total_docs
    
    # Calculate average keyword strength across all instances
    total_score = sum(instance["avg_score"] for instance in theme_group["document_instances"])
    avg_keyword_strength = total_score / len(theme_group["document_instances"])
    
    # Keyword diversity bonus (more unique business terms = higher importance)
    unique_keywords = len(theme_group["all_keywords"])
    diversity_factor = min(unique_keywords / 10, 1.0)
    
    # Business theme bonus for clear semantic meaning
    theme_bonus = 1.3 if any(business_term in theme_group["canonical_name"].lower() 
                           for business_term in ['income', 'revenue', 'contract', 'balance', 'inventory', 
                                                'customer', 'operation', 'process', 'management']) else 1.0
    
    importance = (
        doc_coverage * 0.4 +           # Cross-document presence
        min(avg_keyword_strength, 1.0) * 0.3 +  # Business concept strength
        diversity_factor * 0.2 +       # Keyword richness
        min(len(theme_group["document_instances"]) / 3, 1.0) * 0.1  # Instance frequency
    ) * theme_bonus
    
    return min(1.0, importance)

def aggregate_themes_across_documents(documents):
    """
    Aggregate similar business themes across documents to identify cross-document concepts
    
    Args:
        documents: List of documents with keyword_clusters
        
    Returns:
        dict: Aggregated theme groups by canonical name
    """
    theme_aggregation = {}
    
    for doc in documents:
        doc_id = doc.get("doc_id", "unknown")
        
        for cluster in doc.get("keyword_clusters", []):
            theme_name = cluster.get("theme_name", "Unknown Theme")
            
            # Create canonical theme name for aggregation (normalize variations)
            canonical_name = normalize_theme_name(theme_name)
            
            if canonical_name not in theme_aggregation:
                theme_aggregation[canonical_name] = {
                    "canonical_name": canonical_name,
                    "document_instances": [],
                    "all_keywords": set(),
                    "all_keyword_ids": set(),
                    "theme_variants": set()
                }
            
            # Extract keyword information with keyword_id tracking
            instance_keywords = []
            instance_keyword_ids = []
            total_score = 0
            
            for kw in cluster.get("keywords", []):
                instance_keywords.append(kw.get("term", ""))
                instance_keyword_ids.append(kw.get("keyword_id", ""))
                total_score += kw.get("score", 0)
                
                theme_aggregation[canonical_name]["all_keywords"].add(kw.get("term", ""))
                theme_aggregation[canonical_name]["all_keyword_ids"].add(kw.get("keyword_id", ""))
            
            avg_score = total_score / len(cluster.get("keywords", [])) if cluster.get("keywords") else 0
            
            # Add document instance
            theme_aggregation[canonical_name]["document_instances"].append({
                "doc_id": doc_id,
                "cluster_id": cluster.get("cluster_id"),
                "original_theme_name": theme_name,
                "keywords": instance_keywords,
                "keyword_ids": instance_keyword_ids,
                "avg_score": avg_score,
                "keyword_count": len(instance_keywords)
            })
            
            theme_aggregation[canonical_name]["theme_variants"].add(theme_name)
    
    return theme_aggregation

def normalize_theme_name(theme_name):
    """
    Normalize theme names for cross-document aggregation
    
    Args:
        theme_name: Original theme name
        
    Returns:
        str: Normalized canonical name
    """
    # Convert to lower case and extract key business terms
    name_lower = theme_name.lower()
    
    # Define business concept mappings for normalization
    business_mappings = {
        'deferred income': ['deferred', 'income'],
        'revenue recognition': ['revenue', 'recognition'],
        'contract balances': ['contract', 'balance'],
        'inventory management': ['inventory', 'management'],
        'customer operations': ['customer', 'operation'],
        'financial reporting': ['financial', 'report']
    }
    
    # Find best match
    for canonical, terms in business_mappings.items():
        if all(term in name_lower for term in terms):
            return canonical
    
    # Fallback: use first two significant words
    words = [w for w in name_lower.replace('&', ' ').split() if len(w) > 2]
    return ' '.join(words[:2]) if len(words) >= 2 else theme_name.lower()

def identify_core_concepts(theme_aggregation, total_docs, top_k=10):
    """
    Identify core business concepts from aggregated themes
    
    Args:
        theme_aggregation: Aggregated themes across documents
        total_docs: Total number of documents
        top_k: Number of core concepts to identify
        
    Returns:
        list: Core concepts with metadata and keyword_id tracking
    """
    core_concepts = []
    
    for theme_group in theme_aggregation.values():
        importance = calculate_concept_importance(theme_group, total_docs)
        
        # Get most representative keywords
        keyword_frequency = {}
        for instance in theme_group["document_instances"]:
            for kw in instance["keywords"]:
                keyword_frequency[kw] = keyword_frequency.get(kw, 0) + 1
        
        top_keywords = sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate enhanced concept metadata
        business_category = categorize_business_concept(theme_group["canonical_name"])
        concept_type_info = generate_detailed_concept_type(
            theme_group["canonical_name"], 
            [kw[0] for kw in top_keywords], 
            business_category
        )
        concept_definition = generate_concept_definition(
            theme_group["canonical_name"],
            [kw[0] for kw in top_keywords],
            list(theme_group["theme_variants"]),
            business_category,
            concept_type_info
        )
        
        core_concept = {
            "concept_id": f"core_{len(core_concepts) + 1}",
            "canonical_name": theme_group["canonical_name"],
            "theme_variants": list(theme_group["theme_variants"]),
            "importance_score": importance,
            "document_count": len(theme_group["document_instances"]),
            "coverage_ratio": len(theme_group["document_instances"]) / total_docs,
            "primary_keywords": [kw[0] for kw in top_keywords],
            "keyword_frequencies": dict(top_keywords),
            "related_documents": [inst["doc_id"] for inst in theme_group["document_instances"]],
            "all_keyword_ids": list(theme_group["all_keyword_ids"]),
            "unique_keyword_count": len(theme_group["all_keywords"]),
            "total_instances": len(theme_group["document_instances"]),
            "document_instances": theme_group["document_instances"],
            
            # Enhanced metadata
            "business_category": business_category,
            "concept_definition": concept_definition,
            "concept_type": concept_type_info
        }
        
        core_concepts.append(core_concept)
    
    # Sort by importance and take top k
    core_concepts.sort(key=lambda x: x["importance_score"], reverse=True)
    return core_concepts[:top_k]

def create_concept_hierarchy(core_concepts):
    """
    Create hierarchical structure of business concepts based on semantic similarity
    
    Args:
        core_concepts: List of core concepts
        
    Returns:
        dict: Hierarchical concept structure by business category
    """
    # Group concepts by business category
    business_categories = {
        "Financial Concepts": [],
        "Operational Concepts": [], 
        "Customer & Contract Concepts": [],
        "General Business Concepts": []
    }
    
    for concept in core_concepts:
        canonical_name = concept["canonical_name"]
        category = categorize_business_concept(canonical_name)
        business_categories[category].append(concept)
    
    # Calculate category-level statistics
    category_hierarchy = {}
    for category, concepts in business_categories.items():
        if concepts:  # Only include categories with concepts
            category_hierarchy[category] = {
                "category": category,
                "concepts": concepts,
                "concept_count": len(concepts),
                "total_importance": sum(c["importance_score"] for c in concepts),
                "avg_importance": sum(c["importance_score"] for c in concepts) / len(concepts),
                "document_coverage": len(set().union(*[c["related_documents"] for c in concepts])),
                "unique_keywords": len(set().union(*[c["all_keyword_ids"] for c in concepts]))
            }
    
    return category_hierarchy

def categorize_business_concept(canonical_name):
    """
    Categorize business concept based on semantic content
    
    Args:
        canonical_name: Canonical concept name
        
    Returns:
        str: Business category
    """
    name_lower = canonical_name.lower()
    
    if any(term in name_lower for term in ['income', 'revenue', 'financial', 'deferred', 'balance']):
        return "Financial Concepts"
    elif any(term in name_lower for term in ['contract', 'customer', 'service', 'agreement']):
        return "Customer & Contract Concepts"
    elif any(term in name_lower for term in ['operation', 'process', 'inventory', 'management', 'system']):
        return "Operational Concepts"
    else:
        return "General Business Concepts"

def get_business_concept_definitions():
    """
    Get comprehensive business concept definitions and templates
    
    Returns:
        dict: Business concept definition templates and knowledge base
    """
    return {
        # Financial concept definitions
        "financial_definitions": {
            "deferred income": {
                "definition": "Revenue received by a company for goods or services not yet delivered, recorded as a liability on the balance sheet until the performance obligation is fulfilled.",
                "synonyms": ["unearned revenue", "advance payments", "prepaid income", "deferred revenue"],
                "concept_type": "Financial Liability",
                "subcategory": "Current/Non-Current Liability",
                "accounting_standard": "GAAP/IFRS Revenue Recognition",
                "business_process": "Revenue Recognition",
                "related_concepts": ["revenue recognition", "contract liabilities", "performance obligations"]
            },
            "revenue": {
                "definition": "Income generated from normal business operations, calculated as the sum of all sales and services provided to customers.",
                "synonyms": ["sales", "turnover", "income from operations", "gross revenue"],
                "concept_type": "Financial Performance",
                "subcategory": "Income Statement Item", 
                "accounting_standard": "GAAP/IFRS Revenue Recognition",
                "business_process": "Sales and Revenue Management",
                "related_concepts": ["sales recognition", "gross profit", "operating income"]
            },
            "contract balances": {
                "definition": "Net amounts owed to or from customers under long-term contracts, representing the difference between revenue recognized and cash received.",
                "synonyms": ["contract assets", "contract liabilities", "customer advances", "progress billings"],
                "concept_type": "Financial Position",
                "subcategory": "Balance Sheet Item",
                "accounting_standard": "ASC 606/IFRS 15 Revenue Recognition",
                "business_process": "Contract Management",
                "related_concepts": ["revenue recognition", "performance obligations", "contract modifications"]
            }
        },
        
        # Operational concept definitions
        "operational_definitions": {
            "inventory": {
                "definition": "Goods and materials held by a business for sale, production, or consumption in the ordinary course of business operations.",
                "synonyms": ["stock", "goods", "merchandise", "raw materials", "finished goods"],
                "concept_type": "Operational Asset",
                "subcategory": "Current Asset",
                "accounting_standard": "GAAP/IFRS Inventory Accounting",
                "business_process": "Supply Chain Management",
                "related_concepts": ["cost of goods sold", "inventory valuation", "supply chain"]
            },
            "operations": {
                "definition": "Core business activities that generate revenue and manage resources to deliver products or services to customers.",
                "synonyms": ["business operations", "core activities", "operational processes", "business activities"],
                "concept_type": "Business Process",
                "subcategory": "Core Operations",
                "accounting_standard": "Operational Accounting",
                "business_process": "Operations Management",
                "related_concepts": ["operational efficiency", "business processes", "resource management"]
            }
        },
        
        # Customer & Contract concept definitions
        "customer_contract_definitions": {
            "customers": {
                "definition": "Individuals or entities that purchase goods or services from a business, representing the primary source of revenue generation.",
                "synonyms": ["clients", "buyers", "purchasers", "consumers", "end users"],
                "concept_type": "Business Relationship",
                "subcategory": "External Stakeholder",
                "accounting_standard": "Customer Accounting",
                "business_process": "Customer Relationship Management",
                "related_concepts": ["customer satisfaction", "customer retention", "customer acquisition"]
            },
            "contracts": {
                "definition": "Legal agreements between parties that establish terms, conditions, and obligations for the delivery of goods or services.",
                "synonyms": ["agreements", "service contracts", "sales contracts", "legal agreements"],
                "concept_type": "Legal Instrument",
                "subcategory": "Business Agreement",
                "accounting_standard": "ASC 606/IFRS 15 Contract Accounting",
                "business_process": "Contract Management",
                "related_concepts": ["contract performance", "contract modifications", "contract enforcement"]
            }
        },
        
        # Definition templates for dynamic generation
        "definition_templates": {
            "financial_income": "Revenue or income related to {concept_name}, representing financial performance and accounting treatment in business operations.",
            "financial_liability": "Financial obligation or liability related to {concept_name}, recorded on the balance sheet until fulfilled or settled.",
            "operational_process": "Business process or operational activity related to {concept_name}, supporting core business functions and operations.",
            "customer_relationship": "Customer-related concept involving {concept_name}, representing business relationships and customer management activities.",
            "general_business": "Business concept related to {concept_name}, representing important terminology and relationships in business operations."
        }
    }

def generate_detailed_concept_type(canonical_name, primary_keywords, business_category):
    """
    Generate detailed concept type classification based on semantic analysis
    
    Args:
        canonical_name: Canonical concept name
        primary_keywords: List of primary keywords
        business_category: High-level business category
        
    Returns:
        dict: Detailed concept type information
    """
    name_lower = canonical_name.lower()
    keywords_lower = [kw.lower() for kw in primary_keywords]
    
    # Financial concept typing
    if business_category == "Financial Concepts":
        if any(term in name_lower for term in ['income', 'revenue']):
            if any(term in name_lower for term in ['deferred', 'unearned', 'advance']):
                return {
                    "concept_type": "Financial Liability",
                    "subcategory": "Deferred Revenue/Income",
                    "accounting_classification": "Current/Non-Current Liability",
                    "financial_statement": "Balance Sheet",
                    "business_impact": "Revenue Recognition and Cash Flow Management"
                }
            else:
                return {
                    "concept_type": "Financial Performance",
                    "subcategory": "Revenue/Income",
                    "accounting_classification": "Revenue Account",
                    "financial_statement": "Income Statement", 
                    "business_impact": "Profitability and Performance Measurement"
                }
        elif any(term in name_lower for term in ['balance', 'asset', 'liability']):
            return {
                "concept_type": "Financial Position",
                "subcategory": "Balance Sheet Item",
                "accounting_classification": "Asset/Liability Account",
                "financial_statement": "Balance Sheet",
                "business_impact": "Financial Position and Capital Structure"
            }
        elif any(term in name_lower for term in ['tax', 'deferred tax']):
            return {
                "concept_type": "Tax Accounting",
                "subcategory": "Tax Asset/Liability",
                "accounting_classification": "Deferred Tax Item",
                "financial_statement": "Balance Sheet",
                "business_impact": "Tax Planning and Compliance"
            }
    
    # Operational concept typing  
    elif business_category == "Operational Concepts":
        if any(term in name_lower for term in ['inventory', 'stock', 'goods']):
            return {
                "concept_type": "Operational Asset",
                "subcategory": "Inventory Management",
                "accounting_classification": "Current Asset",
                "financial_statement": "Balance Sheet",
                "business_impact": "Supply Chain and Cost Management"
            }
        elif any(term in name_lower for term in ['operation', 'process', 'activity']):
            return {
                "concept_type": "Business Process",
                "subcategory": "Core Operations",
                "accounting_classification": "Operational Activity",
                "financial_statement": "Income Statement",
                "business_impact": "Operational Efficiency and Performance"
            }
    
    # Customer & Contract concept typing
    elif business_category == "Customer & Contract Concepts":
        if any(term in name_lower for term in ['contract', 'agreement']):
            return {
                "concept_type": "Legal Instrument", 
                "subcategory": "Business Contract",
                "accounting_classification": "Contract Asset/Liability",
                "financial_statement": "Balance Sheet",
                "business_impact": "Revenue Recognition and Risk Management"
            }
        elif any(term in name_lower for term in ['customer', 'client']):
            return {
                "concept_type": "Business Relationship",
                "subcategory": "External Stakeholder",
                "accounting_classification": "Customer Account",
                "financial_statement": "Multiple Statements",
                "business_impact": "Revenue Generation and Relationship Management"
            }
    
    # Default general business typing
    return {
        "concept_type": "General Business Concept",
        "subcategory": "Business Terminology",
        "accounting_classification": "General Business Item",
        "financial_statement": "Not Specific",
        "business_impact": "General Business Operations"
    }

def generate_concept_definition(canonical_name, primary_keywords, theme_variants, business_category, concept_type_info):
    """
    Generate comprehensive concept definition using templates and knowledge base
    
    Args:
        canonical_name: Canonical concept name
        primary_keywords: List of primary keywords
        theme_variants: List of theme variants
        business_category: High-level business category
        concept_type_info: Detailed concept type information
        
    Returns:
        dict: Comprehensive concept definition and metadata
    """
    knowledge_base = get_business_concept_definitions()
    
    # Try to find exact definition from knowledge base
    for category_key, definitions in knowledge_base.items():
        if isinstance(definitions, dict) and canonical_name in definitions:
            base_definition = definitions[canonical_name]
            return {
                "definition": base_definition["definition"],
                "synonyms": base_definition["synonyms"],
                "alternative_names": list(set(theme_variants + [canonical_name])),
                "detailed_type": concept_type_info,
                "domain_classification": base_definition["accounting_standard"],
                "business_process": base_definition["business_process"],
                "related_concepts": base_definition["related_concepts"],
                "definition_source": "Knowledge Base",
                "confidence": "High"
            }
    
    # Generate definition using semantic analysis and templates
    definition_templates = knowledge_base["definition_templates"]
    
    # Select appropriate template based on concept type
    if concept_type_info["concept_type"] == "Financial Liability":
        template = definition_templates["financial_liability"]
    elif concept_type_info["concept_type"] == "Financial Performance":
        template = definition_templates["financial_income"] 
    elif concept_type_info["concept_type"] == "Business Process":
        template = definition_templates["operational_process"]
    elif concept_type_info["concept_type"] == "Business Relationship":
        template = definition_templates["customer_relationship"]
    else:
        template = definition_templates["general_business"]
    
    # Generate definition from template
    generated_definition = template.format(concept_name=canonical_name)
    
    # Extract synonyms from keywords and variants
    synonyms = []
    for keyword in primary_keywords[:10]:  # Limit to avoid noise
        if keyword.lower() != canonical_name.lower() and len(keyword) > 2:
            synonyms.append(keyword.lower())
    
    # Clean up synonyms (remove duplicates and variations)
    cleaned_synonyms = list(set(synonyms))[:8]  # Limit to 8 synonyms
    
    # Generate related concepts from keywords
    related_concepts = []
    business_terms = ['revenue', 'income', 'balance', 'contract', 'customer', 'inventory', 'operation', 'management']
    for keyword in primary_keywords:
        for term in business_terms:
            if term in keyword.lower() and term != canonical_name.lower():
                related_concepts.append(term)
    
    return {
        "definition": generated_definition,
        "synonyms": cleaned_synonyms,
        "alternative_names": list(set(theme_variants + [canonical_name])),
        "detailed_type": concept_type_info,
        "domain_classification": f"{business_category} - {concept_type_info['subcategory']}",
        "business_process": concept_type_info.get('business_impact', 'Business Operations'),
        "related_concepts": list(set(related_concepts))[:5],
        "definition_source": "Generated from Keywords and Semantic Analysis",
        "confidence": "Medium"
    }

def discover_concept_relationships(core_concepts):
    """
    Discover relationships between concepts based on keyword overlap and semantic similarity
    
    Args:
        core_concepts: List of core concept objects
        
    Returns:
        dict: Concept relationship mappings
    """
    relationships = {}
    
    for i, concept_a in enumerate(core_concepts):
        concept_a_id = concept_a["concept_id"]
        relationships[concept_a_id] = {
            "strongly_related": [],
            "moderately_related": [],
            "same_category": [],
            "cross_category": []
        }
        
        # Find relationships with other concepts
        for j, concept_b in enumerate(core_concepts):
            if i == j:
                continue
                
            concept_b_id = concept_b["concept_id"]
            
            # Calculate keyword overlap
            keywords_a = set(kw.lower() for kw in concept_a["primary_keywords"])
            keywords_b = set(kw.lower() for kw in concept_b["primary_keywords"])
            overlap = keywords_a.intersection(keywords_b)
            overlap_ratio = len(overlap) / min(len(keywords_a), len(keywords_b)) if keywords_a and keywords_b else 0
            
            # Document co-occurrence
            docs_a = set(concept_a["related_documents"])
            docs_b = set(concept_b["related_documents"])
            doc_overlap = len(docs_a.intersection(docs_b))
            
            # Determine relationship strength
            if overlap_ratio > 0.3 or doc_overlap > 0:
                if overlap_ratio > 0.5:
                    relationships[concept_a_id]["strongly_related"].append({
                        "concept_id": concept_b_id,
                        "canonical_name": concept_b["canonical_name"],
                        "overlap_ratio": overlap_ratio,
                        "shared_documents": doc_overlap,
                        "shared_keywords": list(overlap)
                    })
                else:
                    relationships[concept_a_id]["moderately_related"].append({
                        "concept_id": concept_b_id, 
                        "canonical_name": concept_b["canonical_name"],
                        "overlap_ratio": overlap_ratio,
                        "shared_documents": doc_overlap,
                        "shared_keywords": list(overlap)
                    })
            
            # Category relationships
            category_a = categorize_business_concept(concept_a["canonical_name"])
            category_b = categorize_business_concept(concept_b["canonical_name"])
            
            if category_a == category_b:
                relationships[concept_a_id]["same_category"].append({
                    "concept_id": concept_b_id,
                    "canonical_name": concept_b["canonical_name"],
                    "shared_category": category_a
                })
            else:
                relationships[concept_a_id]["cross_category"].append({
                    "concept_id": concept_b_id,
                    "canonical_name": concept_b["canonical_name"],
                    "categories": [category_a, category_b]
                })
    
    return relationships

def generate_concept_mappings(core_concepts):
    """
    Generate mappings between concepts and documents
    
    Args:
        core_concepts: List of core concepts
        
    Returns:
        dict: Concept-document mappings
    """
    concept_to_docs = {}
    doc_to_concepts = {}
    
    for concept in core_concepts:
        concept_id = concept["concept_id"]
        concept_to_docs[concept_id] = concept["related_documents"]
        
        for doc_id in concept["related_documents"]:
            if doc_id not in doc_to_concepts:
                doc_to_concepts[doc_id] = []
            doc_to_concepts[doc_id].append(concept_id)
    
    return {
        "concept_to_documents": concept_to_docs,
        "document_to_concepts": doc_to_concepts
    }

def load_input(input_path="outputs/A2.3_concept_grouping_thematic.json"):
    """Load concept grouping from A2.3"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_core_concepts(data):
    """
    Process core concept synthesis from intra-document business themes
    
    Args:
        data: A2.3 intra-document clustering data with business themes
        
    Returns:
        dict: Core concepts with metadata and keyword_id tracking
    """
    documents = data.get("documents", [])
    total_docs = len(documents)
    
    if total_docs == 0:
        return {
            "core_concepts": [],
            "concept_hierarchy": {},
            "mappings": {"concept_to_documents": {}, "document_to_concepts": {}},
            "statistics": {
                "total_core_concepts": 0,
                "total_documents": 0,
                "document_coverage": 0,
                "coverage_percentage": 0,
                "avg_importance": 0,
                "business_categories": 0,
                "total_keyword_ids": 0
            },
            "processing_timestamp": datetime.now().isoformat()
        }
    
    # Aggregate themes across documents to identify cross-document concepts
    theme_aggregation = aggregate_themes_across_documents(documents)
    
    # Identify core concepts from aggregated themes
    core_concepts = identify_core_concepts(theme_aggregation, total_docs)
    
    # Create business hierarchy
    hierarchy = create_concept_hierarchy(core_concepts)
    
    # Generate mappings with keyword_id tracking
    mappings = generate_concept_mappings(core_concepts)
    
    # Discover concept relationships
    concept_relationships = discover_concept_relationships(core_concepts)
    
    # Calculate comprehensive statistics
    total_coverage = len(set().union(*[c["related_documents"] for c in core_concepts])) if core_concepts else 0
    total_keyword_ids = len(set().union(*[c["all_keyword_ids"] for c in core_concepts])) if core_concepts else 0
    
    # Enhanced statistics
    definition_sources = {}
    concept_types = {}
    for concept in core_concepts:
        def_source = concept["concept_definition"]["definition_source"]
        definition_sources[def_source] = definition_sources.get(def_source, 0) + 1
        
        concept_type = concept["concept_type"]["concept_type"]
        concept_types[concept_type] = concept_types.get(concept_type, 0) + 1
    
    return {
        "core_concepts": core_concepts,
        "concept_hierarchy": hierarchy,
        "concept_relationships": concept_relationships,
        "theme_aggregation_summary": {
            "total_unique_themes": len(theme_aggregation),
            "themes_promoted_to_core": len(core_concepts),
            "cross_document_themes": len([t for t in theme_aggregation.values() if len(t["document_instances"]) > 1])
        },
        "mappings": mappings,
        "statistics": {
            "total_core_concepts": len(core_concepts),
            "total_documents": total_docs,
            "document_coverage": total_coverage,
            "coverage_percentage": (total_coverage / total_docs * 100) if total_docs > 0 else 0,
            "avg_importance": sum(c["importance_score"] for c in core_concepts) / len(core_concepts) if core_concepts else 0,
            "business_categories": len(hierarchy),
            "total_keyword_ids": total_keyword_ids,
            "avg_keywords_per_concept": sum(c["unique_keyword_count"] for c in core_concepts) / len(core_concepts) if core_concepts else 0,
            "definition_sources": definition_sources,
            "concept_types": concept_types,
            "enhanced_features": {
                "concept_definitions": True,
                "detailed_typing": True,
                "relationship_discovery": True,
                "synonym_generation": True
            }
        },
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/A2.4_core_concepts.json"):
    """Save core concepts"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved core concepts to {full_path}")
    
    # Save domain mappings separately
    domain_path = full_path.with_name("A2.4_domain_mappings.json")
    with open(domain_path, 'w') as f:
        json.dump(data["mappings"], f, indent=2)
    
    # Save statistics
    stats_path = full_path.with_name("A2.4_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(data["statistics"], f, indent=2)

def main():
    """Main execution"""
    print("="*60)
    print("A2.4: Synthesize Core Concepts")
    print("="*60)
    
    try:
        # Load intra-document business themes from A2.3
        print("Loading intra-document business themes...")
        input_data = load_input()
        
        # Process core concept synthesis from business themes
        print("Synthesizing core business concepts from themes...")
        output_data = process_core_concepts(input_data)
        
        # Display results
        stats = output_data["statistics"]
        theme_summary = output_data["theme_aggregation_summary"]
        print(f"\nCore Concept Statistics:")
        print(f"  Total Core Concepts: {stats['total_core_concepts']}")
        print(f"  Document Coverage: {stats['document_coverage']}/{stats['total_documents']} ({stats['coverage_percentage']:.1f}%)")
        print(f"  Business Categories: {stats['business_categories']}")
        print(f"  Average Importance: {stats['avg_importance']:.3f}")
        print(f"  Total Keyword IDs: {stats['total_keyword_ids']}")
        print(f"  Avg Keywords per Concept: {stats['avg_keywords_per_concept']:.1f}")
        print(f"\nTheme Aggregation Summary:")
        print(f"  Unique Themes Identified: {theme_summary['total_unique_themes']}")
        print(f"  Themes Promoted to Core: {theme_summary['themes_promoted_to_core']}")
        print(f"  Cross-Document Themes: {theme_summary['cross_document_themes']}")
        
        print(f"\nEnhanced Concept Analysis:")
        print(f"  Definition Sources: {stats.get('definition_sources', {})}")
        print(f"  Concept Types: {stats.get('concept_types', {})}")
        enhanced = stats.get('enhanced_features', {})
        enabled_features = [k for k, v in enhanced.items() if v]
        print(f"  Enhanced Features: {', '.join(enabled_features)}")
        
        print(f"\nTop 5 Enhanced Core Business Concepts:")
        for i, concept in enumerate(output_data["core_concepts"][:5], 1):
            print(f"  {i}. {concept['canonical_name']}")
            print(f"     Importance: {concept['importance_score']:.3f}")
            print(f"     Type: {concept['concept_type']['concept_type']}")
            print(f"     Category: {concept['business_category']}")
            print(f"     Documents: {concept['document_count']}/{stats['total_documents']} ({concept['coverage_ratio']:.1%})")
            print(f"     Definition: {concept['concept_definition']['definition'][:120]}...")
            print(f"     Synonyms: {', '.join(concept['concept_definition']['synonyms'][:3])}")
            print(f"     Keywords: {', '.join(concept['primary_keywords'][:3])}")
            print(f"     Keyword IDs: {len(concept['all_keyword_ids'])} tracked")
            if concept['theme_variants']:
                print(f"     Theme Variants: {', '.join(list(concept['theme_variants'])[:2])}")
            print()
        
        # Save output
        save_output(output_data)
        
        print("\nA2.4 Core Concept Synthesis completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.4 Core Concept Synthesis: {str(e)}")
        raise

if __name__ == "__main__":
    main()