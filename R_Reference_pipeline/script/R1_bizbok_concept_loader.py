#!/usr/bin/env python3
"""
R1: BizBOK Concept Loader
Loads and processes Business Body of Knowledge (BizBOK) reference concepts
for concept validation and reference mapping in the conceptual space system
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# BizBOK Business Concepts Reference Data
BIZBOK_CONCEPTS = {
    "finance": {
        "revenue_recognition": {
            "definition": "Process of recording revenue when earned according to accounting principles",
            "related_terms": ["revenue", "income", "recognition", "accounting", "earnings"],
            "domain": "finance",
            "category": "accounting_principles",
            "importance": 0.9
        },
        "cash_flow": {
            "definition": "Movement of cash in and out of business operations",
            "related_terms": ["cash", "flow", "operations", "liquidity", "working_capital"],
            "domain": "finance", 
            "category": "financial_analysis",
            "importance": 0.85
        },
        "deferred_income": {
            "definition": "Revenue received but not yet earned, recorded as liability",
            "related_terms": ["deferred", "unearned", "liability", "prepaid", "advance"],
            "domain": "finance",
            "category": "accounting_principles", 
            "importance": 0.8
        },
        "cost_of_goods_sold": {
            "definition": "Direct costs attributable to production of goods sold",
            "related_terms": ["cost", "goods", "sold", "cogs", "direct_cost", "production"],
            "domain": "finance",
            "category": "cost_accounting",
            "importance": 0.85
        },
        "working_capital": {
            "definition": "Short-term assets minus short-term liabilities",
            "related_terms": ["working", "capital", "current_assets", "current_liabilities", "liquidity"],
            "domain": "finance",
            "category": "financial_analysis",
            "importance": 0.8
        }
    },
    "operations": {
        "supply_chain": {
            "definition": "Network of suppliers, manufacturers, and distributors",
            "related_terms": ["supply", "chain", "logistics", "procurement", "distribution"],
            "domain": "operations",
            "category": "supply_management",
            "importance": 0.85
        },
        "quality_management": {
            "definition": "Systematic approach to ensuring product and service quality",
            "related_terms": ["quality", "management", "control", "assurance", "improvement"],
            "domain": "operations",
            "category": "quality_systems",
            "importance": 0.8
        },
        "inventory_management": {
            "definition": "Oversight of ordering, storing, and using company inventory",
            "related_terms": ["inventory", "stock", "management", "ordering", "warehousing"],
            "domain": "operations",
            "category": "inventory_systems",
            "importance": 0.75
        }
    },
    "strategy": {
        "competitive_advantage": {
            "definition": "Unique position that allows superior performance over competitors",
            "related_terms": ["competitive", "advantage", "differentiation", "strategy", "positioning"],
            "domain": "strategy",
            "category": "strategic_planning",
            "importance": 0.9
        },
        "market_analysis": {
            "definition": "Assessment of market conditions, trends, and competitive landscape",
            "related_terms": ["market", "analysis", "research", "competitive", "trends"],
            "domain": "strategy",
            "category": "market_research",
            "importance": 0.8
        }
    },
    "technology": {
        "digital_transformation": {
            "definition": "Integration of digital technology into all business areas",
            "related_terms": ["digital", "transformation", "technology", "automation", "innovation"],
            "domain": "technology",
            "category": "digital_strategy",
            "importance": 0.85
        },
        "data_analytics": {
            "definition": "Process of examining data to draw conclusions and insights",
            "related_terms": ["data", "analytics", "analysis", "insights", "intelligence"],
            "domain": "technology",
            "category": "data_management",
            "importance": 0.8
        }
    }
}

def simulate_concept_embedding(concept_terms, dimension=384):
    """
    Simulate embedding vector for BizBOK concept
    
    Args:
        concept_terms: List of terms related to concept
        dimension: Vector dimension
        
    Returns:
        numpy.ndarray: Simulated concept embedding
    """
    if not concept_terms:
        return np.random.random(dimension)
    
    # Create reproducible vector based on concept terms
    combined_text = " ".join(str(term).lower() for term in concept_terms)
    hash_value = abs(hash(combined_text))
    
    # Generate deterministic vector
    np.random.seed(hash_value % (2**32))
    vector = np.random.normal(0, 1, dimension)
    
    # Add concept-specific characteristics
    for i, term in enumerate(concept_terms[:10]):
        term_hash = abs(hash(term)) % dimension
        vector[term_hash] += 1.0
    
    # Normalize vector
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def process_bizbok_concepts():
    """
    Process BizBOK concepts into standardized format
    
    Returns:
        dict: Processed BizBOK concepts with embeddings and metadata
    """
    processed_concepts = {}
    
    for domain, domain_concepts in BIZBOK_CONCEPTS.items():
        for concept_id, concept_data in domain_concepts.items():
            
            # Create full concept ID
            full_concept_id = f"bizbok_{domain}_{concept_id}"
            
            # Extract related terms
            related_terms = concept_data["related_terms"]
            all_terms = related_terms + [concept_id.replace("_", " ")]
            
            # Generate concept embedding
            concept_embedding = simulate_concept_embedding(all_terms)
            
            # Process concept
            processed_concept = {
                "concept_id": full_concept_id,
                "name": concept_id.replace("_", " ").title(),
                "definition": concept_data["definition"],
                "domain": concept_data["domain"],
                "category": concept_data["category"],
                "related_terms": related_terms,
                "all_terms": all_terms,
                "importance_score": concept_data["importance"],
                "embedding_vector": concept_embedding.tolist(),
                "source": "BizBOK",
                "validation_status": "reference"
            }
            
            processed_concepts[full_concept_id] = processed_concept
    
    return processed_concepts

def generate_concept_mappings(processed_concepts):
    """
    Generate mappings and relationships between BizBOK concepts
    
    Args:
        processed_concepts: Processed BizBOK concepts
        
    Returns:
        dict: Concept mappings and relationships
    """
    # Domain mappings
    domain_mapping = defaultdict(list)
    category_mapping = defaultdict(list)
    term_to_concepts = defaultdict(list)
    
    for concept_id, concept_data in processed_concepts.items():
        domain = concept_data["domain"]
        category = concept_data["category"]
        
        domain_mapping[domain].append(concept_id)
        category_mapping[category].append(concept_id)
        
        # Map terms to concepts
        for term in concept_data["all_terms"]:
            term_to_concepts[term.lower()].append(concept_id)
    
    # Calculate concept similarities
    concept_similarities = {}
    concept_ids = list(processed_concepts.keys())
    
    for i, concept1_id in enumerate(concept_ids):
        concept_similarities[concept1_id] = {}
        concept1 = processed_concepts[concept1_id]
        
        for concept2_id in concept_ids[i+1:]:
            concept2 = processed_concepts[concept2_id]
            
            # Calculate similarity based on term overlap
            terms1 = set(term.lower() for term in concept1["all_terms"])
            terms2 = set(term.lower() for term in concept2["all_terms"])
            
            if terms1 and terms2:
                similarity = len(terms1 & terms2) / len(terms1 | terms2)
            else:
                similarity = 0.0
            
            concept_similarities[concept1_id][concept2_id] = similarity
            
            # Symmetric relationship
            if concept2_id not in concept_similarities:
                concept_similarities[concept2_id] = {}
            concept_similarities[concept2_id][concept1_id] = similarity
    
    return {
        "domain_mapping": dict(domain_mapping),
        "category_mapping": dict(category_mapping),
        "term_to_concepts": dict(term_to_concepts),
        "concept_similarities": concept_similarities
    }

def validate_concept_completeness(processed_concepts):
    """
    Validate completeness and quality of BizBOK concept collection
    
    Args:
        processed_concepts: Processed concepts
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        "total_concepts": len(processed_concepts),
        "domains_covered": set(),
        "categories_covered": set(),
        "avg_terms_per_concept": 0,
        "avg_importance_score": 0,
        "completeness_issues": []
    }
    
    total_terms = 0
    total_importance = 0
    
    for concept_data in processed_concepts.values():
        validation_results["domains_covered"].add(concept_data["domain"])
        validation_results["categories_covered"].add(concept_data["category"])
        total_terms += len(concept_data["all_terms"])
        total_importance += concept_data["importance_score"]
        
        # Check for completeness issues
        if len(concept_data["related_terms"]) < 3:
            validation_results["completeness_issues"].append(
                f"{concept_data['concept_id']}: Insufficient related terms"
            )
        
        if len(concept_data["definition"]) < 20:
            validation_results["completeness_issues"].append(
                f"{concept_data['concept_id']}: Definition too brief"
            )
    
    validation_results["domains_covered"] = list(validation_results["domains_covered"])
    validation_results["categories_covered"] = list(validation_results["categories_covered"])
    validation_results["avg_terms_per_concept"] = total_terms / len(processed_concepts)
    validation_results["avg_importance_score"] = total_importance / len(processed_concepts)
    
    return validation_results

def save_outputs(processed_concepts, mappings, validation_results):
    """
    Save BizBOK concept processing results
    
    Args:
        processed_concepts: Processed concepts
        mappings: Concept mappings
        validation_results: Validation results
    """
    script_dir = Path(__file__).parent.parent
    
    # Main output
    output_data = {
        "bizbok_concepts": processed_concepts,
        "concept_mappings": mappings,
        "validation_results": validation_results,
        "processing_metadata": {
            "processing_timestamp": datetime.now().isoformat(),
            "total_concepts": len(processed_concepts),
            "source": "BizBOK Reference",
            "version": "1.0"
        }
    }
    
    output_path = script_dir / "output/R1_bizbok_concepts.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved BizBOK concepts to {output_path}")
    
    # Domain mappings
    domain_path = script_dir / "output/R1_domain_mapping.json"
    with open(domain_path, 'w', encoding='utf-8') as f:
        json.dump(mappings["domain_mapping"], f, indent=2)
    
    print(f"✓ Saved domain mappings to {domain_path}")

def main():
    """Main execution"""
    print("="*60)
    print("R1: BizBOK Concept Loader")
    print("="*60)
    
    try:
        # Process BizBOK concepts
        print("Processing BizBOK reference concepts...")
        processed_concepts = process_bizbok_concepts()
        
        # Generate mappings
        print("Generating concept mappings and relationships...")
        mappings = generate_concept_mappings(processed_concepts)
        
        # Validate completeness
        print("Validating concept completeness...")
        validation_results = validate_concept_completeness(processed_concepts)
        
        # Display results
        print(f"\nBizBOK Concept Processing Results:")
        print(f"  Total Concepts: {validation_results['total_concepts']}")
        print(f"  Domains Covered: {len(validation_results['domains_covered'])}")
        print(f"  Categories Covered: {len(validation_results['categories_covered'])}")
        print(f"  Avg Terms per Concept: {validation_results['avg_terms_per_concept']:.1f}")
        print(f"  Avg Importance Score: {validation_results['avg_importance_score']:.3f}")
        
        print(f"\nDomains: {', '.join(validation_results['domains_covered'])}")
        
        if validation_results["completeness_issues"]:
            print(f"\nCompleteness Issues:")
            for issue in validation_results["completeness_issues"][:5]:
                print(f"  • {issue}")
        
        # Save outputs
        save_outputs(processed_concepts, mappings, validation_results)
        
        print("\nR1 BizBOK Concept Loader completed successfully!")
        
    except Exception as e:
        print(f"Error in R1 BizBOK Concept Loader: {str(e)}")
        raise

if __name__ == "__main__":
    main()