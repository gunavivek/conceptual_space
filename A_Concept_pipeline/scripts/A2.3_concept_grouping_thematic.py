#!/usr/bin/env python3
"""
A2.3: Concept Grouping Thematic
Groups related concepts thematically using clustering and semantic analysis
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import re
import math

def calculate_concept_similarity(concept1, concept2):
    """
    Calculate similarity between two concepts
    
    Args:
        concept1: First concept keywords
        concept2: Second concept keywords
        
    Returns:
        float: Similarity score (0-1)
    """
    # Simple Jaccard similarity for now
    set1 = set(concept1)
    set2 = set(concept2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union

def group_concepts_by_theme(all_concepts, threshold=0.3):
    """
    Group concepts into thematic clusters
    
    Args:
        all_concepts: List of concept data
        threshold: Similarity threshold for grouping
        
    Returns:
        list: Thematic groups
    """
    # Create concept vectors (simplified - using keywords as features)
    concept_vectors = {}
    for concept in all_concepts:
        keywords = [kw["term"] for kw in concept.get("keywords", [])]
        concept_vectors[concept["doc_id"]] = keywords
    
    # Group concepts by similarity
    groups = []
    processed = set()
    
    for doc_id, keywords in concept_vectors.items():
        if doc_id in processed:
            continue
            
        # Start new group
        group = {
            "theme_id": f"theme_{len(groups) + 1}",
            "documents": [doc_id],
            "common_keywords": Counter(keywords),
            "representative_keywords": []
        }
        processed.add(doc_id)
        
        # Find similar concepts to add to group
        for other_id, other_keywords in concept_vectors.items():
            if other_id in processed:
                continue
                
            similarity = calculate_concept_similarity(keywords, other_keywords)
            if similarity > threshold:
                group["documents"].append(other_id)
                group["common_keywords"].update(other_keywords)
                processed.add(other_id)
        
        # Get representative keywords for the group
        group["representative_keywords"] = [
            kw for kw, count in group["common_keywords"].most_common(10)
        ]
        
        groups.append(group)
    
    return groups

def identify_domain_themes(groups, domain_info):
    """
    Identify domain-specific themes
    
    Args:
        groups: Concept groups
        domain_info: Domain information from documents
        
    Returns:
        dict: Domain-specific themes
    """
    domain_themes = defaultdict(list)
    
    for group in groups:
        # Determine dominant domain for this group
        domains = []
        for doc_id in group["documents"]:
            doc_domain = domain_info.get(doc_id, "general")
            domains.append(doc_domain)
        
        # Find most common domain
        domain_counter = Counter(domains)
        dominant_domain = domain_counter.most_common(1)[0][0]
        
        group["dominant_domain"] = dominant_domain
        domain_themes[dominant_domain].append(group)
    
    return domain_themes

def generate_theme_names(groups):
    """
    Generate descriptive names for themes
    
    Args:
        groups: Concept groups
        
    Returns:
        list: Groups with generated theme names
    """
    for group in groups:
        keywords = group["representative_keywords"][:3]
        
        # Create theme name from top keywords
        if len(keywords) >= 2:
            theme_name = f"{keywords[0].title()} & {keywords[1].title()}"
        elif len(keywords) == 1:
            theme_name = f"{keywords[0].title()} Concepts"
        else:
            theme_name = f"Theme {group['theme_id']}"
        
        group["theme_name"] = theme_name
    
    return groups

def load_input(input_path="outputs/A2.2_keyword_extractions.json"):
    """Load keyword extractions from A2.2"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / input_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_concept_grouping(data):
    """
    Process concept grouping for documents
    
    Args:
        data: Keyword extraction data
        
    Returns:
        dict: Grouped concepts
    """
    documents = data.get("documents", [])
    
    # Extract domain information if available
    domain_info = {}
    for doc in documents:
        doc_id = doc.get("doc_id", "")
        domain_info[doc_id] = doc.get("domain", "general")
    
    # Group concepts thematically
    groups = group_concepts_by_theme(documents)
    
    # Identify domain themes
    domain_themes = identify_domain_themes(groups, domain_info)
    
    # Generate theme names
    groups = generate_theme_names(groups)
    
    # Calculate statistics
    total_docs = len(documents)
    avg_docs_per_theme = sum(len(g["documents"]) for g in groups) / max(len(groups), 1)
    
    return {
        "documents": documents,
        "thematic_groups": groups,
        "domain_themes": dict(domain_themes),
        "statistics": {
            "total_documents": total_docs,
            "total_themes": len(groups),
            "average_docs_per_theme": avg_docs_per_theme,
            "themes_by_domain": {
                domain: len(themes) for domain, themes in domain_themes.items()
            }
        },
        "processing_timestamp": datetime.now().isoformat()
    }

def save_output(data, output_path="outputs/A2.3_concept_grouping_thematic.json"):
    """Save concept grouping results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved concept grouping to {full_path}")
    
    # Save statistics summary
    stats_path = full_path.with_suffix('.json').with_name(full_path.stem + '_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(data["statistics"], f, indent=2)

def main():
    """Main execution"""
    print("="*60)
    print("A2.3: Concept Grouping Thematic")
    print("="*60)
    
    try:
        # Load keyword extractions
        print("Loading keyword extractions...")
        input_data = load_input()
        
        # Process concept grouping
        print(f"Grouping concepts from {input_data['count']} documents...")
        output_data = process_concept_grouping(input_data)
        
        # Display results
        stats = output_data["statistics"]
        print(f"\nGrouping Statistics:")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Total Themes: {stats['total_themes']}")
        print(f"  Avg Docs/Theme: {stats['average_docs_per_theme']:.1f}")
        
        print(f"\nThemes by Domain:")
        for domain, count in stats["themes_by_domain"].items():
            print(f"  {domain}: {count} themes")
        
        print(f"\nSample Themes:")
        for i, group in enumerate(output_data["thematic_groups"][:5], 1):
            print(f"  {i}. {group['theme_name']}")
            print(f"     Keywords: {', '.join(group['representative_keywords'][:5])}")
            print(f"     Documents: {len(group['documents'])}")
        
        # Save output
        save_output(output_data)
        
        print("\nA2.3 Concept Grouping completed successfully!")
        
    except Exception as e:
        print(f"Error in A2.3 Concept Grouping: {str(e)}")
        raise

if __name__ == "__main__":
    main()