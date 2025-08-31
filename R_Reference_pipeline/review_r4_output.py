#!/usr/bin/env python3
"""
R4 Output Review Tool
Provides an interactive way to explore the R4 semantic ontology
"""

import json
from pathlib import Path
from collections import Counter

def load_ontology():
    """Load the R4 ontology files"""
    output_dir = Path("output")
    
    with open(output_dir / "R4_semantic_ontology.json", 'r') as f:
        ontology = json.load(f)
    
    with open(output_dir / "R4_integration_api.json", 'r') as f:
        api = json.load(f)
    
    with open(output_dir / "R4_ontology_statistics.json", 'r') as f:
        stats = json.load(f)
    
    return ontology, api, stats

def print_statistics(ontology, stats):
    """Print key statistics"""
    print("\n" + "="*60)
    print("R4 SEMANTIC ONTOLOGY STATISTICS")
    print("="*60)
    
    ont_stats = ontology['ontology']['statistics']
    print(f"\nBasic Metrics:")
    print(f"  Total Concepts: {ont_stats['total_concepts']}")
    print(f"  Total Clusters: {ont_stats['total_clusters']}")
    print(f"  Hierarchy Depth: {ont_stats['hierarchy_max_depth']}")
    print(f"  Total Relationships: {ont_stats['relationships_total']}")
    
    print(f"\nRelationship Breakdown:")
    for rel_type, count in ont_stats['relationships_by_type'].items():
        print(f"  {rel_type.capitalize()}: {count}")
    
    print(f"\nAverage Metrics:")
    print(f"  Avg Relationships per Concept: {ont_stats['avg_relationships_per_concept']:.1f}")
    print(f"  Avg Connectivity Score: {ont_stats['avg_connectivity_score']:.2f}")
    print(f"  Cluster Avg Coherence: {ont_stats['cluster_avg_coherence']:.2f}")

def show_top_connected_concepts(ontology, n=10):
    """Show the most connected concepts"""
    print(f"\n" + "="*60)
    print(f"TOP {n} MOST CONNECTED CONCEPTS (Knowledge Hubs)")
    print("="*60)
    
    concepts = ontology['ontology']['concepts']
    
    # Calculate total connections for each concept
    connections = []
    for concept_id, data in concepts.items():
        total = data['ontology_metadata']['relationship_count']
        connections.append((concept_id, data['name'], total))
    
    # Sort by connection count
    connections.sort(key=lambda x: x[2], reverse=True)
    
    for i, (concept_id, name, count) in enumerate(connections[:n], 1):
        domain = concepts[concept_id]['domain']
        print(f"{i:2}. {name:30} - {count} relationships ({domain})")

def explore_concept(ontology, concept_name):
    """Explore a specific concept"""
    concepts = ontology['ontology']['concepts']
    
    # Find concept by name (case-insensitive)
    concept_id = None
    for cid, data in concepts.items():
        if data['name'].lower() == concept_name.lower():
            concept_id = cid
            break
    
    if not concept_id:
        print(f"\nConcept '{concept_name}' not found.")
        return
    
    concept = concepts[concept_id]
    print(f"\n" + "="*60)
    print(f"CONCEPT: {concept['name']}")
    print("="*60)
    
    print(f"\nDefinition:")
    print(f"  {concept['definition'][:150]}...")
    
    print(f"\nMetadata:")
    print(f"  Domain: {concept['domain']}")
    print(f"  Cluster: {concept['cluster']}")
    print(f"  Hierarchy Level: {concept['hierarchy']['level']}")
    print(f"  Parent: {concept['hierarchy']['parent']}")
    print(f"  Is Leaf: {concept['hierarchy'].get('is_leaf', False)}")
    
    print(f"\nKeywords ({len(concept['keywords'])} total):")
    print(f"  {', '.join(concept['keywords'][:10])}")
    
    print(f"\nRelationships:")
    for rel_type, related in concept['relationships'].items():
        if related:
            print(f"  {rel_type.capitalize()} ({len(related)}):")
            if rel_type == 'semantic' and related:
                # Show names for semantic relationships
                names = []
                for rid in related[:5]:
                    if rid in concepts:
                        names.append(concepts[rid]['name'])
                print(f"    {', '.join(names)}")

def show_clusters(ontology):
    """Show cluster information"""
    print(f"\n" + "="*60)
    print("SEMANTIC CLUSTERS")
    print("="*60)
    
    clusters = ontology['ontology']['clusters']
    
    # Sort by size
    sorted_clusters = sorted(clusters.items(), 
                           key=lambda x: x[1]['size'], 
                           reverse=True)
    
    for cluster_id, cluster_data in sorted_clusters[:10]:
        print(f"\n{cluster_data['name']}:")
        print(f"  Size: {cluster_data['size']} concepts")
        print(f"  Coherence: {cluster_data['coherence_score']:.3f}")
        print(f"  Top Keywords: {', '.join(cluster_data['top_keywords'][:5])}")

def show_domain_distribution(ontology):
    """Show concept distribution by domain"""
    print(f"\n" + "="*60)
    print("DOMAIN DISTRIBUTION")
    print("="*60)
    
    concepts = ontology['ontology']['concepts']
    domains = Counter()
    
    for concept in concepts.values():
        domains[concept['domain']] += 1
    
    total = sum(domains.values())
    for domain, count in domains.most_common():
        percentage = (count / total) * 100
        print(f"  {domain:25} - {count:3} concepts ({percentage:5.1f}%)")

def interactive_review():
    """Interactive review menu"""
    ontology, api, stats = load_ontology()
    
    while True:
        print(f"\n" + "="*60)
        print("R4 ONTOLOGY REVIEW MENU")
        print("="*60)
        print("1. Show Statistics")
        print("2. Show Top Connected Concepts")
        print("3. Show Clusters")
        print("4. Show Domain Distribution")
        print("5. Explore Specific Concept")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            print_statistics(ontology, stats)
        elif choice == '2':
            show_top_connected_concepts(ontology)
        elif choice == '3':
            show_clusters(ontology)
        elif choice == '4':
            show_domain_distribution(ontology)
        elif choice == '5':
            concept_name = input("Enter concept name (e.g., 'Agreement'): ").strip()
            explore_concept(ontology, concept_name)
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    print("R4 Semantic Ontology Review Tool")
    print("-" * 40)
    
    # Quick summary first
    ontology, api, stats = load_ontology()
    print_statistics(ontology, stats)
    
    # Ask if user wants interactive mode
    response = input("\nEnter interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_review()
    else:
        # Show quick highlights
        show_top_connected_concepts(ontology, 5)
        show_domain_distribution(ontology)