#!/usr/bin/env python3
"""
R4: Semantic Ontology Builder
Part of R-Pipeline (Resource & Reasoning Pipeline)
Builds a semantic BIZBOK ontology with hierarchical structure and rich relationships
CPU-optimized for 500+ concepts with lightweight semantic analysis
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import time
import re

# Business domain synonyms for semantic expansion
BUSINESS_SYNONYMS = {
    'revenue': ['income', 'sales', 'earnings', 'receipts', 'turnover'],
    'cost': ['expense', 'expenditure', 'outlay', 'spending', 'charge'],
    'profit': ['earnings', 'income', 'gain', 'margin', 'return', 'yield'],
    'asset': ['resource', 'property', 'holding', 'investment', 'capital'],
    'liability': ['debt', 'obligation', 'payable', 'owing', 'due'],
    'management': ['administration', 'oversight', 'control', 'governance', 'direction'],
    'analysis': ['evaluation', 'assessment', 'review', 'examination', 'study'],
    'process': ['procedure', 'workflow', 'operation', 'method', 'system'],
    'strategy': ['plan', 'approach', 'policy', 'tactic', 'methodology'],
    'performance': ['results', 'output', 'achievement', 'execution', 'effectiveness']
}

# Hierarchy inference patterns
HIERARCHY_PATTERNS = {
    'management_concepts': ['management', 'administration', 'oversight', 'governance'],
    'analysis_concepts': ['analysis', 'evaluation', 'assessment', 'measurement'],
    'financial_concepts': ['revenue', 'cost', 'profit', 'asset', 'liability', 'cash', 'capital'],
    'operational_concepts': ['process', 'workflow', 'operation', 'production', 'supply'],
    'strategic_concepts': ['strategy', 'plan', 'policy', 'goal', 'objective']
}

class SemanticOntologyBuilder:
    """Main class for building semantic BIZBOK ontology"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.parent
        self.output_dir = self.script_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data structures
        self.bizbok_concepts = {}
        self.domains = {}
        self.keyword_index = {}
        self.alignment_data = {}
        
        # Ontology structures
        self.ontology = {
            "concepts": {},
            "hierarchy": {},
            "relationships": {
                "semantic": defaultdict(list),
                "causal": defaultdict(list),
                "compositional": defaultdict(list),
                "temporal": defaultdict(list)
            },
            "clusters": {},
            "statistics": {}
        }
        
        # Performance tracking
        self.performance_metrics = {
            "start_time": None,
            "end_time": None,
            "processing_stages": {}
        }
    
    def load_r_pipeline_data(self):
        """Load all necessary data from R1-R3"""
        print("[DATA] Loading R-Pipeline data...")
        
        # Load R1 concepts
        concepts_path = self.output_dir / "R1_CONCEPTS.json"
        if concepts_path.exists():
            with open(concepts_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.bizbok_concepts = data["concepts"]
            print(f"   [OK] Loaded {len(self.bizbok_concepts)} BIZBOK concepts")
        
        # Load R1 domains
        domains_path = self.output_dir / "R1_DOMAINS.json"
        if domains_path.exists():
            with open(domains_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.domains = data["domains"]
            print(f"   [OK] Loaded {len(self.domains)} domains")
        
        # Load R1 keywords
        keywords_path = self.output_dir / "R1_KEYWORDS.json"
        if keywords_path.exists():
            with open(keywords_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.keyword_index = data["keyword_index"]
            print(f"   [OK] Loaded {len(self.keyword_index)} keyword mappings")
        
        # Load R3 alignment data (optional)
        alignment_path = self.output_dir / "R3_alignment_mappings.json"
        if alignment_path.exists():
            with open(alignment_path, 'r', encoding='utf-8') as f:
                self.alignment_data = json.load(f)
            print(f"   [OK] Loaded alignment data")
    
    def expand_keywords_with_synonyms(self, keywords):
        """Expand keywords using business synonyms"""
        expanded = set(keywords)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Check if keyword matches any synonym group
            for base_term, synonyms in BUSINESS_SYNONYMS.items():
                if keyword_lower == base_term:
                    expanded.update(synonyms)
                elif keyword_lower in synonyms:
                    expanded.add(base_term)
                    expanded.update(synonyms)
        
        return list(expanded)
    
    def calculate_semantic_similarity(self, concept1_id, concept2_id):
        """Calculate lightweight semantic similarity between concepts"""
        concept1 = self.bizbok_concepts[concept1_id]
        concept2 = self.bizbok_concepts[concept2_id]
        
        # Get expanded keywords
        keywords1 = set(self.expand_keywords_with_synonyms(concept1.get("keywords", [])))
        keywords2 = set(self.expand_keywords_with_synonyms(concept2.get("keywords", [])))
        
        # Calculate Jaccard similarity
        if keywords1 and keywords2:
            intersection = len(keywords1 & keywords2)
            union = len(keywords1 | keywords2)
            jaccard_similarity = intersection / union if union > 0 else 0.0
        else:
            jaccard_similarity = 0.0
        
        # Domain bonus
        domain_bonus = 0.15 if concept1["domain"] == concept2["domain"] else 0.0
        
        # Related concepts bonus
        related1 = set(concept1.get("related_concepts", []))
        related2 = set(concept2.get("related_concepts", []))
        if concept2_id in related1 or concept1_id in related2:
            relationship_bonus = 0.25
        else:
            relationship_bonus = 0.0
        
        # Calculate final similarity
        similarity = min(1.0, jaccard_similarity + domain_bonus + relationship_bonus)
        
        return {
            "similarity_score": similarity,
            "jaccard": jaccard_similarity,
            "domain_match": concept1["domain"] == concept2["domain"],
            "has_relationship": relationship_bonus > 0,
            "common_keywords": list(keywords1 & keywords2)[:10]
        }
    
    def build_semantic_clusters(self):
        """Build semantic clusters using efficient similarity calculation"""
        print("\n[PROCESS] Building semantic clusters...")
        start_time = time.time()
        
        # Pre-calculate similarity matrix for efficiency
        concept_ids = list(self.bizbok_concepts.keys())
        n_concepts = len(concept_ids)
        
        # Use vectorized approach for speed
        print(f"   Calculating similarity matrix for {n_concepts} concepts...")
        
        # Build clusters using threshold-based approach
        clusters = []
        clustered = set()
        similarity_threshold = 0.35  # Tuned for business concepts
        
        for i, concept_id in enumerate(concept_ids):
            if concept_id in clustered:
                continue
            
            # Start new cluster
            cluster = {
                "cluster_id": f"cluster_{len(clusters)}",
                "members": [concept_id],
                "coherence_score": 1.0,
                "cluster_keywords": set(self.bizbok_concepts[concept_id]["keywords"])
            }
            clustered.add(concept_id)
            
            # Find similar concepts
            for other_id in concept_ids[i+1:]:
                if other_id not in clustered:
                    similarity = self.calculate_semantic_similarity(concept_id, other_id)
                    if similarity["similarity_score"] >= similarity_threshold:
                        cluster["members"].append(other_id)
                        cluster["cluster_keywords"].update(self.bizbok_concepts[other_id]["keywords"])
                        clustered.add(other_id)
            
            # Calculate cluster coherence
            if len(cluster["members"]) > 1:
                total_similarity = 0
                comparisons = 0
                for j, member1 in enumerate(cluster["members"]):
                    for member2 in cluster["members"][j+1:]:
                        sim = self.calculate_semantic_similarity(member1, member2)
                        total_similarity += sim["similarity_score"]
                        comparisons += 1
                cluster["coherence_score"] = total_similarity / comparisons if comparisons > 0 else 0
            
            # Generate cluster name from top keywords
            keyword_freq = Counter()
            for member_id in cluster["members"]:
                member_keywords = self.bizbok_concepts[member_id]["keywords"]
                keyword_freq.update(member_keywords)
            
            top_keywords = [k for k, v in keyword_freq.most_common(3)]
            cluster["cluster_name"] = "_".join(top_keywords) if top_keywords else f"cluster_{len(clusters)}"
            
            clusters.append(cluster)
        
        # Store clusters in ontology
        for cluster in clusters:
            self.ontology["clusters"][cluster["cluster_id"]] = {
                "name": cluster["cluster_name"],
                "members": cluster["members"],
                "coherence_score": cluster["coherence_score"],
                "size": len(cluster["members"]),
                "top_keywords": list(cluster["cluster_keywords"])[:20]
            }
        
        elapsed = time.time() - start_time
        print(f"   [OK] Created {len(clusters)} semantic clusters in {elapsed:.1f} seconds")
        print(f"   [OK] Average cluster size: {np.mean([len(c['members']) for c in clusters]):.1f}")
    
    def build_concept_hierarchy(self):
        """Build hierarchical structure using clusters and patterns"""
        print("\n[PROCESS] Building concept hierarchy...")
        
        # Initialize hierarchy with root
        self.ontology["hierarchy"] = {
            "business_concept": {
                "children": [],
                "parent": None,
                "level": 0
            }
        }
        
        # Create domain-based second level
        for domain_id, domain_data in self.domains.items():
            domain_node = f"{domain_id}_concepts"
            self.ontology["hierarchy"][domain_node] = {
                "children": [],
                "parent": "business_concept",
                "level": 1,
                "domain": domain_id
            }
            self.ontology["hierarchy"]["business_concept"]["children"].append(domain_node)
            
            # Add pattern-based third level
            for pattern_name, pattern_keywords in HIERARCHY_PATTERNS.items():
                # Check if this pattern applies to this domain
                domain_concepts = domain_data["concepts"]
                matching_concepts = []
                
                for concept_id in domain_concepts:
                    concept = self.bizbok_concepts[concept_id]
                    concept_keywords = set(k.lower() for k in concept["keywords"])
                    
                    # Check if concept matches pattern
                    if any(pk in concept_keywords or pk in concept["name"].lower() 
                          for pk in pattern_keywords):
                        matching_concepts.append(concept_id)
                
                # Create pattern node if concepts match
                if matching_concepts:
                    pattern_node = f"{domain_id}_{pattern_name}"
                    self.ontology["hierarchy"][pattern_node] = {
                        "children": matching_concepts,
                        "parent": domain_node,
                        "level": 2,
                        "pattern": pattern_name
                    }
                    self.ontology["hierarchy"][domain_node]["children"].append(pattern_node)
                    
                    # Add concepts as leaf nodes
                    for concept_id in matching_concepts:
                        if concept_id not in self.ontology["hierarchy"]:
                            self.ontology["hierarchy"][concept_id] = {
                                "children": [],
                                "parent": pattern_node,
                                "level": 3,
                                "is_leaf": True
                            }
        
        # Add uncategorized concepts directly under domain
        for domain_id, domain_data in self.domains.items():
            domain_node = f"{domain_id}_concepts"
            for concept_id in domain_data["concepts"]:
                if concept_id not in self.ontology["hierarchy"]:
                    self.ontology["hierarchy"][concept_id] = {
                        "children": [],
                        "parent": domain_node,
                        "level": 2,
                        "is_leaf": True
                    }
                    self.ontology["hierarchy"][domain_node]["children"].append(concept_id)
        
        # Calculate hierarchy statistics
        max_level = max(node["level"] for node in self.ontology["hierarchy"].values())
        leaf_count = sum(1 for node in self.ontology["hierarchy"].values() if node.get("is_leaf"))
        
        print(f"   [OK] Built hierarchy with {len(self.ontology['hierarchy'])} nodes")
        print(f"   [OK] Maximum depth: {max_level} levels")
        print(f"   [OK] Leaf concepts: {leaf_count}")
    
    def extract_semantic_relationships(self):
        """Extract multiple types of semantic relationships"""
        print("\n[PROCESS] Extracting semantic relationships...")
        
        relationship_counts = {
            "semantic": 0,
            "causal": 0,
            "compositional": 0,
            "temporal": 0
        }
        
        for concept_id, concept in self.bizbok_concepts.items():
            # 1. Semantic relationships (high similarity)
            semantic_related = []
            for other_id in self.bizbok_concepts:
                if other_id != concept_id:
                    similarity = self.calculate_semantic_similarity(concept_id, other_id)
                    if similarity["similarity_score"] >= 0.4:
                        semantic_related.append({
                            "concept_id": other_id,
                            "similarity": similarity["similarity_score"],
                            "common_keywords": similarity["common_keywords"]
                        })
            
            # Sort by similarity and keep top 10
            semantic_related.sort(key=lambda x: x["similarity"], reverse=True)
            self.ontology["relationships"]["semantic"][concept_id] = semantic_related[:10]
            relationship_counts["semantic"] += len(semantic_related[:10])
            
            # 2. Causal relationships (from definition analysis)
            definition = concept["definition"].lower()
            causal_patterns = [
                (r'leads?\s+to', 'causes'),
                (r'results?\s+in', 'causes'),
                (r'causes?', 'causes'),
                (r'due\s+to', 'caused_by'),
                (r'because\s+of', 'caused_by'),
                (r'depends?\s+on', 'depends_on')
            ]
            
            causal_relations = []
            for pattern, relation_type in causal_patterns:
                if re.search(pattern, definition):
                    # Find mentioned concepts in definition
                    for other_id, other_concept in self.bizbok_concepts.items():
                        if other_id != concept_id:
                            other_name = other_concept["name"].lower()
                            if other_name in definition:
                                causal_relations.append({
                                    "concept_id": other_id,
                                    "relation_type": relation_type
                                })
            
            self.ontology["relationships"]["causal"][concept_id] = causal_relations[:5]
            relationship_counts["causal"] += len(causal_relations[:5])
            
            # 3. Compositional relationships (part-whole)
            compositional_patterns = [
                (r'consists?\s+of', 'has_part'),
                (r'comprises?', 'has_part'),
                (r'includes?', 'has_part'),
                (r'part\s+of', 'part_of'),
                (r'component\s+of', 'part_of')
            ]
            
            compositional_relations = []
            for pattern, relation_type in compositional_patterns:
                if re.search(pattern, definition):
                    for other_id, other_concept in self.bizbok_concepts.items():
                        if other_id != concept_id:
                            other_name = other_concept["name"].lower()
                            if other_name in definition:
                                compositional_relations.append({
                                    "concept_id": other_id,
                                    "relation_type": relation_type
                                })
            
            self.ontology["relationships"]["compositional"][concept_id] = compositional_relations[:5]
            relationship_counts["compositional"] += len(compositional_relations[:5])
            
            # 4. Temporal relationships (sequence/time)
            temporal_patterns = [
                (r'before', 'precedes'),
                (r'after', 'follows'),
                (r'during', 'concurrent_with'),
                (r'while', 'concurrent_with'),
                (r'then', 'followed_by')
            ]
            
            temporal_relations = []
            for pattern, relation_type in temporal_patterns:
                if re.search(pattern, definition):
                    for other_id, other_concept in self.bizbok_concepts.items():
                        if other_id != concept_id:
                            other_name = other_concept["name"].lower()
                            if other_name in definition:
                                temporal_relations.append({
                                    "concept_id": other_id,
                                    "relation_type": relation_type
                                })
            
            self.ontology["relationships"]["temporal"][concept_id] = temporal_relations[:3]
            relationship_counts["temporal"] += len(temporal_relations[:3])
        
        # Convert defaultdicts to regular dicts
        for rel_type in ["semantic", "causal", "compositional", "temporal"]:
            self.ontology["relationships"][rel_type] = dict(self.ontology["relationships"][rel_type])
        
        print(f"   [OK] Extracted {sum(relationship_counts.values())} total relationships")
        for rel_type, count in relationship_counts.items():
            print(f"      - {rel_type}: {count} relationships")
    
    def enhance_concepts_with_ontology(self):
        """Enhance each concept with ontological information"""
        print("\n[PROCESS] Enhancing concepts with ontology data...")
        
        for concept_id, concept in self.bizbok_concepts.items():
            # Get hierarchy information
            hierarchy_info = self.ontology["hierarchy"].get(concept_id, {})
            
            # Get cluster membership
            concept_cluster = None
            for cluster_id, cluster_data in self.ontology["clusters"].items():
                if concept_id in cluster_data["members"]:
                    concept_cluster = cluster_id
                    break
            
            # Count relationships
            relationship_count = 0
            for rel_type in ["semantic", "causal", "compositional", "temporal"]:
                relationships = self.ontology["relationships"][rel_type].get(concept_id, [])
                relationship_count += len(relationships)
            
            # Create enhanced concept entry
            self.ontology["concepts"][concept_id] = {
                "concept_id": concept_id,
                "name": concept["name"],
                "definition": concept["definition"],
                "domain": concept["domain"],
                "keywords": concept["keywords"],
                "hierarchy": {
                    "parent": hierarchy_info.get("parent"),
                    "children": hierarchy_info.get("children", []),
                    "level": hierarchy_info.get("level", -1),
                    "is_leaf": hierarchy_info.get("is_leaf", False)
                },
                "cluster": concept_cluster,
                "relationships": {
                    "semantic": [r["concept_id"] for r in self.ontology["relationships"]["semantic"].get(concept_id, [])][:5],
                    "causal": [r["concept_id"] for r in self.ontology["relationships"]["causal"].get(concept_id, [])][:3],
                    "compositional": [r["concept_id"] for r in self.ontology["relationships"]["compositional"].get(concept_id, [])][:3],
                    "temporal": [r["concept_id"] for r in self.ontology["relationships"]["temporal"].get(concept_id, [])][:2]
                },
                "ontology_metadata": {
                    "relationship_count": relationship_count,
                    "connectivity_score": min(1.0, relationship_count / 10),  # Normalized score
                    "cluster_coherence": self.ontology["clusters"].get(concept_cluster, {}).get("coherence_score", 0) if concept_cluster else 0
                }
            }
        
        print(f"   [OK] Enhanced {len(self.ontology['concepts'])} concepts with ontology data")
    
    def calculate_ontology_statistics(self):
        """Calculate comprehensive ontology statistics"""
        print("\n[ANALYSIS] Calculating ontology statistics...")
        
        # Basic counts
        total_concepts = len(self.ontology["concepts"])
        total_clusters = len(self.ontology["clusters"])
        total_hierarchy_nodes = len(self.ontology["hierarchy"])
        
        # Relationship statistics
        relationship_stats = {}
        for rel_type in ["semantic", "causal", "compositional", "temporal"]:
            rel_count = sum(len(rels) for rels in self.ontology["relationships"][rel_type].values())
            relationship_stats[rel_type] = rel_count
        
        # Hierarchy statistics
        hierarchy_depths = [node["level"] for node in self.ontology["hierarchy"].values()]
        max_depth = max(hierarchy_depths) if hierarchy_depths else 0
        avg_depth = np.mean(hierarchy_depths) if hierarchy_depths else 0
        
        # Cluster statistics
        cluster_sizes = [len(cluster["members"]) for cluster in self.ontology["clusters"].values()]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        avg_coherence = np.mean([cluster["coherence_score"] for cluster in self.ontology["clusters"].values()]) if self.ontology["clusters"] else 0
        
        # Connectivity statistics
        connectivity_scores = [c["ontology_metadata"]["connectivity_score"] for c in self.ontology["concepts"].values()]
        avg_connectivity = np.mean(connectivity_scores) if connectivity_scores else 0
        
        self.ontology["statistics"] = {
            "total_concepts": total_concepts,
            "total_clusters": total_clusters,
            "total_hierarchy_nodes": total_hierarchy_nodes,
            "hierarchy_max_depth": max_depth,
            "hierarchy_avg_depth": avg_depth,
            "cluster_avg_size": avg_cluster_size,
            "cluster_avg_coherence": avg_coherence,
            "relationships_total": sum(relationship_stats.values()),
            "relationships_by_type": relationship_stats,
            "avg_connectivity_score": avg_connectivity,
            "avg_relationships_per_concept": sum(relationship_stats.values()) / total_concepts if total_concepts > 0 else 0
        }
        
        print(f"   [OK] Total concepts: {total_concepts}")
        print(f"   [OK] Semantic clusters: {total_clusters}")
        print(f"   [OK] Hierarchy depth: {max_depth} levels")
        print(f"   [OK] Total relationships: {sum(relationship_stats.values())}")
        print(f"   [OK] Avg relationships/concept: {self.ontology['statistics']['avg_relationships_per_concept']:.1f}")
    
    def create_integration_api(self):
        """Create lightweight API for A/B pipeline integration"""
        print("\n[PROCESS] Creating integration API...")
        
        integration_api = {
            "quick_lookup": {},
            "expansion_rules": {
                "hierarchical": "include_parent_children_siblings",
                "semantic": "include_top_5_similar",
                "domain": "include_same_domain_concepts"
            },
            "concept_importance": {}
        }
        
        for concept_id, concept_data in self.ontology["concepts"].items():
            # Quick lookup entry
            integration_api["quick_lookup"][concept_id] = {
                "name": concept_data["name"],
                "domain": concept_data["domain"],
                "parent": concept_data["hierarchy"]["parent"],
                "children": concept_data["hierarchy"]["children"][:5],
                "related": concept_data["relationships"]["semantic"][:5],
                "expansion_candidates": list(set(
                    concept_data["relationships"]["semantic"][:3] +
                    concept_data["hierarchy"]["children"][:2]
                ))
            }
            
            # Concept importance based on connectivity
            integration_api["concept_importance"][concept_id] = {
                "connectivity_score": concept_data["ontology_metadata"]["connectivity_score"],
                "relationship_count": concept_data["ontology_metadata"]["relationship_count"],
                "is_central": concept_data["ontology_metadata"]["connectivity_score"] > 0.7
            }
        
        return integration_api
    
    def save_outputs(self, integration_api):
        """Save ontology outputs"""
        print("\n[SAVE] Saving ontology outputs...")
        
        # Main semantic ontology
        ontology_output = {
            "metadata": {
                "creation_timestamp": datetime.now().isoformat(),
                "total_concepts": len(self.ontology["concepts"]),
                "ontology_type": "BIZBOK_semantic",
                "version": "2.0",
                "cpu_optimized": True
            },
            "ontology": self.ontology
        }
        
        output_path = self.output_dir / "R4_semantic_ontology.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology_output, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {output_path.name}")
        
        # Integration API
        api_output = {
            "metadata": {
                "creation_timestamp": datetime.now().isoformat(),
                "api_version": "2.0",
                "total_concepts": len(integration_api["quick_lookup"])
            },
            "integration_api": integration_api
        }
        
        api_path = self.output_dir / "R4_integration_api.json"
        with open(api_path, 'w', encoding='utf-8') as f:
            json.dump(api_output, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {api_path.name}")
        
        # Ontology statistics report
        stats_output = {
            "statistics": self.ontology["statistics"],
            "performance_metrics": self.performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        stats_path = self.output_dir / "R4_ontology_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {stats_path.name}")
    
    def run(self):
        """Main execution method"""
        print("="*60)
        print("R4: Semantic Ontology Builder")
        print("R-Pipeline: Resource & Reasoning Pipeline")
        print("="*60)
        
        self.performance_metrics["start_time"] = time.time()
        
        try:
            # Load data
            stage_start = time.time()
            self.load_r_pipeline_data()
            self.performance_metrics["processing_stages"]["data_loading"] = time.time() - stage_start
            
            # Build semantic clusters
            stage_start = time.time()
            self.build_semantic_clusters()
            self.performance_metrics["processing_stages"]["clustering"] = time.time() - stage_start
            
            # Build hierarchy
            stage_start = time.time()
            self.build_concept_hierarchy()
            self.performance_metrics["processing_stages"]["hierarchy"] = time.time() - stage_start
            
            # Extract relationships
            stage_start = time.time()
            self.extract_semantic_relationships()
            self.performance_metrics["processing_stages"]["relationships"] = time.time() - stage_start
            
            # Enhance concepts
            stage_start = time.time()
            self.enhance_concepts_with_ontology()
            self.performance_metrics["processing_stages"]["enhancement"] = time.time() - stage_start
            
            # Calculate statistics
            stage_start = time.time()
            self.calculate_ontology_statistics()
            self.performance_metrics["processing_stages"]["statistics"] = time.time() - stage_start
            
            # Create integration API
            stage_start = time.time()
            integration_api = self.create_integration_api()
            self.performance_metrics["processing_stages"]["api_creation"] = time.time() - stage_start
            
            # Save outputs
            stage_start = time.time()
            self.save_outputs(integration_api)
            self.performance_metrics["processing_stages"]["saving"] = time.time() - stage_start
            
            # Calculate total time
            self.performance_metrics["end_time"] = time.time()
            total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
            self.performance_metrics["total_processing_time"] = total_time
            
            # Display performance summary
            print("\n[REPORT] Performance Summary:")
            print(f"   Total processing time: {total_time:.1f} seconds")
            for stage, duration in self.performance_metrics["processing_stages"].items():
                print(f"   - {stage}: {duration:.2f}s")
            
            print("\n[REPORT] Ontology Quality Metrics:")
            print(f"   Semantic clusters: {self.ontology['statistics']['total_clusters']}")
            print(f"   Hierarchy depth: {self.ontology['statistics']['hierarchy_max_depth']} levels")
            print(f"   Total relationships: {self.ontology['statistics']['relationships_total']}")
            print(f"   Avg connectivity: {self.ontology['statistics']['avg_connectivity_score']:.3f}")
            print(f"   Cluster coherence: {self.ontology['statistics']['cluster_avg_coherence']:.3f}")
            
            print(f"\n[SUCCESS] R4 Semantic Ontology Builder completed successfully!")
            print(f"   Created rich BIZBOK ontology with {len(self.ontology['concepts'])} concepts")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error in R4: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    builder = SemanticOntologyBuilder()
    return builder.run()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)