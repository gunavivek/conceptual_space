#!/usr/bin/env python3
"""
A37: Document-Chunk-Concept Pipeline Inspection
Enhanced inspection tool for analyzing chunk-concept mappings with advanced metrics
Includes: Affinity Score, Fidelity Score, Semantic Similarity, and Retrieval Weight
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Constants for scoring
AFFINITY_WEIGHTS = {
    'membership_strength': 0.4,
    'concept_density': 0.3,
    'semantic_coherence': 0.3
}

FIDELITY_WEIGHTS = {
    'coverage': 0.35,
    'precision': 0.35,
    'consistency': 0.3
}

RETRIEVAL_WEIGHTS = {
    'affinity': 0.3,
    'fidelity': 0.25,
    'semantic_similarity': 0.25,
    'concept_importance': 0.2
}

class A37_ChunkConceptInspector:
    """Advanced chunk-concept inspection with comprehensive metrics"""
    
    def __init__(self, outputs_dir: str = "../outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.data = None
        self.chunks = []
        self.concepts = {}
        self.metrics = {}
        
    def load_a3_output(self) -> Dict:
        """Load A3 chunking results"""
        a3_path = self.outputs_dir / "A3_concept_based_chunks.json"
        with open(a3_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Extract chunks and concepts
        for doc_id, doc_chunks in self.data['document_chunks'].items():
            self.chunks.extend(doc_chunks)
        
        self.concepts = self.data['concept_centroids']
        return self.data
    
    def calculate_affinity_score(self, chunk: Dict) -> Dict[str, float]:
        """
        Calculate Affinity Score: How naturally a chunk relates to its concepts
        
        Measures:
        - Membership strength: Average membership scores
        - Concept density: Number of concepts relative to chunk size
        - Semantic coherence: How well concepts relate to each other
        """
        memberships = chunk.get('concept_memberships', {})
        
        if not memberships:
            return {
                'affinity_score': 0.0,
                'membership_strength': 0.0,
                'concept_density': 0.0,
                'semantic_coherence': 0.0,
                'details': 'No concept memberships'
            }
        
        # Membership strength: Average of membership scores
        membership_values = list(memberships.values())
        membership_strength = np.mean(membership_values)
        
        # Concept density: Normalized by optimal range (3-7 concepts)
        num_concepts = len(memberships)
        if num_concepts <= 3:
            concept_density = num_concepts / 3
        elif num_concepts <= 7:
            concept_density = 1.0
        else:
            concept_density = 7 / num_concepts  # Penalty for too many concepts
        
        # Semantic coherence: How related are the concepts to each other
        semantic_coherence = self._calculate_semantic_coherence(list(memberships.keys()))
        
        # Weighted affinity score
        affinity_score = (
            AFFINITY_WEIGHTS['membership_strength'] * membership_strength +
            AFFINITY_WEIGHTS['concept_density'] * concept_density +
            AFFINITY_WEIGHTS['semantic_coherence'] * semantic_coherence
        )
        
        return {
            'affinity_score': affinity_score,
            'membership_strength': membership_strength,
            'concept_density': concept_density,
            'semantic_coherence': semantic_coherence,
            'num_concepts': num_concepts
        }
    
    def calculate_fidelity_score(self, chunk: Dict) -> Dict[str, float]:
        """
        Calculate Fidelity Score: How accurately a chunk represents its concepts
        
        Measures:
        - Coverage: How many concept aspects are covered
        - Precision: How focused the chunk is on its concepts
        - Consistency: How uniform the membership scores are
        """
        memberships = chunk.get('concept_memberships', {})
        
        if not memberships:
            return {
                'fidelity_score': 0.0,
                'coverage': 0.0,
                'precision': 0.0,
                'consistency': 0.0,
                'details': 'No concept memberships'
            }
        
        membership_values = list(memberships.values())
        
        # Coverage: Proportion of strong memberships (>0.5)
        strong_memberships = [v for v in membership_values if v > 0.5]
        coverage = len(strong_memberships) / len(membership_values) if membership_values else 0
        
        # Precision: Ratio of high-confidence memberships
        max_membership = max(membership_values)
        precision = sum(v/max_membership for v in membership_values if v > 0.3) / len(membership_values)
        
        # Consistency: 1 - coefficient of variation (lower CV = higher consistency)
        if len(membership_values) > 1:
            std_dev = np.std(membership_values)
            mean_val = np.mean(membership_values)
            cv = std_dev / mean_val if mean_val > 0 else 1
            consistency = 1 / (1 + cv)  # Transform to 0-1 scale
        else:
            consistency = 1.0
        
        # Weighted fidelity score
        fidelity_score = (
            FIDELITY_WEIGHTS['coverage'] * coverage +
            FIDELITY_WEIGHTS['precision'] * precision +
            FIDELITY_WEIGHTS['consistency'] * consistency
        )
        
        return {
            'fidelity_score': fidelity_score,
            'coverage': coverage,
            'precision': precision,
            'consistency': consistency,
            'strong_memberships': len(strong_memberships),
            'total_memberships': len(membership_values)
        }
    
    def calculate_semantic_similarity(self, chunk: Dict) -> Dict[str, float]:
        """
        Calculate Semantic Similarity between chunk and its concepts
        Using cosine similarity of embeddings (simulated with membership scores)
        """
        memberships = chunk.get('concept_memberships', {})
        
        if not memberships:
            return {
                'semantic_similarity': 0.0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'details': 'No concept memberships'
            }
        
        # Use membership scores as proxy for semantic similarity
        similarity_scores = list(memberships.values())
        
        # Calculate statistics
        avg_similarity = np.mean(similarity_scores)
        max_similarity = max(similarity_scores)
        min_similarity = min(similarity_scores)
        
        # Weighted semantic similarity (emphasize high similarities)
        weights = np.array(similarity_scores) ** 2  # Square to emphasize high scores
        weighted_similarity = np.average(similarity_scores, weights=weights)
        
        return {
            'semantic_similarity': weighted_similarity,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'similarity_range': max_similarity - min_similarity
        }
    
    def calculate_retrieval_weight(self, chunk: Dict, chunk_metrics: Dict) -> Dict[str, float]:
        """
        Calculate Retrieval Weight: Priority score for retrieval ranking
        Combines all metrics to produce a final retrieval priority
        """
        # Get component scores
        affinity = chunk_metrics.get('affinity', {}).get('affinity_score', 0)
        fidelity = chunk_metrics.get('fidelity', {}).get('fidelity_score', 0)
        semantic = chunk_metrics.get('semantic', {}).get('semantic_similarity', 0)
        
        # Calculate concept importance (based on frequency and distribution)
        concept_importance = self._calculate_concept_importance(chunk)
        
        # Weighted retrieval score
        retrieval_weight = (
            RETRIEVAL_WEIGHTS['affinity'] * affinity +
            RETRIEVAL_WEIGHTS['fidelity'] * fidelity +
            RETRIEVAL_WEIGHTS['semantic_similarity'] * semantic +
            RETRIEVAL_WEIGHTS['concept_importance'] * concept_importance
        )
        
        # Determine retrieval priority category
        if retrieval_weight >= 0.8:
            priority = 'critical'
        elif retrieval_weight >= 0.6:
            priority = 'high'
        elif retrieval_weight >= 0.4:
            priority = 'medium'
        else:
            priority = 'low'
        
        return {
            'retrieval_weight': retrieval_weight,
            'priority': priority,
            'components': {
                'affinity_contribution': RETRIEVAL_WEIGHTS['affinity'] * affinity,
                'fidelity_contribution': RETRIEVAL_WEIGHTS['fidelity'] * fidelity,
                'semantic_contribution': RETRIEVAL_WEIGHTS['semantic_similarity'] * semantic,
                'importance_contribution': RETRIEVAL_WEIGHTS['concept_importance'] * concept_importance
            }
        }
    
    def _calculate_semantic_coherence(self, concept_ids: List[str]) -> float:
        """Calculate how semantically coherent a set of concepts are"""
        if len(concept_ids) <= 1:
            return 1.0
        
        # Check for concept families (similar prefixes)
        prefixes = set()
        for cid in concept_ids:
            if '_' in cid:
                prefix = cid.split('_')[0]
                prefixes.add(prefix)
        
        # Higher coherence if concepts share prefixes (same family)
        coherence = 1.0 - (len(prefixes) - 1) / len(concept_ids)
        return max(0.0, coherence)
    
    def _calculate_concept_importance(self, chunk: Dict) -> float:
        """Calculate importance of concepts in a chunk"""
        memberships = chunk.get('concept_memberships', {})
        if not memberships:
            return 0.0
        
        importance_scores = []
        for concept_id in memberships:
            # Check if it's a core concept (higher importance)
            if concept_id.startswith('core_'):
                importance_scores.append(1.0)
            # A2.5 generated concepts have medium importance
            elif concept_id.startswith('a25'):
                importance_scores.append(0.7)
            else:
                importance_scores.append(0.5)
        
        return np.mean(importance_scores)
    
    def analyze_all_chunks(self) -> Dict:
        """Perform comprehensive analysis of all chunks"""
        if not self.chunks:
            self.load_a3_output()
        
        all_metrics = []
        
        for chunk in self.chunks:
            chunk_id = chunk.get('chunk_id', 'unknown')
            
            # Calculate all metrics
            affinity = self.calculate_affinity_score(chunk)
            fidelity = self.calculate_fidelity_score(chunk)
            semantic = self.calculate_semantic_similarity(chunk)
            
            chunk_metrics = {
                'affinity': affinity,
                'fidelity': fidelity,
                'semantic': semantic
            }
            
            retrieval = self.calculate_retrieval_weight(chunk, chunk_metrics)
            
            # Compile metrics
            metrics = {
                'chunk_id': chunk_id,
                'chunk_type': chunk.get('chunk_type', 'unknown'),
                'affinity_score': affinity['affinity_score'],
                'fidelity_score': fidelity['fidelity_score'],
                'semantic_similarity': semantic['semantic_similarity'],
                'retrieval_weight': retrieval['retrieval_weight'],
                'retrieval_priority': retrieval['priority'],
                'num_concepts': affinity['num_concepts'],
                'detailed_metrics': chunk_metrics,
                'retrieval_details': retrieval
            }
            
            all_metrics.append(metrics)
        
        self.metrics = all_metrics
        return self._generate_summary_statistics(all_metrics)
    
    def _generate_summary_statistics(self, metrics: List[Dict]) -> Dict:
        """Generate summary statistics from all chunk metrics"""
        
        # Extract scores
        affinity_scores = [m['affinity_score'] for m in metrics]
        fidelity_scores = [m['fidelity_score'] for m in metrics]
        semantic_scores = [m['semantic_similarity'] for m in metrics]
        retrieval_weights = [m['retrieval_weight'] for m in metrics]
        
        # Priority distribution
        priority_dist = Counter(m['retrieval_priority'] for m in metrics)
        
        # Chunk type analysis
        chunk_types = defaultdict(list)
        for m in metrics:
            chunk_types[m['chunk_type']].append({
                'affinity': m['affinity_score'],
                'fidelity': m['fidelity_score'],
                'semantic': m['semantic_similarity'],
                'retrieval': m['retrieval_weight']
            })
        
        type_averages = {}
        for ctype, scores in chunk_types.items():
            type_averages[ctype] = {
                'avg_affinity': np.mean([s['affinity'] for s in scores]),
                'avg_fidelity': np.mean([s['fidelity'] for s in scores]),
                'avg_semantic': np.mean([s['semantic'] for s in scores]),
                'avg_retrieval': np.mean([s['retrieval'] for s in scores]),
                'count': len(scores)
            }
        
        return {
            'total_chunks': len(metrics),
            'average_scores': {
                'affinity': np.mean(affinity_scores),
                'fidelity': np.mean(fidelity_scores),
                'semantic_similarity': np.mean(semantic_scores),
                'retrieval_weight': np.mean(retrieval_weights)
            },
            'score_ranges': {
                'affinity': (min(affinity_scores), max(affinity_scores)),
                'fidelity': (min(fidelity_scores), max(fidelity_scores)),
                'semantic': (min(semantic_scores), max(semantic_scores)),
                'retrieval': (min(retrieval_weights), max(retrieval_weights))
            },
            'retrieval_priority_distribution': dict(priority_dist),
            'chunk_type_analysis': type_averages,
            'top_chunks': self._get_top_chunks(metrics, 5),
            'problematic_chunks': self._identify_problematic_chunks(metrics)
        }
    
    def _get_top_chunks(self, metrics: List[Dict], n: int = 5) -> List[Dict]:
        """Get top n chunks by retrieval weight"""
        sorted_metrics = sorted(metrics, key=lambda x: x['retrieval_weight'], reverse=True)
        return [
            {
                'chunk_id': m['chunk_id'],
                'retrieval_weight': m['retrieval_weight'],
                'priority': m['retrieval_priority'],
                'scores': {
                    'affinity': m['affinity_score'],
                    'fidelity': m['fidelity_score'],
                    'semantic': m['semantic_similarity']
                }
            }
            for m in sorted_metrics[:n]
        ]
    
    def _identify_problematic_chunks(self, metrics: List[Dict]) -> List[Dict]:
        """Identify chunks with potential issues"""
        problematic = []
        
        for m in metrics:
            issues = []
            
            if m['affinity_score'] < 0.3:
                issues.append(f"Low affinity: {m['affinity_score']:.3f}")
            
            if m['fidelity_score'] < 0.3:
                issues.append(f"Low fidelity: {m['fidelity_score']:.3f}")
            
            if m['semantic_similarity'] < 0.3:
                issues.append(f"Low semantic similarity: {m['semantic_similarity']:.3f}")
            
            if m['retrieval_weight'] < 0.2:
                issues.append(f"Very low retrieval weight: {m['retrieval_weight']:.3f}")
            
            if issues:
                problematic.append({
                    'chunk_id': m['chunk_id'],
                    'issues': issues,
                    'retrieval_priority': m['retrieval_priority']
                })
        
        return problematic
    
    def generate_detailed_report(self) -> None:
        """Generate comprehensive inspection report"""
        if not self.metrics:
            summary = self.analyze_all_chunks()
        else:
            summary = self._generate_summary_statistics(self.metrics)
        
        print("="*80)
        print("A37: DOCUMENT-CHUNK-CONCEPT PIPELINE INSPECTION REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Overall Statistics
        print("[OVERALL STATISTICS]")
        print("-"*40)
        print(f"Total Chunks Analyzed: {summary['total_chunks']}")
        print()
        
        # Average Scores
        print("[AVERAGE METRIC SCORES]")
        print("-"*40)
        for metric, score in summary['average_scores'].items():
            print(f"  {metric.replace('_', ' ').title()}: {score:.3f}")
        print()
        
        # Score Ranges
        print("[SCORE RANGES]")
        print("-"*40)
        for metric, (min_val, max_val) in summary['score_ranges'].items():
            print(f"  {metric.replace('_', ' ').title()}: {min_val:.3f} - {max_val:.3f}")
        print()
        
        # Retrieval Priority Distribution
        print("[RETRIEVAL PRIORITY DISTRIBUTION]")
        print("-"*40)
        total = summary['total_chunks']
        for priority in ['critical', 'high', 'medium', 'low']:
            count = summary['retrieval_priority_distribution'].get(priority, 0)
            percentage = (count / total * 100) if total > 0 else 0
            bar = '=' * int(percentage / 2)
            print(f"  {priority.upper():8}: {count:3} ({percentage:5.1f}%) {bar}")
        print()
        
        # Chunk Type Analysis
        print("[CHUNK TYPE PERFORMANCE]")
        print("-"*40)
        for ctype, stats in summary['chunk_type_analysis'].items():
            print(f"\n  {ctype.upper()} ({stats['count']} chunks):")
            print(f"    Affinity: {stats['avg_affinity']:.3f}")
            print(f"    Fidelity: {stats['avg_fidelity']:.3f}")
            print(f"    Semantic: {stats['avg_semantic']:.3f}")
            print(f"    Retrieval: {stats['avg_retrieval']:.3f}")
        print()
        
        # Top Performing Chunks
        print("[TOP 5 CHUNKS BY RETRIEVAL WEIGHT]")
        print("-"*40)
        for i, chunk in enumerate(summary['top_chunks'], 1):
            print(f"  {i}. {chunk['chunk_id']}")
            print(f"     Weight: {chunk['retrieval_weight']:.3f} ({chunk['priority']})")
            print(f"     A:{chunk['scores']['affinity']:.2f} F:{chunk['scores']['fidelity']:.2f} S:{chunk['scores']['semantic']:.2f}")
        print()
        
        # Problematic Chunks
        if summary['problematic_chunks']:
            print("[PROBLEMATIC CHUNKS]")
            print("-"*40)
            for chunk in summary['problematic_chunks'][:5]:
                print(f"  {chunk['chunk_id']} ({chunk['retrieval_priority']})")
                for issue in chunk['issues']:
                    print(f"    - {issue}")
            if len(summary['problematic_chunks']) > 5:
                print(f"  ... and {len(summary['problematic_chunks']) - 5} more")
            print()
        
        # System Health
        print("[SYSTEM HEALTH CHECK]")
        print("-"*40)
        avg_retrieval = summary['average_scores']['retrieval_weight']
        health_checks = [
            ("Average retrieval weight > 0.5", avg_retrieval > 0.5),
            ("Critical priority chunks exist", summary['retrieval_priority_distribution'].get('critical', 0) > 0),
            ("Low priority chunks < 50%", summary['retrieval_priority_distribution'].get('low', 0) < total/2),
            ("Average affinity > 0.4", summary['average_scores']['affinity'] > 0.4),
            ("Average fidelity > 0.4", summary['average_scores']['fidelity'] > 0.4)
        ]
        
        for check, passed in health_checks:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} {check}")
        
        print()
        print("="*80)
        print("[INSPECTION COMPLETE]")
        print("="*80)
    
    def export_metrics_to_csv(self, filename: str = "A37_chunk_concept_metrics.csv") -> None:
        """Export detailed metrics to CSV"""
        if not self.metrics:
            self.analyze_all_chunks()
        
        # Ensure output goes to A-Pipeline outputs directory
        output_path = self.outputs_dir / filename
        
        # Flatten metrics for CSV
        flat_metrics = []
        for m in self.metrics:
            flat = {
                'chunk_id': m['chunk_id'],
                'chunk_type': m['chunk_type'],
                'affinity_score': m['affinity_score'],
                'fidelity_score': m['fidelity_score'],
                'semantic_similarity': m['semantic_similarity'],
                'retrieval_weight': m['retrieval_weight'],
                'retrieval_priority': m['retrieval_priority'],
                'num_concepts': m['num_concepts']
            }
            
            # Add detailed sub-metrics
            if 'detailed_metrics' in m:
                flat['membership_strength'] = m['detailed_metrics']['affinity']['membership_strength']
                flat['concept_density'] = m['detailed_metrics']['affinity']['concept_density']
                flat['semantic_coherence'] = m['detailed_metrics']['affinity']['semantic_coherence']
                flat['coverage'] = m['detailed_metrics']['fidelity']['coverage']
                flat['precision'] = m['detailed_metrics']['fidelity']['precision']
                flat['consistency'] = m['detailed_metrics']['fidelity']['consistency']
            
            flat_metrics.append(flat)
        
        df = pd.DataFrame(flat_metrics)
        df.to_csv(output_path, index=False)
        print(f"[EXPORTED] Metrics saved to {output_path}")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("A37: DOCUMENT-CHUNK-CONCEPT PIPELINE INSPECTION")
    print("Advanced Metrics: Affinity, Fidelity, Semantic Similarity, Retrieval Weight")
    print("="*80)
    
    # Initialize inspector
    inspector = A37_ChunkConceptInspector()
    
    print("\n[LOADING] Loading A3 chunking output...")
    inspector.load_a3_output()
    print(f"[LOADED] {len(inspector.chunks)} chunks, {len(inspector.concepts)} concepts")
    
    print("\n[ANALYZING] Computing advanced metrics...")
    summary = inspector.analyze_all_chunks()
    
    print("\n[GENERATING] Creating detailed report...")
    inspector.generate_detailed_report()
    
    print("\n[EXPORTING] Saving metrics to CSV...")
    inspector.export_metrics_to_csv()
    
    return inspector

if __name__ == "__main__":
    inspector = main()