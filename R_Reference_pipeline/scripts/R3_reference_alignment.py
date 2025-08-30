#!/usr/bin/env python3
"""
R3: Reference Alignment
Part of R-Pipeline (Resource & Reasoning Pipeline)
Aligns pipeline concepts with BIZBOK reference standards and creates 
alignment mappings for improved concept consistency and standardization
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import time

class ReferenceAligner:
    """Main class for creating reference alignments"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.parent
        self.output_dir = self.script_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_data = {}
        self.bizbok_concepts = {}
        self.alignment_mappings = {}
        self.standardized_concepts = {}
        self.quality_analysis = {}
    
    def load_validation_results(self):
        """Load concept validation results from R2"""
        validation_path = self.output_dir / "R2_validation_report.json"
        
        if not validation_path.exists():
            raise FileNotFoundError(f"R2 validation report not found: {validation_path}")
        
        print(f"[DATA] Loading validation results from R2...")
        with open(validation_path, 'r', encoding='utf-8') as f:
            self.validation_data = json.load(f)
        
        validation_count = len(self.validation_data.get("validation_results", {}))
        print(f"   [OK] Loaded {validation_count} validation results")
        
        return self.validation_data
    
    def load_bizbok_concepts(self):
        """Load BIZBOK concepts from R1"""
        concepts_path = self.output_dir / "R1_CONCEPTS.json"
        
        if not concepts_path.exists():
            raise FileNotFoundError(f"R1 concepts not found: {concepts_path}")
        
        print(f"[DATA] Loading BIZBOK concepts from R1...")
        with open(concepts_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.bizbok_concepts = data["concepts"]
        
        print(f"   [OK] Loaded {len(self.bizbok_concepts)} BIZBOK concepts")
        return self.bizbok_concepts
    
    def create_alignment_mappings(self):
        """Create alignment mappings between pipeline and BIZBOK concepts"""
        print("\n[PROCESS] Creating alignment mappings...")
        
        self.alignment_mappings = {
            "direct_alignments": {},      # High-confidence alignments (>= 0.7)
            "suggested_alignments": {},   # Medium-confidence alignments (0.4-0.7)
            "custom_concepts": {},         # No alignment found
            "term_unifications": defaultdict(set),  # Unified terminology
            "alignment_statistics": {
                "total_processed": 0,
                "direct_count": 0,
                "suggested_count": 0,
                "custom_count": 0
            }
        }
        
        validation_results = self.validation_data.get("validation_results", {})
        
        for pipeline_id, validation in validation_results.items():
            validation_score = validation["validation_score"]
            self.alignment_mappings["alignment_statistics"]["total_processed"] += 1
            
            if validation_score >= 0.7:
                # High-confidence direct alignment
                self.create_direct_alignment(pipeline_id, validation)
                self.alignment_mappings["alignment_statistics"]["direct_count"] += 1
                
            elif validation_score >= 0.4:
                # Medium-confidence suggested alignment
                self.create_suggested_alignment(pipeline_id, validation)
                self.alignment_mappings["alignment_statistics"]["suggested_count"] += 1
                
            else:
                # No good alignment - custom concept
                self.create_custom_concept(pipeline_id, validation)
                self.alignment_mappings["alignment_statistics"]["custom_count"] += 1
        
        # Convert term unifications to lists
        self.alignment_mappings["term_unifications"] = {
            term: list(concepts) 
            for term, concepts in self.alignment_mappings["term_unifications"].items()
        }
        
        print(f"   [OK] Created {self.alignment_mappings['alignment_statistics']['direct_count']} direct alignments")
        print(f"   [OK] Created {self.alignment_mappings['alignment_statistics']['suggested_count']} suggested alignments")
        print(f"   [OK] Created {self.alignment_mappings['alignment_statistics']['custom_count']} custom concepts")
    
    def create_direct_alignment(self, pipeline_id, validation):
        """Create high-confidence direct alignment"""
        best_match = validation["best_match"]
        bizbok_id = best_match["bizbok_id"]
        bizbok_concept = self.bizbok_concepts[bizbok_id]
        
        self.alignment_mappings["direct_alignments"][pipeline_id] = {
            "bizbok_id": bizbok_id,
            "bizbok_name": bizbok_concept["name"],
            "alignment_confidence": validation["validation_score"],
            "standardized_definition": bizbok_concept["definition"],
            "standardized_terms": bizbok_concept["keywords"],
            "alignment_type": "direct",
            "domain": bizbok_concept["domain"]
        }
        
        # Add to term unifications
        similarity = best_match["similarity_analysis"]
        common_terms = similarity.get("common_keywords", [])
        for term in common_terms:
            self.alignment_mappings["term_unifications"][term].add(pipeline_id)
            self.alignment_mappings["term_unifications"][term].add(bizbok_id)
    
    def create_suggested_alignment(self, pipeline_id, validation):
        """Create medium-confidence suggested alignment"""
        best_match = validation["best_match"]
        bizbok_id = best_match["bizbok_id"]
        bizbok_concept = self.bizbok_concepts[bizbok_id]
        
        self.alignment_mappings["suggested_alignments"][pipeline_id] = {
            "bizbok_id": bizbok_id,
            "bizbok_name": bizbok_concept["name"],
            "alignment_confidence": validation["validation_score"],
            "suggested_definition": bizbok_concept["definition"],
            "suggested_terms": bizbok_concept["keywords"],
            "alignment_type": "suggested",
            "domain": bizbok_concept["domain"],
            "review_required": True,
            "confidence_reason": "Medium similarity - manual review recommended"
        }
    
    def create_custom_concept(self, pipeline_id, validation):
        """Create custom concept entry for unaligned concepts"""
        pipeline_concept = validation["pipeline_concept"]
        
        self.alignment_mappings["custom_concepts"][pipeline_id] = {
            "alignment_type": "custom",
            "alignment_confidence": 0.0,
            "original_name": self.get_concept_name(pipeline_concept),
            "domain": self.get_concept_domain(pipeline_concept),
            "reason": "No suitable BIZBOK alignment found",
            "recommendation": "Consider adding to BIZBOK resources"
        }
    
    def generate_standardized_concepts(self):
        """Generate standardized concept definitions based on alignments"""
        print("\n[PROCESS] Generating standardized concepts...")
        
        validation_results = self.validation_data.get("validation_results", {})
        
        for pipeline_id, validation in validation_results.items():
            pipeline_concept = validation["pipeline_concept"]
            
            # Check alignment type
            if pipeline_id in self.alignment_mappings["direct_alignments"]:
                # Direct alignment - use BIZBOK standard
                self.create_standardized_from_direct(pipeline_id, pipeline_concept)
                
            elif pipeline_id in self.alignment_mappings["suggested_alignments"]:
                # Suggested alignment - blend with BIZBOK
                self.create_standardized_from_suggested(pipeline_id, pipeline_concept)
                
            else:
                # Custom concept - preserve original
                self.create_standardized_custom(pipeline_id, pipeline_concept)
        
        print(f"   [OK] Generated {len(self.standardized_concepts)} standardized concepts")
    
    def create_standardized_from_direct(self, pipeline_id, pipeline_concept):
        """Create standardized concept from direct alignment"""
        alignment = self.alignment_mappings["direct_alignments"][pipeline_id]
        
        self.standardized_concepts[pipeline_id] = {
            "concept_id": pipeline_id,
            "standardized_name": alignment["bizbok_name"],
            "original_name": self.get_concept_name(pipeline_concept),
            "standardized_definition": alignment["standardized_definition"],
            "standardized_terms": alignment["standardized_terms"],
            "domain": alignment["domain"],
            "alignment_confidence": alignment["alignment_confidence"],
            "source": "bizbok_aligned",
            "bizbok_id": alignment["bizbok_id"],
            "standardization_type": "direct"
        }
        
        # Merge original terms with standardized terms
        original_terms = self.get_concept_terms(pipeline_concept)
        merged_terms = list(set(original_terms + alignment["standardized_terms"]))
        self.standardized_concepts[pipeline_id]["merged_terms"] = merged_terms[:30]  # Limit size
    
    def create_standardized_from_suggested(self, pipeline_id, pipeline_concept):
        """Create standardized concept from suggested alignment"""
        alignment = self.alignment_mappings["suggested_alignments"][pipeline_id]
        
        self.standardized_concepts[pipeline_id] = {
            "concept_id": pipeline_id,
            "standardized_name": alignment["bizbok_name"],
            "original_name": self.get_concept_name(pipeline_concept),
            "standardized_definition": alignment["suggested_definition"],
            "standardized_terms": alignment["suggested_terms"],
            "domain": alignment["domain"],
            "alignment_confidence": alignment["alignment_confidence"],
            "source": "bizbok_suggested",
            "bizbok_id": alignment["bizbok_id"],
            "standardization_type": "suggested",
            "review_required": True
        }
        
        # Blend terms
        original_terms = self.get_concept_terms(pipeline_concept)
        blended_terms = list(set(original_terms + alignment["suggested_terms"][:10]))
        self.standardized_concepts[pipeline_id]["merged_terms"] = blended_terms[:30]
    
    def create_standardized_custom(self, pipeline_id, pipeline_concept):
        """Create standardized concept for custom/unaligned concepts"""
        concept_name = self.get_concept_name(pipeline_concept)
        
        self.standardized_concepts[pipeline_id] = {
            "concept_id": pipeline_id,
            "standardized_name": concept_name,
            "original_name": concept_name,
            "standardized_definition": f"Custom concept: {concept_name}",
            "standardized_terms": self.get_concept_terms(pipeline_concept),
            "domain": self.get_concept_domain(pipeline_concept),
            "alignment_confidence": 0.0,
            "source": "pipeline_custom",
            "bizbok_id": None,
            "standardization_type": "custom",
            "custom_standardization": True
        }
    
    def get_concept_name(self, concept):
        """Extract concept name from various concept formats"""
        if "original_concept" in concept:
            return concept["original_concept"].get("theme_name", "Unknown")
        return concept.get("theme_name", concept.get("name", "Unknown"))
    
    def get_concept_domain(self, concept):
        """Extract concept domain from various concept formats"""
        if "original_concept" in concept:
            return concept["original_concept"].get("domain", "general")
        return concept.get("domain", "general")
    
    def get_concept_terms(self, concept):
        """Extract concept terms from various concept formats"""
        terms = []
        if "original_concept" in concept:
            terms.extend(concept["original_concept"].get("primary_keywords", []))
            terms.extend(concept.get("all_expanded_terms", []))
        else:
            terms.extend(concept.get("primary_keywords", []))
            terms.extend(concept.get("keywords", []))
        return list(set(terms))[:20]  # Limit and deduplicate
    
    def analyze_alignment_quality(self):
        """Analyze quality of alignment results"""
        print("\n[ANALYSIS] Analyzing alignment quality...")
        
        total_concepts = len(self.standardized_concepts)
        direct_alignments = self.alignment_mappings["alignment_statistics"]["direct_count"]
        suggested_alignments = self.alignment_mappings["alignment_statistics"]["suggested_count"]
        custom_concepts = self.alignment_mappings["alignment_statistics"]["custom_count"]
        
        # Domain alignment analysis
        domain_alignment_quality = defaultdict(list)
        for concept in self.standardized_concepts.values():
            domain = concept["domain"]
            confidence = concept["alignment_confidence"]
            domain_alignment_quality[domain].append(confidence)
        
        domain_averages = {}
        for domain, confidences in domain_alignment_quality.items():
            domain_averages[domain] = np.mean(confidences) if confidences else 0.0
        
        # Term unification analysis
        unified_terms = len(self.alignment_mappings["term_unifications"])
        avg_concepts_per_term = 0
        if unified_terms > 0:
            concepts_per_term = [len(concepts) for concepts in self.alignment_mappings["term_unifications"].values()]
            avg_concepts_per_term = np.mean(concepts_per_term)
        
        self.quality_analysis = {
            "total_concepts": total_concepts,
            "alignment_distribution": {
                "direct_alignments": direct_alignments,
                "suggested_alignments": suggested_alignments,
                "custom_concepts": custom_concepts
            },
            "alignment_percentages": {
                "direct": (direct_alignments / total_concepts * 100) if total_concepts > 0 else 0,
                "suggested": (suggested_alignments / total_concepts * 100) if total_concepts > 0 else 0,
                "custom": (custom_concepts / total_concepts * 100) if total_concepts > 0 else 0
            },
            "domain_alignment_quality": domain_averages,
            "terminology_unification": {
                "unified_terms": unified_terms,
                "avg_concepts_per_term": avg_concepts_per_term
            },
            "overall_alignment_rate": ((direct_alignments + suggested_alignments) / total_concepts * 100) if total_concepts > 0 else 0
        }
        
        print(f"   [OK] Overall alignment rate: {self.quality_analysis['overall_alignment_rate']:.1f}%")
    
    def generate_recommendations(self):
        """Generate recommendations for improving alignment"""
        recommendations = []
        
        # Overall alignment rate recommendations
        alignment_rate = self.quality_analysis["overall_alignment_rate"]
        if alignment_rate < 70:
            recommendations.append({
                "type": "alignment_rate",
                "priority": "high",
                "message": f"Low alignment rate ({alignment_rate:.1f}%) - review concept extraction to better match BIZBOK standards"
            })
        
        # Custom concept recommendations
        custom_percentage = self.quality_analysis["alignment_percentages"]["custom"]
        if custom_percentage > 40:
            recommendations.append({
                "type": "custom_concepts",
                "priority": "medium",
                "message": f"High custom concepts ({custom_percentage:.1f}%) - consider expanding BIZBOK resources"
            })
        
        # Domain-specific recommendations
        for domain, avg_confidence in self.quality_analysis["domain_alignment_quality"].items():
            if avg_confidence < 0.5:
                recommendations.append({
                    "type": "domain_quality",
                    "priority": "medium",
                    "message": f"Poor {domain} alignment quality ({avg_confidence:.2f}) - strengthen {domain} BIZBOK concepts"
                })
        
        # Suggested alignment recommendations
        suggested_count = self.quality_analysis["alignment_distribution"]["suggested_alignments"]
        if suggested_count > 5:
            recommendations.append({
                "type": "review_required",
                "priority": "low",
                "message": f"{suggested_count} concepts need manual review for alignment confirmation"
            })
        
        return recommendations
    
    def create_alignment_export(self):
        """Create exportable alignment data for use in other systems"""
        export_data = {
            "concept_dictionary": {},
            "term_mappings": {},
            "domain_hierarchies": defaultdict(list),
            "alignment_registry": {}
        }
        
        # Create concept dictionary
        for concept_id, concept in self.standardized_concepts.items():
            export_data["concept_dictionary"][concept_id] = {
                "name": concept["standardized_name"],
                "definition": concept["standardized_definition"],
                "terms": concept.get("merged_terms", concept["standardized_terms"]),
                "domain": concept["domain"],
                "source": concept["source"]
            }
            
            # Add to domain hierarchy
            export_data["domain_hierarchies"][concept["domain"]].append(concept_id)
        
        # Create term mappings
        for term, concept_ids in self.alignment_mappings["term_unifications"].items():
            export_data["term_mappings"][term] = concept_ids
        
        # Create alignment registry
        for pipeline_id, alignment in self.alignment_mappings["direct_alignments"].items():
            export_data["alignment_registry"][pipeline_id] = {
                "type": "direct",
                "bizbok_id": alignment["bizbok_id"],
                "confidence": alignment["alignment_confidence"]
            }
        
        for pipeline_id, alignment in self.alignment_mappings["suggested_alignments"].items():
            export_data["alignment_registry"][pipeline_id] = {
                "type": "suggested",
                "bizbok_id": alignment["bizbok_id"],
                "confidence": alignment["alignment_confidence"]
            }
        
        # Convert defaultdict to dict
        export_data["domain_hierarchies"] = dict(export_data["domain_hierarchies"])
        
        return export_data
    
    def save_outputs(self, recommendations, export_data):
        """Save alignment results and analysis"""
        print("\n[SAVE] Saving alignment outputs...")
        
        # Main alignment report
        report_data = {
            "metadata": {
                "alignment_timestamp": datetime.now().isoformat(),
                "total_concepts_aligned": len(self.standardized_concepts),
                "alignment_method": "BIZBOK_similarity_matching",
                "version": "2.0"
            },
            "alignment_mappings": self.alignment_mappings,
            "standardized_concepts": self.standardized_concepts,
            "quality_analysis": self.quality_analysis,
            "recommendations": recommendations
        }
        
        output_path = self.output_dir / "R3_alignment_mappings.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {output_path.name}")
        
        # Export data for other systems
        export_path = self.output_dir / "R3_alignment_export.json"
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {export_path.name}")
    
    def run(self):
        """Main execution method"""
        print("="*60)
        print("R3: Reference Alignment")
        print("R-Pipeline: Resource & Reasoning Pipeline")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Load data
            self.load_validation_results()
            self.load_bizbok_concepts()
            
            # Create alignments
            self.create_alignment_mappings()
            
            # Generate standardized concepts
            self.generate_standardized_concepts()
            
            # Analyze quality
            self.analyze_alignment_quality()
            
            # Generate recommendations
            recommendations = self.generate_recommendations()
            
            # Create export data
            export_data = self.create_alignment_export()
            
            # Display results
            print("\n[ANALYSIS] Alignment Results:")
            print(f"   Total Concepts: {self.quality_analysis['total_concepts']}")
            print(f"   Direct Alignments: {self.quality_analysis['alignment_distribution']['direct_alignments']} ({self.quality_analysis['alignment_percentages']['direct']:.1f}%)")
            print(f"   Suggested Alignments: {self.quality_analysis['alignment_distribution']['suggested_alignments']} ({self.quality_analysis['alignment_percentages']['suggested']:.1f}%)")
            print(f"   Custom Concepts: {self.quality_analysis['alignment_distribution']['custom_concepts']} ({self.quality_analysis['alignment_percentages']['custom']:.1f}%)")
            print(f"   Overall Alignment Rate: {self.quality_analysis['overall_alignment_rate']:.1f}%")
            
            print("\n[REPORT] Domain Alignment Quality:")
            for domain, quality in sorted(self.quality_analysis["domain_alignment_quality"].items(),
                                         key=lambda x: x[1], reverse=True):
                print(f"   {domain.title()}: {quality:.3f}")
            
            print("\n[REPORT] Terminology Unification:")
            print(f"   Unified Terms: {self.quality_analysis['terminology_unification']['unified_terms']}")
            print(f"   Avg Concepts/Term: {self.quality_analysis['terminology_unification']['avg_concepts_per_term']:.1f}")
            
            if recommendations:
                print("\n[REPORT] Recommendations:")
                for rec in recommendations[:3]:
                    print(f"   [{rec['priority'].upper()}] {rec['message']}")
            
            # Save outputs
            self.save_outputs(recommendations, export_data)
            
            elapsed_time = time.time() - start_time
            print(f"\n[SUCCESS] R3 completed successfully in {elapsed_time:.1f} seconds!")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error in R3: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    aligner = ReferenceAligner()
    return aligner.run()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)