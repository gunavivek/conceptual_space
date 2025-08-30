#!/usr/bin/env python3
"""
R2: Concept Validator
Part of R-Pipeline (Resource & Reasoning Pipeline)
Validates extracted concepts against BIZBOK resources to assess
concept quality, coverage, and alignment with established business knowledge
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import time

class ConceptValidator:
    """Main class for concept validation against BIZBOK resources"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.parent
        self.output_dir = self.script_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.bizbok_concepts = {}
        self.pipeline_concepts = {}
        self.validation_results = {}
        self.coverage_analysis = {}
        self.gap_analysis = {}
    
    def load_bizbok_resources(self):
        """Load BIZBOK concepts from R1 output"""
        concepts_path = self.output_dir / "R1_CONCEPTS.json"
        
        if not concepts_path.exists():
            raise FileNotFoundError(f"R1 concepts not found: {concepts_path}")
        
        print(f"[DATA] Loading BIZBOK resources from R1...")
        with open(concepts_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.bizbok_concepts = data["concepts"]
        
        print(f"   [OK] Loaded {len(self.bizbok_concepts)} BIZBOK concepts")
        return self.bizbok_concepts
    
    def load_pipeline_concepts(self):
        """Load concepts from A-pipeline for validation"""
        print("\n[DATA] Loading pipeline concepts for validation...")
        
        # Try multiple potential sources
        potential_paths = [
            self.script_dir.parent / "A_Concept_pipeline/outputs/A2.5_expanded_concepts.json",
            self.script_dir.parent / "A_Concept_pipeline/outputs/A2.4_core_concepts.json",
            self.script_dir / "../outputs/A2.5_expanded_concepts.json"
        ]
        
        for path in potential_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle different formats
                    if "expanded_concepts" in data:
                        self.pipeline_concepts = data["expanded_concepts"]
                    elif "core_concepts" in data:
                        self.pipeline_concepts = {c["concept_id"]: c for c in data["core_concepts"]}
                    else:
                        self.pipeline_concepts = data
                    
                    print(f"   [OK] Loaded {len(self.pipeline_concepts)} pipeline concepts from {path.name}")
                    return self.pipeline_concepts
        
        # NEW: Option to validate all BIZBOK concepts (self-validation mode)
        print("   [INFO] No A-pipeline concepts found")
        print("   [INFO] Running in SELF-VALIDATION mode - validating all 97 BIZBOK concepts")
        
        # Convert BIZBOK concepts to pipeline format for validation
        self.pipeline_concepts = {}
        for concept_id, concept in self.bizbok_concepts.items():
            self.pipeline_concepts[concept_id] = {
                "concept_id": concept_id,
                "theme_name": concept["name"],
                "primary_keywords": list(concept["keywords"]),
                "domain": concept["domain"],
                "importance_score": len(concept["related_concepts"]) / 50.0  # Normalize by max relationships
            }
        
        print(f"   [OK] Loaded all {len(self.pipeline_concepts)} BIZBOK concepts for self-validation")
        return self.pipeline_concepts
    
    def create_mock_pipeline_concepts(self):
        """Create mock pipeline concepts for testing"""
        return {
            "mock_1": {
                "concept_id": "mock_1",
                "theme_name": "Revenue Management",
                "primary_keywords": ["revenue", "income", "earnings", "sales"],
                "domain": "finance",
                "importance_score": 0.85
            },
            "mock_2": {
                "concept_id": "mock_2",
                "theme_name": "Supply Chain Operations",
                "primary_keywords": ["supply", "chain", "logistics", "inventory"],
                "domain": "operations",
                "importance_score": 0.75
            },
            "mock_3": {
                "concept_id": "mock_3",
                "theme_name": "Cash Flow Analysis",
                "primary_keywords": ["cash", "flow", "liquidity", "working", "capital"],
                "domain": "finance",
                "importance_score": 0.80
            }
        }
    
    def calculate_concept_similarity(self, pipeline_concept, bizbok_concept):
        """Calculate semantic similarity between pipeline and BIZBOK concepts"""
        
        # Extract keywords from both concepts
        pipeline_keywords = set()
        if "original_concept" in pipeline_concept:
            pipeline_keywords.update(pipeline_concept["original_concept"].get("primary_keywords", []))
            pipeline_keywords.update(pipeline_concept.get("all_expanded_terms", []))
        else:
            pipeline_keywords.update(pipeline_concept.get("primary_keywords", []))
            pipeline_keywords.update(pipeline_concept.get("keywords", []))
        
        # Normalize to lowercase
        pipeline_keywords = {k.lower() for k in pipeline_keywords if k}
        
        bizbok_keywords = set(bizbok_concept.get("keywords", []))
        bizbok_keywords = {k.lower() for k in bizbok_keywords if k}
        
        # Calculate Jaccard similarity
        if pipeline_keywords and bizbok_keywords:
            intersection = len(pipeline_keywords & bizbok_keywords)
            union = len(pipeline_keywords | bizbok_keywords)
            jaccard_similarity = intersection / union if union > 0 else 0.0
        else:
            jaccard_similarity = 0.0
        
        # Domain alignment bonus
        pipeline_domain = pipeline_concept.get("domain", "general").lower()
        bizbok_domain = bizbok_concept.get("domain", "general").lower()
        domain_bonus = 0.2 if pipeline_domain == bizbok_domain else 0.0
        
        # Name similarity bonus
        pipeline_name = pipeline_concept.get("theme_name", "").lower()
        bizbok_name = bizbok_concept.get("name", "").lower()
        name_similarity = 0.0
        
        if pipeline_name and bizbok_name:
            pipeline_name_words = set(pipeline_name.split())
            bizbok_name_words = set(bizbok_name.split())
            if pipeline_name_words & bizbok_name_words:
                name_similarity = 0.15
        
        # Calculate overall similarity
        overall_similarity = min(1.0, jaccard_similarity + domain_bonus + name_similarity)
        
        return {
            "jaccard_similarity": jaccard_similarity,
            "domain_alignment": pipeline_domain == bizbok_domain,
            "domain_bonus": domain_bonus,
            "name_similarity": name_similarity,
            "overall_similarity": overall_similarity,
            "common_keywords": list(pipeline_keywords & bizbok_keywords)[:10],
            "pipeline_unique": list(pipeline_keywords - bizbok_keywords)[:10],
            "bizbok_unique": list(bizbok_keywords - pipeline_keywords)[:10]
        }
    
    def validate_concepts(self):
        """Validate pipeline concepts against BIZBOK resources"""
        print("\n[VALIDATE] Validating pipeline concepts...")
        
        for pipeline_id, pipeline_concept in self.pipeline_concepts.items():
            concept_validation = {
                "pipeline_concept_id": pipeline_id,
                "pipeline_concept": pipeline_concept,
                "bizbok_matches": [],
                "best_match": None,
                "validation_score": 0.0,
                "coverage_quality": "unknown"
            }
            
            # Compare with all BIZBOK concepts
            for bizbok_id, bizbok_concept in self.bizbok_concepts.items():
                similarity = self.calculate_concept_similarity(pipeline_concept, bizbok_concept)
                
                if similarity["overall_similarity"] > 0.1:  # Minimum threshold
                    match_data = {
                        "bizbok_id": bizbok_id,
                        "bizbok_concept": {
                            "name": bizbok_concept["name"],
                            "domain": bizbok_concept["domain"]
                        },
                        "similarity_analysis": similarity
                    }
                    concept_validation["bizbok_matches"].append(match_data)
            
            # Sort matches by similarity
            concept_validation["bizbok_matches"].sort(
                key=lambda x: x["similarity_analysis"]["overall_similarity"],
                reverse=True
            )
            
            # Set best match and validation score
            if concept_validation["bizbok_matches"]:
                concept_validation["best_match"] = concept_validation["bizbok_matches"][0]
                concept_validation["validation_score"] = concept_validation["best_match"]["similarity_analysis"]["overall_similarity"]
                
                # Determine coverage quality
                score = concept_validation["validation_score"]
                if score >= 0.7:
                    concept_validation["coverage_quality"] = "excellent"
                elif score >= 0.5:
                    concept_validation["coverage_quality"] = "good"
                elif score >= 0.3:
                    concept_validation["coverage_quality"] = "fair"
                else:
                    concept_validation["coverage_quality"] = "poor"
            else:
                concept_validation["coverage_quality"] = "no_match"
            
            self.validation_results[pipeline_id] = concept_validation
        
        print(f"   [OK] Validated {len(self.validation_results)} concepts")
    
    def analyze_coverage(self):
        """Analyze overall validation coverage and quality"""
        print("\n[ANALYSIS] Analyzing coverage...")
        
        total_concepts = len(self.validation_results)
        coverage_counts = Counter()
        validation_scores = []
        domain_performance = defaultdict(list)
        
        for concept_id, validation in self.validation_results.items():
            coverage = validation["coverage_quality"]
            score = validation["validation_score"]
            
            coverage_counts[coverage] += 1
            validation_scores.append(score)
            
            # Track domain performance
            if "original_concept" in validation["pipeline_concept"]:
                domain = validation["pipeline_concept"]["original_concept"].get("domain", "general")
            else:
                domain = validation["pipeline_concept"].get("domain", "general")
            
            domain_performance[domain].append(score)
        
        # Calculate domain averages
        domain_averages = {}
        for domain, scores in domain_performance.items():
            domain_averages[domain] = np.mean(scores) if scores else 0.0
        
        # Overall statistics
        avg_validation_score = np.mean(validation_scores) if validation_scores else 0.0
        
        self.coverage_analysis = {
            "total_concepts_validated": total_concepts,
            "coverage_distribution": dict(coverage_counts),
            "coverage_percentages": {k: (v/total_concepts)*100 for k, v in coverage_counts.items()},
            "average_validation_score": avg_validation_score,
            "domain_performance": domain_averages,
            "score_distribution": {
                "high_quality": len([s for s in validation_scores if s >= 0.7]),
                "medium_quality": len([s for s in validation_scores if 0.4 <= s < 0.7]),
                "low_quality": len([s for s in validation_scores if s < 0.4])
            }
        }
        
        print(f"   [OK] Average validation score: {avg_validation_score:.3f}")
    
    def identify_gaps(self):
        """Identify coverage gaps in pipeline concepts"""
        print("\n[ANALYZE] Identifying coverage gaps...")
        
        # Track which BIZBOK concepts are covered
        covered_bizbok = set()
        coverage_quality = {}
        
        for validation in self.validation_results.values():
            if validation["best_match"]:
                bizbok_id = validation["best_match"]["bizbok_id"]
                score = validation["validation_score"]
                
                covered_bizbok.add(bizbok_id)
                if bizbok_id not in coverage_quality or coverage_quality[bizbok_id] < score:
                    coverage_quality[bizbok_id] = score
        
        # Identify uncovered BIZBOK concepts
        all_bizbok = set(self.bizbok_concepts.keys())
        uncovered_bizbok = all_bizbok - covered_bizbok
        
        # Analyze uncovered by domain
        gap_analysis = defaultdict(list)
        for bizbok_id in uncovered_bizbok:
            bizbok_concept = self.bizbok_concepts[bizbok_id]
            domain = bizbok_concept["domain"]
            
            gap_analysis[domain].append({
                "bizbok_id": bizbok_id,
                "concept_name": bizbok_concept["name"],
                "keywords": bizbok_concept["keywords"][:5]  # Top 5 keywords
            })
        
        # Limit gaps per domain for readability
        for domain in gap_analysis:
            gap_analysis[domain] = gap_analysis[domain][:10]  # Top 10 per domain
        
        self.gap_analysis = {
            "total_bizbok_concepts": len(all_bizbok),
            "covered_concepts": len(covered_bizbok),
            "uncovered_concepts": len(uncovered_bizbok),
            "coverage_ratio": len(covered_bizbok) / len(all_bizbok) if all_bizbok else 0,
            "gaps_by_domain": dict(gap_analysis),
            "critical_gaps": []  # Will be populated based on importance
        }
        
        # Identify critical gaps (high-connectivity concepts)
        for bizbok_id in uncovered_bizbok:
            bizbok_concept = self.bizbok_concepts[bizbok_id]
            if len(bizbok_concept.get("related_concepts", [])) > 3:
                self.gap_analysis["critical_gaps"].append({
                    "bizbok_id": bizbok_id,
                    "name": bizbok_concept["name"],
                    "domain": bizbok_concept["domain"],
                    "connections": len(bizbok_concept["related_concepts"])
                })
        
        # Sort critical gaps by connection count
        self.gap_analysis["critical_gaps"].sort(key=lambda x: x["connections"], reverse=True)
        self.gap_analysis["critical_gaps"] = self.gap_analysis["critical_gaps"][:10]
        
        print(f"   [OK] Coverage ratio: {self.gap_analysis['coverage_ratio']:.1%}")
        print(f"   [OK] Identified {len(uncovered_bizbok)} uncovered concepts")
    
    def generate_recommendations(self):
        """Generate recommendations for improving concept validation"""
        recommendations = []
        
        # Overall quality recommendations
        avg_score = self.coverage_analysis["average_validation_score"]
        if avg_score < 0.5:
            recommendations.append({
                "type": "quality",
                "priority": "high",
                "message": f"Low overall validation quality ({avg_score:.2f}) - review concept extraction strategies"
            })
        
        # Coverage recommendations
        coverage_ratio = self.gap_analysis["coverage_ratio"]
        if coverage_ratio < 0.7:
            recommendations.append({
                "type": "coverage",
                "priority": "high",
                "message": f"Low BIZBOK coverage ({coverage_ratio:.1%}) - expand concept identification scope"
            })
        
        # Domain-specific recommendations
        for domain, avg_score in self.coverage_analysis["domain_performance"].items():
            if avg_score < 0.4:
                recommendations.append({
                    "type": "domain",
                    "priority": "medium",
                    "message": f"Poor {domain} domain performance ({avg_score:.2f}) - strengthen domain-specific extraction"
                })
        
        # Critical gaps recommendations
        if len(self.gap_analysis["critical_gaps"]) > 5:
            recommendations.append({
                "type": "gaps",
                "priority": "medium",
                "message": f"{len(self.gap_analysis['critical_gaps'])} highly-connected concepts missing - prioritize for extraction"
            })
        
        # Quality distribution recommendations
        score_dist = self.coverage_analysis["score_distribution"]
        if score_dist["low_quality"] > score_dist["high_quality"]:
            recommendations.append({
                "type": "distribution",
                "priority": "medium",
                "message": "More low-quality than high-quality matches - improve concept extraction precision"
            })
        
        return recommendations
    
    def save_outputs(self, recommendations):
        """Save validation results and analysis"""
        print("\n[SAVE] Saving validation outputs...")
        
        # Main validation report
        report_data = {
            "metadata": {
                "validation_timestamp": datetime.now().isoformat(),
                "total_concepts_validated": len(self.validation_results),
                "validation_method": "BIZBOK_resource_comparison",
                "version": "2.0"
            },
            "validation_results": self.validation_results,
            "coverage_analysis": self.coverage_analysis,
            "gap_analysis": self.gap_analysis,
            "recommendations": recommendations
        }
        
        output_path = self.output_dir / "R2_validation_report.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {output_path.name}")
        
        # Summary text file
        summary_path = self.output_dir / "R2_validation_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("CONCEPT VALIDATION SUMMARY\n")
            f.write("="*50 + "\n")
            f.write("R-Pipeline: Resource & Reasoning Pipeline\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total Concepts Validated: {self.coverage_analysis['total_concepts_validated']}\n")
            f.write(f"Average Validation Score: {self.coverage_analysis['average_validation_score']:.3f}\n")
            f.write(f"BIZBOK Coverage: {self.gap_analysis['coverage_ratio']:.1%}\n\n")
            
            f.write("Coverage Distribution:\n")
            for coverage, percentage in self.coverage_analysis["coverage_percentages"].items():
                f.write(f"  {coverage.replace('_', ' ').title()}: {percentage:.1f}%\n")
            
            f.write("\nDomain Performance:\n")
            for domain, score in self.coverage_analysis["domain_performance"].items():
                f.write(f"  {domain.title()}: {score:.3f}\n")
            
            f.write("\nRecommendations:\n")
            for rec in recommendations:
                f.write(f"  [{rec['priority'].upper()}] {rec['message']}\n")
            
            if self.gap_analysis["critical_gaps"]:
                f.write("\nCritical Missing Concepts:\n")
                for gap in self.gap_analysis["critical_gaps"][:5]:
                    f.write(f"  - {gap['name']} ({gap['domain']}) - {gap['connections']} connections\n")
        
        print(f"   [OK] Saved {summary_path.name}")
    
    def run(self):
        """Main execution method"""
        print("="*60)
        print("R2: Concept Validator")
        print("R-Pipeline: Resource & Reasoning Pipeline")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Load resources
            self.load_bizbok_resources()
            self.load_pipeline_concepts()
            
            # Perform validation
            self.validate_concepts()
            
            # Analyze results
            self.analyze_coverage()
            self.identify_gaps()
            
            # Generate recommendations
            recommendations = self.generate_recommendations()
            
            # Display results
            print("\n[ANALYSIS] Validation Results:")
            print(f"   Total Concepts: {self.coverage_analysis['total_concepts_validated']}")
            print(f"   Average Score: {self.coverage_analysis['average_validation_score']:.3f}")
            print(f"   BIZBOK Coverage: {self.gap_analysis['coverage_ratio']:.1%}")
            
            print("\n[REPORT] Coverage Quality:")
            for coverage, percentage in self.coverage_analysis["coverage_percentages"].items():
                print(f"   {coverage.replace('_', ' ').title()}: {percentage:.1f}%")
            
            print("\n[REPORT] Domain Performance:")
            for domain, score in sorted(self.coverage_analysis["domain_performance"].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"   {domain.title()}: {score:.3f}")
            
            if recommendations:
                print("\n[REPORT] Top Recommendations:")
                for rec in recommendations[:3]:
                    print(f"   [{rec['priority'].upper()}] {rec['message']}")
            
            # Save outputs
            self.save_outputs(recommendations)
            
            elapsed_time = time.time() - start_time
            print(f"\n[SUCCESS] R2 completed successfully in {elapsed_time:.1f} seconds!")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error in R2: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    validator = ConceptValidator()
    return validator.run()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)