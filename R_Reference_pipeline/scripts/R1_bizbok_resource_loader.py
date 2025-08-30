#!/usr/bin/env python3
"""
R1: BIZBOK Resource Loader
Part of R-Pipeline (Resource & Reasoning Pipeline)
Loads and processes Business Body of Knowledge (BIZBOK) resources from Excel
Generates CONCEPTS, DOMAINS, and KEYWORDS outputs for ontological reasoning
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import time

# Stop words for keyword extraction
STOP_WORDS = {
    "the", "a", "an", "is", "are", "of", "for", "in", "on", "to", "and", "or", 
    "with", "by", "from", "as", "at", "that", "which", "this", "these", "those", 
    "has", "have", "had", "will", "would", "could", "should", "may", "might", 
    "must", "can", "been", "being", "be", "was", "were", "am", "into", "through",
    "during", "before", "after", "above", "below", "between", "under", "over",
    "all", "any", "some", "few", "more", "most", "other", "such", "no", "not"
}

class BIZBOKResourceLoader:
    """Main class for loading and processing BIZBOK resources"""
    
    def __init__(self, add_bidirectional=True):
        self.script_dir = Path(__file__).parent.parent
        self.excel_path = self.script_dir / "data/Information Map for all industry and common.xlsx"
        self.output_dir = self.script_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.concepts = {}
        self.domains = {}
        self.keyword_index = defaultdict(list)
        self.processing_stats = {}
        self.add_bidirectional = add_bidirectional  # Toggle for bidirectional relationships
    
    def load_excel_data(self):
        """Load data from Excel file"""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        
        print(f"[DATA] Loading BIZBOK resources from: {self.excel_path.name}")
        df = pd.read_excel(self.excel_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        print(f"   Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        return df
    
    def extract_keywords(self, concept_name, definition):
        """Extract keywords from concept name and definition"""
        keywords = set()
        
        # 1. Extract from concept name
        name_clean = concept_name.lower().replace('-', ' ').replace('_', ' ')
        name_parts = [w for w in name_clean.split() if w not in STOP_WORDS and len(w) > 2]
        keywords.update(name_parts)
        
        # Add full name if multi-word
        if len(name_parts) > 1:
            keywords.add(name_clean)
        
        # 2. Extract from definition
        # Clean definition
        definition_clean = re.sub(r'[^\w\s]', ' ', definition.lower())
        definition_words = definition_clean.split()
        
        # Add meaningful words
        meaningful_words = [w for w in definition_words 
                          if w not in STOP_WORDS and len(w) > 2 and not w.isdigit()]
        keywords.update(meaningful_words[:20])  # Limit to top 20 words from definition
        
        # 3. Extract compound terms (2-grams)
        for i in range(len(definition_words) - 1):
            if (definition_words[i] not in STOP_WORDS and 
                definition_words[i+1] not in STOP_WORDS and
                len(definition_words[i]) > 2 and 
                len(definition_words[i+1]) > 2):
                compound = f"{definition_words[i]} {definition_words[i+1]}"
                # Add if compound appears in original definition
                if compound in definition.lower():
                    keywords.add(compound)
        
        # 4. Extract acronyms (uppercase words)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', definition)
        keywords.update([a.lower() for a in acronyms])
        
        return sorted(list(keywords))
    
    def parse_related_concepts(self, related_str):
        """Parse related concepts from Excel string"""
        if pd.isna(related_str) or str(related_str).strip() == '' or str(related_str) == 'nan':
            return []
        
        related_concepts = []
        # Split by common delimiters
        delimiters = [',', ';', '|', '\n', '/', '&']
        temp_concepts = [str(related_str)]
        
        for delimiter in delimiters:
            new_concepts = []
            for concept in temp_concepts:
                new_concepts.extend(concept.split(delimiter))
            temp_concepts = new_concepts
        
        # Clean and convert to concept IDs
        for concept in temp_concepts:
            concept_clean = concept.strip()
            if concept_clean and len(concept_clean) > 1:
                # Convert to concept ID format
                concept_id = concept_clean.lower().replace(' ', '_').replace('-', '_')
                concept_id = re.sub(r'[^\w_]', '', concept_id)  # Remove special chars
                if concept_id:
                    related_concepts.append(concept_id)
        
        return list(set(related_concepts))  # Remove duplicates
    
    def process_concepts(self, df):
        """Process concepts from DataFrame"""
        print("\n[PROCESS] Processing concepts...")
        
        # Map columns (case-insensitive)
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'domain' in col_lower:
                column_mapping['domain'] = col
            elif 'information concept' == col_lower:
                column_mapping['concept_name'] = col
            elif 'definition' in col_lower:
                column_mapping['definition'] = col
            elif 'related' in col_lower:
                column_mapping['related'] = col
        
        print(f"   Column mapping: {column_mapping}")
        
        valid_concepts = 0
        skipped_rows = 0
        
        for idx, row in df.iterrows():
            # Extract fields (with proper trimming of whitespace)
            domain = str(row.get(column_mapping.get('domain', 'Domain'), 'general')).strip().lower()
            concept_name_raw = row.get(column_mapping.get('concept_name', 'Concept Name'), '')
            # Strip whitespace from concept name to handle Excel inconsistencies
            concept_name = str(concept_name_raw).strip() if pd.notna(concept_name_raw) else ''
            definition = str(row.get(column_mapping.get('definition', 'Concept Definition'), '')).strip()
            related_str = row.get(column_mapping.get('related', 'Related Concepts'), '')
            
            # Skip invalid rows
            if not concept_name or pd.isna(row.get(column_mapping.get('concept_name', 'Concept Name'))) or concept_name == 'nan':
                skipped_rows += 1
                continue
            
            # Create concept ID
            concept_id = concept_name.lower().replace(' ', '_').replace('-', '_')
            concept_id = re.sub(r'[^\w_]', '', concept_id)
            
            # Extract keywords
            keywords = self.extract_keywords(concept_name, definition)
            
            # Parse related concepts
            related_concepts = self.parse_related_concepts(related_str)
            
            # Store concept (preserve first occurrence of duplicates)
            if concept_id not in self.concepts:
                self.concepts[concept_id] = {
                    "concept_id": concept_id,
                    "name": concept_name,
                    "definition": definition if definition != 'nan' else f"Concept: {concept_name}",
                    "domain": domain,
                    "related_concepts": related_concepts.copy(),  # Original relationships from Excel
                    "original_relationships": related_concepts.copy(),  # Preserve original for reference
                    "keywords": keywords,
                    "source_row": idx + 2  # Excel row number (1-indexed + header)
                }
            else:
                # Track duplicate for reporting
                skipped_rows += 1
                print(f"   [INFO] Skipped duplicate concept '{concept_name}' at row {idx + 2} (keeping first occurrence at row {self.concepts[concept_id]['source_row']})")
            
            # Update domain
            if domain not in self.domains:
                self.domains[domain] = {
                    "domain_id": domain,
                    "name": domain.replace('_', ' ').title(),
                    "description": f"Business concepts related to {domain}",
                    "concepts": [],
                    "concept_count": 0
                }
            self.domains[domain]["concepts"].append(concept_id)
            
            # Update keyword index
            for keyword in keywords:
                self.keyword_index[keyword].append(concept_id)
            
            valid_concepts += 1
        
        # Update domain counts
        for domain in self.domains:
            self.domains[domain]["concept_count"] = len(self.domains[domain]["concepts"])
        
        self.processing_stats = {
            "total_rows": len(df),
            "valid_concepts": valid_concepts,
            "skipped_rows": skipped_rows,
            "domains": len(self.domains),
            "unique_keywords": len(self.keyword_index)
        }
        
        print(f"   [OK] Processed {valid_concepts} valid concepts")
        print(f"   [OK] Created {len(self.domains)} domains")
        print(f"   [OK] Extracted {len(self.keyword_index)} unique keywords")
    
    def validate_relationships(self):
        """Validate and fix relationship references"""
        print("\n[VALIDATE] Validating relationships...")
        
        valid_relationships = 0
        invalid_relationships = []
        bidirectional_added = 0
        
        for concept_id, concept_data in self.concepts.items():
            valid_related = []
            
            for related_id in concept_data["related_concepts"]:
                if related_id in self.concepts:
                    valid_related.append(related_id)
                    valid_relationships += 1
                    
                    # Add bidirectional relationship (optional - can be disabled)
                    # Note: This can create many relationships for highly-connected concepts
                    if self.add_bidirectional and concept_id not in self.concepts[related_id]["related_concepts"]:
                        self.concepts[related_id]["related_concepts"].append(concept_id)
                        bidirectional_added += 1
                else:
                    invalid_relationships.append({
                        "from": concept_id,
                        "to": related_id,
                        "reason": "Target concept not found"
                    })
            
            # Update with valid relationships only
            concept_data["related_concepts"] = valid_related
        
        print(f"   [OK] Validated {valid_relationships} relationships")
        print(f"   [OK] Added {bidirectional_added} bidirectional relationships")
        if invalid_relationships:
            print(f"   [WARNING] Found {len(invalid_relationships)} invalid references")
        
        return {
            "valid_relationships": valid_relationships,
            "invalid_relationships": invalid_relationships[:10],  # Limit to first 10
            "bidirectional_added": bidirectional_added
        }
    
    def generate_statistics(self):
        """Generate comprehensive statistics"""
        stats = {
            "processing": self.processing_stats,
            "concepts": {
                "total": len(self.concepts),
                "avg_keywords_per_concept": 0,
                "avg_relationships_per_concept": 0,
                "max_keywords": 0,
                "max_relationships": 0
            },
            "domains": {
                "total": len(self.domains),
                "distribution": {}
            },
            "keywords": {
                "total_unique": len(self.keyword_index),
                "avg_concepts_per_keyword": 0,
                "top_keywords": []
            }
        }
        
        # Concept statistics
        if self.concepts:
            keyword_counts = [len(c["keywords"]) for c in self.concepts.values()]
            relationship_counts = [len(c["related_concepts"]) for c in self.concepts.values()]
            
            stats["concepts"]["avg_keywords_per_concept"] = np.mean(keyword_counts)
            stats["concepts"]["avg_relationships_per_concept"] = np.mean(relationship_counts)
            stats["concepts"]["max_keywords"] = max(keyword_counts)
            stats["concepts"]["max_relationships"] = max(relationship_counts)
        
        # Domain distribution
        for domain_id, domain_data in self.domains.items():
            stats["domains"]["distribution"][domain_id] = domain_data["concept_count"]
        
        # Keyword statistics
        if self.keyword_index:
            concepts_per_keyword = [len(concepts) for concepts in self.keyword_index.values()]
            stats["keywords"]["avg_concepts_per_keyword"] = np.mean(concepts_per_keyword)
            
            # Top keywords
            keyword_counts = Counter()
            for keyword, concepts in self.keyword_index.items():
                keyword_counts[keyword] = len(concepts)
            
            stats["keywords"]["top_keywords"] = [
                {"keyword": k, "concept_count": v} 
                for k, v in keyword_counts.most_common(20)
            ]
        
        return stats
    
    def save_outputs(self, validation_results, statistics):
        """Save all outputs in specified format"""
        print("\n[SAVE] Saving outputs...")
        
        # 1. Save R1_CONCEPTS.json
        concepts_output = {
            "metadata": {
                "source": "BIZBOK Excel Resource",
                "processing_timestamp": datetime.now().isoformat(),
                "total_concepts": len(self.concepts),
                "version": "2.0"
            },
            "concepts": self.concepts
        }
        
        concepts_path = self.output_dir / "R1_CONCEPTS.json"
        with open(concepts_path, 'w', encoding='utf-8') as f:
            json.dump(concepts_output, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {concepts_path.name}")
        
        # 2. Save R1_DOMAINS.json
        domains_output = {
            "metadata": {
                "source": "BIZBOK Excel Resource",
                "processing_timestamp": datetime.now().isoformat(),
                "total_domains": len(self.domains),
                "version": "2.0"
            },
            "domains": self.domains
        }
        
        domains_path = self.output_dir / "R1_DOMAINS.json"
        with open(domains_path, 'w', encoding='utf-8') as f:
            json.dump(domains_output, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {domains_path.name}")
        
        # 3. Save R1_KEYWORDS.json
        keywords_output = {
            "metadata": {
                "source": "BIZBOK Excel Resource",
                "processing_timestamp": datetime.now().isoformat(),
                "total_keywords": len(self.keyword_index),
                "version": "2.0"
            },
            "keyword_index": dict(self.keyword_index)
        }
        
        keywords_path = self.output_dir / "R1_KEYWORDS.json"
        with open(keywords_path, 'w', encoding='utf-8') as f:
            json.dump(keywords_output, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {keywords_path.name}")
        
        # 4. Save processing report
        report_output = {
            "validation": validation_results,
            "statistics": statistics,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        report_path = self.output_dir / "R1_processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_output, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved {report_path.name}")
    
    def run(self):
        """Main execution method"""
        print("="*60)
        print("R1: BIZBOK Resource Loader")
        print("R-Pipeline: Resource & Reasoning Pipeline")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Load Excel data
            df = self.load_excel_data()
            
            # Process concepts
            self.process_concepts(df)
            
            # Validate relationships
            validation_results = self.validate_relationships()
            
            # Generate statistics
            statistics = self.generate_statistics()
            
            # Display summary
            print("\n[ANALYSIS] Processing Summary:")
            print(f"   Total Concepts: {statistics['concepts']['total']}")
            print(f"   Total Domains: {statistics['domains']['total']}")
            print(f"   Unique Keywords: {statistics['keywords']['total_unique']}")
            print(f"   Avg Keywords/Concept: {statistics['concepts']['avg_keywords_per_concept']:.1f}")
            print(f"   Avg Relationships/Concept: {statistics['concepts']['avg_relationships_per_concept']:.1f}")
            
            print("\n[REPORT] Domain Distribution:")
            for domain, count in sorted(statistics['domains']['distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {domain}: {count} concepts")
            
            print("\n[REPORT] Top Keywords:")
            for item in statistics['keywords']['top_keywords'][:5]:
                print(f"   '{item['keyword']}': {item['concept_count']} concepts")
            
            # Save outputs
            self.save_outputs(validation_results, statistics)
            
            elapsed_time = time.time() - start_time
            print(f"\n[SUCCESS] R1 completed successfully in {elapsed_time:.1f} seconds!")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error in R1: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    # Enable bidirectional relationships for richer ontology building
    loader = BIZBOKResourceLoader(add_bidirectional=True)  # Enabled for A/B pipeline integration
    return loader.run()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)