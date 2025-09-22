"""
Knowledge Graph Processor: Semantic relationship-based expansion
Implements is-a, part-of, causes, requires relationships for concept expansion
"""

class KnowledgeGraphProcessor:
    """Knowledge graph-based concept expansion using semantic relationships"""

    def __init__(self, learning_enabled=True):
        self.relationships = {
            "is_a": {},      # Hierarchical relationships
            "part_of": {},   # Compositional relationships
            "causes": {},    # Causal relationships
            "requires": {}   # Dependency relationships
        }
        self.learning_enabled = learning_enabled
        self.learned_relationships = {"is_a": {}, "part_of": {}, "causes": {}, "requires": {}}

    def add_relationship(self, subject, predicate, object_term):
        """Add a relationship to the knowledge graph"""
        if predicate in self.relationships:
            if subject not in self.relationships[predicate]:
                self.relationships[predicate][subject] = []
            self.relationships[predicate][subject].append(object_term)

    def populate_common_relationships(self):
        """Populate with common domain relationships"""
        # Technical relationships
        self.add_relationship("algorithm", "is_a", "procedure")
        self.add_relationship("database", "is_a", "system")
        self.add_relationship("function", "part_of", "program")
        self.add_relationship("bug", "causes", "error")
        self.add_relationship("testing", "requires", "specification")

        # General relationships
        self.add_relationship("car", "is_a", "vehicle")
        self.add_relationship("wheel", "part_of", "car")
        self.add_relationship("rain", "causes", "flooding")
        self.add_relationship("cooking", "requires", "ingredients")

    def learn_relationships_from_concepts(self, concepts):
        """Learn relationships from concept collection"""
        learned_stats = {"is_a": 0, "part_of": 0, "causes": 0, "requires": 0}

        if not self.learning_enabled:
            return learned_stats

        for concept in concepts:
            keywords = concept.get("keywords", [])
            canonical_name = concept.get("canonical_name", "")

            # Learn hierarchical relationships (is_a)
            for keyword in keywords:
                if any(indicator in keyword.lower() for indicator in ["type", "kind", "category"]):
                    if canonical_name and keyword != canonical_name:
                        self.learned_relationships["is_a"][canonical_name.lower()] = [keyword]
                        learned_stats["is_a"] += 1

            # Learn compositional relationships (part_of)
            for keyword in keywords:
                if any(indicator in keyword.lower() for indicator in ["component", "element", "part"]):
                    if canonical_name and keyword != canonical_name:
                        self.learned_relationships["part_of"][keyword.lower()] = [canonical_name]
                        learned_stats["part_of"] += 1

        return learned_stats

    def discover_implicit_relationships(self, concept, all_concepts, max_discoveries=5):
        """Discover implicit relationships based on keyword patterns"""
        implicit_relationships = []
        keywords = concept.get("keywords", [])

        for other_concept in all_concepts:
            if other_concept.get("concept_id") == concept.get("concept_id"):
                continue

            other_keywords = other_concept.get("keywords", [])

            # Find potential relationships based on keyword patterns
            for keyword in keywords:
                for other_keyword in other_keywords:
                    # Hierarchical pattern detection
                    if keyword.lower().endswith("tion") and other_keyword.lower().endswith("er"):
                        implicit_relationships.append({
                            "term": other_keyword,
                            "relation_type": "implicit_agent",
                            "source_keyword": keyword,
                            "confidence": 0.6,
                            "pattern": "action_agent"
                        })

                    # Causal pattern detection
                    if "problem" in keyword.lower() and "solution" in other_keyword.lower():
                        implicit_relationships.append({
                            "term": other_keyword,
                            "relation_type": "implicit_solution",
                            "source_keyword": keyword,
                            "confidence": 0.7,
                            "pattern": "problem_solution"
                        })

        return implicit_relationships[:max_discoveries]

    def expand_with_relationships(self, concept, max_expansions=5):
        """Expand concept using knowledge graph relationships"""
        keywords = concept.get("keywords", [])
        expansions = []

        # Check predefined relationships
        for keyword in keywords:
            for relation_type, relations in self.relationships.items():
                if keyword.lower() in relations:
                    for related_term in relations[keyword.lower()]:
                        expansions.append({
                            "term": related_term,
                            "relation_type": relation_type,
                            "source_keyword": keyword,
                            "confidence": 0.8
                        })

        # Check learned relationships
        for keyword in keywords:
            for relation_type, relations in self.learned_relationships.items():
                if keyword.lower() in relations:
                    for related_term in relations[keyword.lower()]:
                        expansions.append({
                            "term": related_term,
                            "relation_type": f"learned_{relation_type}",
                            "source_keyword": keyword,
                            "confidence": 0.7
                        })

        return expansions[:max_expansions]

    def get_processor_info(self):
        """Get information about the knowledge graph processor"""
        total_predefined = sum(len(relations) for relations in self.relationships.values())
        total_learned = sum(len(relations) for relations in self.learned_relationships.values())

        return {
            "total_relationships": total_predefined + total_learned,
            "predefined_relationships": total_predefined,
            "learned_relationships": total_learned,
            "relationship_types": list(self.relationships.keys()),
            "learning_enabled": self.learning_enabled
        }