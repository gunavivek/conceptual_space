"""
Embedding Manager: Vector embeddings + cosine similarity for semantic neighbor discovery
Supports Word2Vec, GloVe, and sentence transformers for concept expansion
"""

import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import gensim.downloader as api
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

class EmbeddingManager:
    """
    Advanced embedding manager for semantic neighbor discovery
    """

    def __init__(self, model_type='sentence_transformer', model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding manager

        Args:
            model_type: 'sentence_transformer', 'word2vec', 'glove', or 'tfidf'
            model_name: Specific model to use
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}

        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        if self.model_type == 'sentence_transformer' and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                print(f"  [OK] Loaded SentenceTransformer: {self.model_name}")
            except Exception as e:
                print(f"  [WARNING] Failed to load SentenceTransformer: {e}")
                self._fallback_to_tfidf()

        elif self.model_type == 'word2vec' and GENSIM_AVAILABLE:
            try:
                self.model = api.load("word2vec-google-news-300")
                print(f"  [OK] Loaded Word2Vec model")
            except Exception as e:
                print(f"  [WARNING] Failed to load Word2Vec: {e}")
                self._fallback_to_tfidf()

        else:
            self._fallback_to_tfidf()

    def _fallback_to_tfidf(self):
        """Fallback to TF-IDF if advanced models unavailable"""
        self.model_type = 'tfidf'
        self.model = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        print(f"  [OK] Using TF-IDF fallback")

    def get_concept_embedding(self, concept):
        """
        Get embedding vector for a concept

        Args:
            concept: Concept dictionary with keywords

        Returns:
            numpy.array: Embedding vector
        """
        concept_id = concept.get("concept_id", "")

        # Check cache
        if concept_id in self.embeddings_cache:
            return self.embeddings_cache[concept_id]

        # Create text representation
        keywords = concept.get("keywords", [])
        canonical_name = concept.get("canonical_name", "")
        text = " ".join(keywords + [canonical_name]).strip()

        if not text:
            return np.zeros(300)  # Default dimension

        # Get embedding based on model type
        if self.model_type == 'sentence_transformer':
            embedding = self.model.encode([text])[0]

        elif self.model_type == 'word2vec':
            # Average word vectors
            word_vectors = []
            for word in text.split():
                try:
                    word_vectors.append(self.model[word])
                except KeyError:
                    continue

            if word_vectors:
                embedding = np.mean(word_vectors, axis=0)
            else:
                embedding = np.zeros(300)

        elif self.model_type == 'tfidf':
            # For TF-IDF, we need to fit on all concepts first
            embedding = text  # Return text for batch processing

        else:
            embedding = np.zeros(300)

        # Cache result
        self.embeddings_cache[concept_id] = embedding
        return embedding

    def get_all_embeddings(self, concepts):
        """
        Get embeddings for all concepts efficiently

        Args:
            concepts: List of concept dictionaries

        Returns:
            numpy.array: Matrix of embeddings
        """
        if self.model_type == 'tfidf':
            # Special handling for TF-IDF
            texts = []
            for concept in concepts:
                keywords = concept.get("keywords", [])
                canonical_name = concept.get("canonical_name", "")
                text = " ".join(keywords + [canonical_name]).strip()
                texts.append(text)

            try:
                embeddings = self.model.fit_transform(texts).toarray()
                return embeddings
            except ValueError:
                return np.zeros((len(concepts), 100))

        else:
            # Get individual embeddings
            embeddings = []
            for concept in concepts:
                embedding = self.get_concept_embedding(concept)
                embeddings.append(embedding)

            return np.array(embeddings)

    def find_semantic_neighbors(self, target_concept, all_concepts, similarity_threshold=0.7, max_neighbors=10):
        """
        Find semantic neighbors using cosine similarity

        Args:
            target_concept: Target concept to find neighbors for
            all_concepts: All available concepts
            similarity_threshold: Minimum similarity score
            max_neighbors: Maximum number of neighbors

        Returns:
            list: Semantic neighbors with similarity scores
        """
        # Get all embeddings
        all_embeddings = self.get_all_embeddings(all_concepts)

        if all_embeddings.size == 0:
            return []

        # Find target concept index
        target_id = target_concept.get("concept_id", "")
        target_index = None

        for i, concept in enumerate(all_concepts):
            if concept.get("concept_id") == target_id:
                target_index = i
                break

        if target_index is None:
            return []

        # Calculate similarities
        target_embedding = all_embeddings[target_index].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, all_embeddings)[0]

        # Find neighbors
        neighbors = []
        for i, similarity in enumerate(similarities):
            if i != target_index and similarity >= similarity_threshold:
                neighbors.append({
                    "concept": all_concepts[i],
                    "similarity_score": float(similarity),
                    "concept_index": i
                })

        # Sort by similarity and return top neighbors
        neighbors.sort(key=lambda x: x["similarity_score"], reverse=True)
        return neighbors[:max_neighbors]

    def extract_expansion_terms(self, target_concept, semantic_neighbors, max_terms=5):
        """
        Extract expansion terms from semantic neighbors

        Args:
            target_concept: Target concept
            semantic_neighbors: List of semantic neighbors
            max_terms: Maximum expansion terms

        Returns:
            list: Expansion terms with metadata
        """
        target_keywords = set(kw.lower().strip() for kw in target_concept.get("keywords", []))
        expansion_candidates = []

        for neighbor_data in semantic_neighbors:
            neighbor_concept = neighbor_data["concept"]
            similarity_score = neighbor_data["similarity_score"]
            neighbor_keywords = neighbor_concept.get("keywords", [])

            for keyword in neighbor_keywords:
                if keyword.lower().strip() not in target_keywords:
                    expansion_candidates.append({
                        "term": keyword,
                        "similarity_score": similarity_score,
                        "source_concept_id": neighbor_concept.get("concept_id"),
                        "source_concept_name": neighbor_concept.get("canonical_name", ""),
                        "expansion_method": "semantic_neighbor"
                    })

        # Remove duplicates and sort by similarity
        seen_terms = set()
        unique_candidates = []
        for candidate in sorted(expansion_candidates, key=lambda x: x["similarity_score"], reverse=True):
            if candidate["term"] not in seen_terms:
                seen_terms.add(candidate["term"])
                unique_candidates.append(candidate)

        return unique_candidates[:max_terms]

    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "cache_size": len(self.embeddings_cache),
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "gensim_available": GENSIM_AVAILABLE
        }