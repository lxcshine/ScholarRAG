# -*- coding: utf-8 -*-
"""
RF-Mem (Recollection-Familiarity Memory Retrieval)
Dual-path memory retriever inspired by human cognitive science

Core Mechanism:
  - Familiarity Path: Fast single retrieval for high-familiarity scenarios
  - Recollection Path: Iterative clustering extended retrieval for low-familiarity scenarios
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RFMemRetriever:
    """
    RF-Mem Dual-Path Memory Retriever
    
    Algorithm Flow:
      Phase 1: Probe Retrieval & Familiarity Signal
      Phase 2: Gating Strategy
      Phase 3: Familiarity Retrieval (Fast Path)
      Phase 4: Recollection Retrieval (Iterative Clustering Path)
    """

    def __init__(
        self,
        memory_embeddings: np.ndarray,
        memory_texts: List[Document],
        K: int = 10,
        # Gating parameters
        lambda_temp: float = 20.0,
        theta_high: float = 0.6,
        theta_low: float = 0.3,
        tau: float = 0.2,
        # Recollection parameters
        beam_width_B: int = 3,
        fanout_F: int = 2,
        max_rounds_R: int = 3,
        alpha: float = 0.5
    ):
        """
        Initialize RF-Mem Retriever
        
        Args:
            memory_embeddings: Memory vector matrix [N, D], should be unit-normalized
            memory_texts: Memory document list, corresponding to embeddings
            K: Number of Top-K results to return
            lambda_temp: Softmax temperature coefficient
            theta_high: High familiarity threshold
            theta_low: Low familiarity threshold
            tau: Entropy threshold
            beam_width_B: Beam width for exploration
            fanout_F: Fanout factor for retrieval expansion
            max_rounds_R: Maximum iteration rounds
            alpha: Alpha-mix coefficient for query update
        """
        self.memory_embeddings = self._normalize(memory_embeddings)
        self.memory_texts = memory_texts
        self.K = K
        
        self.lambda_temp = lambda_temp
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.tau = tau
        
        self.B = beam_width_B
        self.F = fanout_F
        self.R = max_rounds_R
        self.alpha = alpha
        
        logger.info(
            f"RF-Mem Retriever initialized | "
            f"Memory size: {len(memory_texts)} | K={K} | "
            f"Gating: theta_high={theta_high}, theta_low={theta_low}, tau={tau}"
        )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2 unit normalization
        
        Args:
            vectors: Input vector matrix [N, D]
            
        Returns:
            Normalized vector matrix
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        return vectors / norms

    def _calculate_familiarity_signal(
        self,
        query_emb: np.ndarray,
        top_k_scores: np.ndarray,
        top_k_indices: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Phase 1: Calculate familiarity signal (mean and entropy)
        
        Args:
            query_emb: Query vector [D]
            top_k_scores: Top-K cosine similarity scores [K]
            top_k_indices: Top-K indices [K]
            
        Returns:
            mean_score: Similarity mean
            entropy: Distribution information entropy H(p)
            softmax_probs: Softmax distribution p_i
        """
        K = len(top_k_scores)
        
        # Calculate mean similarity
        mean_score = np.mean(top_k_scores)
        
        # Calculate softmax distribution with temperature
        s_max = np.max(top_k_scores)
        scaled_scores = self.lambda_temp * (top_k_scores - s_max)
        
        exp_scores = np.exp(scaled_scores)
        softmax_probs = exp_scores / (np.sum(exp_scores) + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10))
        
        # Normalize entropy to [0, 1]
        max_entropy = np.log(K)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        logger.debug(
            f"Familiarity signal | Mean: {mean_score:.4f} | "
            f"Entropy: {normalized_entropy:.4f} (raw: {entropy:.4f})"
        )
        
        return mean_score, normalized_entropy, softmax_probs

    def _familiarity_path(
        self,
        top_k_scores: np.ndarray,
        top_k_indices: np.ndarray
    ) -> List[Tuple[Document, float]]:
        """
        Phase 3: Familiarity Fast Path
        
        Directly return Top-K results from Phase 1
        
        Args:
            top_k_scores: Top-K cosine similarity scores
            top_k_indices: Top-K indices
            
        Returns:
            Document list with scores
        """
        logger.info("Routing: Familiarity Path (fast retrieval)")
        
        results = []
        for score, idx in zip(top_k_scores, top_k_indices):
            idx = int(idx)
            if 0 <= idx < len(self.memory_texts):
                results.append((self.memory_texts[idx], float(score)))
        
        return results

    def _recollection_path(
        self,
        query_emb: np.ndarray
    ) -> List[Tuple[Document, float]]:
        """
        Phase 4: Recollection Iterative Clustering Extended Path
        
        Core flow: Retrieve -> Cluster -> Alpha-mix update query -> Iterate
        
        Args:
            query_emb: Initial query vector [D]
            
        Returns:
            Top-K documents and scores from Bag
        """
        logger.info(
            f"Routing: Recollection Path (iterative retrieval) | "
            f"Beam={self.B}, Fanout={self.F}, Rounds={self.R}"
        )
        
        beam = [query_emb.copy()]
        seen = set()
        bag = []
        
        for r in range(self.R):
            N = int((self.B + r) * self.F)
            logger.debug(f"  Round {r+1}/{self.R} | Retrieval count N={N}")
            
            new_queries = []
            new_clusters = []
            
            for q_idx, x_r in enumerate(beam):
                # Retrieve Top-N
                sims = self.memory_embeddings @ x_r
                
                # Filter seen items
                for idx in seen:
                    sims[idx] = -1.0
                
                top_N_indices = np.argsort(sims)[-N:][::-1]
                top_N_indices = [int(i) for i in top_N_indices if sims[i] > -0.99]
                
                if not top_N_indices:
                    continue
                
                candidate_vectors = self.memory_embeddings[top_N_indices]
                
                # KMeans clustering
                k_clusters = min(self.B, len(candidate_vectors))
                
                if k_clusters < 1:
                    continue
                
                kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
                labels = kmeans.fit_predict(candidate_vectors)
                centroids = kmeans.cluster_centers_
                
                # Alpha-mix query update
                for b in range(k_clusters):
                    cluster_mask = labels == b
                    cluster_elements = candidate_vectors[cluster_mask]
                    cluster_indices = np.array(top_N_indices)[cluster_mask]
                    
                    if len(cluster_elements) == 0:
                        continue
                    
                    # Calculate centroid
                    g_b = np.mean(cluster_elements, axis=0)
                    g_b = self._normalize(g_b.reshape(1, -1)).flatten()
                    
                    # Alpha-mix formula: x_new = norm(alpha * x_r + (1-alpha) * g_b + x_t)
                    x_new = self.alpha * x_r + (1 - self.alpha) * g_b + query_emb
                    x_new = self._normalize(x_new.reshape(1, -1)).flatten()
                    
                    # Score new query
                    cluster_sim_sum = np.sum(cluster_elements @ g_b)
                    
                    new_queries.append((x_new, cluster_sim_sum, q_idx))
                    new_clusters.append((cluster_indices, labels == b))
                    
                    # Add to bag and seen
                    for idx in cluster_indices:
                        if idx not in seen:
                            seen.add(int(idx))
                            sim = float(self.memory_embeddings[idx] @ query_emb)
                            bag.append((self.memory_texts[int(idx)], sim))
            
            if not new_queries:
                logger.debug("  No new queries generated, early termination")
                break
            
            # Select Top-B queries for next beam
            new_queries_sorted = sorted(new_queries, key=lambda x: x[1], reverse=True)
            beam = [q[0] for q in new_queries_sorted[:self.B]]
            
            # Early termination if bag is full
            if len(bag) >= self.K:
                logger.debug(f"  Bag full ({len(bag)} >= {self.K}), early termination")
                break
        
        # Return Top-K from bag
        bag_sorted = sorted(bag, key=lambda x: x[1], reverse=True)
        results = bag_sorted[:self.K]
        
        logger.info(f"  Recollection complete | Bag size: {len(bag)} | Returned: {len(results)}")
        
        return results

    def retrieve(
        self,
        query_emb: np.ndarray
    ) -> List[Tuple[Document, float]]:
        """
        RF-Mem main entry: Execute complete dual-path retrieval
        
        Args:
            query_emb: Query vector [D], will be L2 normalized automatically
            
        Returns:
            List[Tuple[Document, float]]: Document list with scores
        """
        # Normalize query
        query_emb = self._normalize(query_emb.reshape(1, -1)).flatten()
        
        # Phase 1: Probe retrieval
        sims = self.memory_embeddings @ query_emb
        top_k_indices = np.argsort(sims)[-self.K:][::-1]
        top_k_scores = sims[top_k_indices]
        
        # Calculate familiarity signal
        mean_score, entropy, softmax_probs = self._calculate_familiarity_signal(
            query_emb, top_k_scores, top_k_indices
        )
        
        # Phase 2: Gating strategy
        use_familiarity = (mean_score >= self.theta_high) or (entropy <= self.tau)
        use_recollection = (mean_score <= self.theta_low) or (entropy > self.tau)
        
        # Route to appropriate path
        if use_familiarity and not use_recollection:
            results = self._familiarity_path(top_k_scores, top_k_indices)
        elif use_recollection and not use_familiarity:
            results = self._recollection_path(query_emb)
        else:
            # Ambiguous case: prefer familiarity if mean is high
            if mean_score >= self.theta_high:
                results = self._familiarity_path(top_k_scores, top_k_indices)
            else:
                results = self._recollection_path(query_emb)
        
        return results[:self.K]

    def get_memory_embeddings(self) -> np.ndarray:
        """Get memory embeddings (for external access)"""
        return self.memory_embeddings

    def get_memory_texts(self) -> List[Document]:
        """Get memory documents (for external access)"""
        return self.memory_texts

    def add_to_memory(self, doc: Document, embedding: np.ndarray):
        """
        Dynamically add new memory to memory bank
        
        Args:
            doc: New document
            embedding: Corresponding embedding
        """
        embedding = self._normalize(embedding.reshape(1, -1)).flatten()
        self.memory_embeddings = np.vstack([self.memory_embeddings, embedding])
        self.memory_texts.append(doc)
        logger.debug(f"Memory expanded | Current size: {len(self.memory_texts)}")
