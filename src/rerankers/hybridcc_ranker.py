# hybrid_cc_ranker.py

from typing import List, Dict, Any

class HybridCCRanker:
    """
    Reranks search results from dense and sparse retrievers using the
    Hybrid Convex Combination (CC) fusion method with min-max normalization,
    as described in the AutoRAG paper (Bruch et al., 2023).

    This ranker combines scores from two sets of search results (typically dense
    and sparse) after normalizing them using min-max scaling. The final score
    is a weighted average based on the provided alpha value.

    Formula:
    f_Conver(q, d) = alpha * phi_LEX(f_LEX(q, d)) + (1 - alpha) * phi_SEM(f_SEM(q, d))

    where phi is the min-max normalization function:
    phi_MM(f_o(q, d)) = (f_o(q, d) - m_o(q)) / (M_o(q) - m_o(q))

    Attributes:
        alpha (float): The weight assigned to the sparse retrieval score (phi_LEX).
                       The weight for the dense score (phi_SEM) is (1 - alpha).
    """
    def __init__(self, alpha: float = 0.5):
        """
        Initializes the HybridCCRanker.

        Args:
            alpha: The weight parameter for the sparse retrieval score (f_LEX).
                   Must be between 0 and 1. Defaults to 0.5 (equal weighting).

        Raises:
            ValueError: If alpha is not between 0 and 1.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        # Consider adding logging instead of print for better integration
        # import logging
        # logging.info(f"Initialized HybridCCRanker with alpha = {self.alpha}")
        print(f"Initialized HybridCCRanker with alpha = {self.alpha}") # Keep print for now based on previous code

    def _min_max_normalize(self, score: float, min_score: float, max_score: float) -> float:
        """
        Applies min-max normalization to a single score.

        Scales the score to be within the range [0, 1].

        Args:
            score: The score to normalize.
            min_score: The minimum score observed in the result set.
            max_score: The maximum score observed in the result set.

        Returns:
            The normalized score (between 0 and 1). Returns 0.5 if all scores
            in the set were identical (max_score == min_score).
        """
        if max_score == min_score:
            # Avoid division by zero if all scores are the same
            # Returning 0.5 places it neutrally in the 0-1 range.
            # Alternatively, return 0 or 1 depending on interpretation.
            return 0.5
        # Ensure score does not exceed bounds due to floating point issues maybe? (Optional)
        # score = max(min_score, min(score, max_score))
        return (score - min_score) / (max_score - min_score)

    def rerank(self, dense_results: List[List[Dict]], sparse_results: List[List[Dict]], top_k: int = 5) -> List[Dict]:
        """
        Reranks the combined results from dense and sparse searches using Hybrid CC.

        Args:
            dense_results: Results from the dense vector search (e.g., Milvus search output).
                           Expected format: [[{'id': ..., 'distance': score, 'entity': {...}}]].
                           Assumes 'distance' represents a similarity/relevance score
                           where higher values indicate greater relevance.
            sparse_results: Results from the sparse vector search (e.g., Milvus search output).
                            Expected format: [[{'id': ..., 'distance': score, 'entity': {...}}]].
                            Assumes 'distance' represents a similarity/relevance score
                            where higher values indicate greater relevance.
            top_k: The maximum number of results to return after reranking.

        Returns:
            A list of reranked documents, sorted by the fused Convex Combination
            score in descending order (highest score first). Limited to top_k items.
            Each item in the list is a dictionary:
            {'id': str, 'score': float, 'entity': dict, ...}.
            May include internal debug scores if needed.

        Raises:
            TypeError: If input results are not in the expected list-of-lists format.
            KeyError: If hits within the results are missing 'id' or 'distance'.
        """
        # --- Input Validation and Parsing ---
        try:
            # Expecting [[hit1, hit2,...]] for a single query result
            dense_hits = dense_results[0] if dense_results and isinstance(dense_results[0], list) else []
        except (IndexError, TypeError):
            # Log this potential issue
            print("Warning: Dense results are empty or not in the expected format [[...]].")
            dense_hits = []

        try:
            sparse_hits = sparse_results[0] if sparse_results and isinstance(sparse_results[0], list) else []
        except (IndexError, TypeError):
            print("Warning: Sparse results are empty or not in the expected format [[...]].")
            sparse_hits = []

        # --- 1. Combine results and extract scores ---
        combined_docs: Dict[str, Dict[str, Any]] = {}
        # Structure: {doc_id: {'dense_score': float, 'sparse_score': float, 'entity': dict}}

        all_dense_scores = []
        for hit in dense_hits:
            try:
                doc_id = hit['id']
                # *** CRITICAL ASSUMPTION: 'distance' is a score where higher is better ***
                # If 'distance' means lower is better, convert it here. E.g.:
                # score = 1.0 / (1.0 + hit['distance']) # If distance >= 0
                # score = -hit['distance']             # If distance can be negative/positive
                score = float(hit['distance']) # Ensure it's a float
                all_dense_scores.append(score)

                if doc_id not in combined_docs:
                    # Store the full entity data or just specific fields as needed
                    combined_docs[doc_id] = {'entity': hit.get('entity', {})}
                combined_docs[doc_id]['dense_score'] = score
            except KeyError as e:
                print(f"Warning: Skipping dense hit due to missing key: {e}. Hit: {hit}")
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping dense hit due to invalid score type: {e}. Hit: {hit}")


        all_sparse_scores = []
        for hit in sparse_hits:
            try:
                doc_id = hit['id']
                # *** CRITICAL ASSUMPTION: 'distance' is a score where higher is better ***
                score = float(hit['distance']) # Ensure it's a float
                all_sparse_scores.append(score)

                if doc_id not in combined_docs:
                    combined_docs[doc_id] = {'entity': hit.get('entity', {})}
                combined_docs[doc_id]['sparse_score'] = score
            except KeyError as e:
                print(f"Warning: Skipping sparse hit due to missing key: {e}. Hit: {hit}")
            except (ValueError, TypeError) as e:
                 print(f"Warning: Skipping sparse hit due to invalid score type: {e}. Hit: {hit}")

        if not combined_docs:
             print("No documents found in either dense or sparse results.")
             return []

        # --- 2. Calculate Min/Max for Normalization ---
        # Handle cases where one or both result sets might be empty
        min_dense = min(all_dense_scores) if all_dense_scores else 0.0
        max_dense = max(all_dense_scores) if all_dense_scores else 0.0
        min_sparse = min(all_sparse_scores) if all_sparse_scores else 0.0
        max_sparse = max(all_sparse_scores) if all_sparse_scores else 0.0

        # --- 3. Normalize and Fuse Scores ---
        fused_results = []
        for doc_id, data in combined_docs.items():
            dense_score = data.get('dense_score')
            sparse_score = data.get('sparse_score')

            # Normalize dense score (phi_SEM)
            # If a document is missing from dense results, its dense_score will be None.
            # Assign normalized score of 0 if missing or normalization isn't possible.
            norm_dense = 0.0
            if dense_score is not None and all_dense_scores: # Check if score exists and normalization possible
                try:
                    norm_dense = self._min_max_normalize(dense_score, min_dense, max_dense)
                except Exception as e: # Catch potential math errors, though _min_max_normalize handles division by zero
                    print(f"Warning: Error normalizing dense score for doc {doc_id}: {e}")
                    norm_dense = 0.0 # Default on error

            # Normalize sparse score (phi_LEX)
            norm_sparse = 0.0
            if sparse_score is not None and all_sparse_scores:
                try:
                    norm_sparse = self._min_max_normalize(sparse_score, min_sparse, max_sparse)
                except Exception as e:
                    print(f"Warning: Error normalizing sparse score for doc {doc_id}: {e}")
                    norm_sparse = 0.0

            # Calculate fused score using Convex Combination
            fused_score = self.alpha * norm_sparse + (1 - self.alpha) * norm_dense

            # Append result with fused score and original entity data
            result_item = {
                'id': doc_id,
                'score': fused_score, # The final reranked score
                'entity': data['entity'],
                 # Optional: Include debug info if helpful during development
                 # '_debug_dense_orig': dense_score,
                 # '_debug_sparse_orig': sparse_score,
                 # '_debug_dense_norm': norm_dense,
                 # '_debug_sparse_norm': norm_sparse
            }
            fused_results.append(result_item)

        # --- 4. Sort by fused score ---
        # Sorts in place, descending order (highest score first)
        fused_results.sort(key=lambda x: x['score'], reverse=True)

        # --- 5. Return top_k results ---
        return fused_results[:top_k]
