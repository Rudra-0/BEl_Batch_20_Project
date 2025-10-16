from typing import Literal, Optional, List, Tuple
import numpy as np
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    # Some newer NLTK distributions separate punkt tables
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DetailLevel = Literal['high', 'medium', 'low']


class EmbeddingCentroidSummarizer:
    """Offline extractive summarizer using TF-IDF sentence embeddings + centroid + MMR.

    This module is intentionally isolated from the existing LSA-based summarizer.
    It exposes a simple API compatible with the project's detail levels.
    """

    def summarize(
        self,
        text: str,
        detail_level: DetailLevel,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Summarize input text using TF-IDF centroid ranking with MMR diversity.

        - detail_level: 'high'|'medium'|'low' controls target summary length
        - progress_callback: optional callable(percent:int, message:str)
        """
        if not text or not text.strip():
            raise ValueError("No text provided for summarization")

        if progress_callback:
            progress_callback(10, "Preparing text for summarization...")

        word_count = len(text.split())
        if word_count < 50:
            raise ValueError("Text is too short to summarize (minimum 50 words)")

        normalized = self._normalize_text(text)
        sentences = self._split_into_sentences(normalized)
        if len(sentences) < 3:
            raise ValueError("Not enough sentences to summarize (minimum 3 sentences)")

        target_sentences = self._compute_sentence_target(word_count, detail_level)

        if progress_callback:
            progress_callback(30, "Vectorizing sentences (TF-IDF)...")

        vectors, valid_sentences, original_indices = self._vectorize_sentences(sentences)
        if vectors.shape[0] == 0:
            raise ValueError("Vectorization produced empty features; input may be malformed")

        if progress_callback:
            progress_callback(60, "Scoring sentences via centroid similarity and selecting (MMR)...")

        centroid = vectors.mean(axis=0, keepdims=True)
        sim_to_centroid = cosine_similarity(vectors, centroid).ravel()

        selected_local_idx = self._mmr_select(
            vectors=vectors,
            base_scores=sim_to_centroid,
            k=target_sentences,
            lambda_diversity=0.6,
        )

        # Map back to original sentence indices and order by appearance
        selected_global_idx = [original_indices[i] for i in selected_local_idx]
        selected_global_idx.sort()

        if progress_callback:
            progress_callback(90, "Finalizing summary...")

        summary_sentences = [self._cleanup_sentence(sentences[i]) for i in selected_global_idx]
        # Remove any residual non-informative picks post-selection
        summary_sentences = [s for s in summary_sentences if self._is_informative_sentence(s)]
        summary_text = " ".join(summary_sentences).strip()
        if not summary_text or len(summary_text) < 20:
            raise ValueError("Summary is too short or empty")

        if progress_callback:
            progress_callback(100, "Summary complete!")

        return summary_text

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize raw text from PDFs to improve sentence tokenization.

        - Merge hyphenated line breaks ("-\n")
        - Replace line breaks with spaces
        - Collapse repeated whitespace
        """
        import re

        # Join hyphenated line breaks (end-of-line hyphens)
        text = re.sub(r"-\s*\n\s*", "", text)
        # Replace newlines/tabs with spaces
        text = re.sub(r"[\r\n\t]+", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        # Use NLTK's sentence tokenizer
        return [s.strip() for s in sent_tokenize(text) if s and s.strip()]

    @staticmethod
    def _compute_sentence_target(word_count: int, detail_level: DetailLevel) -> int:
        """Mirror existing sizing behavior by approximating sentences from words.

        Estimate sentence count by assuming ~20 words/sentence, then scale by
        the same ratios used elsewhere and enforce minimums.
        """
        if detail_level == 'high':
            sentence_count = max(int(word_count * 0.35 / 20), 10)
        elif detail_level == 'medium':
            sentence_count = max(int(word_count * 0.175 / 20), 5)
        else:
            sentence_count = max(int(word_count * 0.075 / 20), 3)
        return max(sentence_count, 1)

    @staticmethod
    def _vectorize_sentences(sentences: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
        """Vectorize sentences with TF-IDF. Remove empty/very short sentences prior to fit.

        Returns
        -------
        vectors : np.ndarray (n_sentences, n_features)
        valid_sentences : List[str] sentences that produced vectors
        original_indices : List[int] mapping from local row to original index
        """
        filtered = [(i, s) for i, s in enumerate(sentences) if EmbeddingCentroidSummarizer._is_informative_sentence(s)]
        if not filtered:
            return np.zeros((0, 0), dtype=float), [], []

        original_indices = [i for i, _ in filtered]
        valid_sentences = [s for _, s in filtered]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            sublinear_tf=True,
            min_df=1,
        )
        vectors = vectorizer.fit_transform(valid_sentences)
        # Return dense array for easier cosine ops with centroid
        return vectors.toarray().astype(np.float32), valid_sentences, original_indices

    @staticmethod
    def _mmr_select(
        vectors: np.ndarray,
        base_scores: np.ndarray,
        k: int,
        lambda_diversity: float = 0.7,
    ) -> List[int]:
        """Maximal Marginal Relevance selection.

        Selects k indices balancing relevance (base_scores) and diversity (distance
        from already selected items). Uses cosine similarity for both aspects.
        """
        n = vectors.shape[0]
        if k >= n:
            return list(range(n))

        # Precompute cosine similarity matrix for diversity term
        sim_matrix = cosine_similarity(vectors)

        selected: List[int] = []
        candidates = set(range(n))

        # Seed with the most relevant sentence
        first = int(np.argmax(base_scores))
        selected.append(first)
        candidates.remove(first)

        while len(selected) < k and candidates:
            mmr_scores = []
            for c in candidates:
                relevance = base_scores[c]
                diversity = max(sim_matrix[c, selected]) if selected else 0.0
                score = lambda_diversity * relevance - (1.0 - lambda_diversity) * diversity
                mmr_scores.append((score, c))
            mmr_scores.sort(reverse=True)
            _, best = mmr_scores[0]
            selected.append(best)
            candidates.remove(best)

        return selected

    @staticmethod
    def _is_informative_sentence(sentence: str) -> bool:
        """Heuristic filter to remove fragments and list-like noise.

        - At least 6 tokens
        - At least 60% alphabetic tokens
        - Ends with sentence punctuation or is reasonably long
        """
        import re

        if not sentence:
            return False
        tokens = sentence.strip().split()
        if len(tokens) < 6:
            return False
        alpha_tokens = sum(1 for t in tokens if re.search(r"[A-Za-z]", t))
        if alpha_tokens / max(len(tokens), 1) < 0.6:
            return False
        if len(sentence) < 40 and not re.search(r"[\.!?]$", sentence.strip()):
            # Very short and no terminal punctuation → likely a fragment
            return False
        return True

    @staticmethod
    def _cleanup_sentence(sentence: str) -> str:
        """Light cleanup for readability: fix spacing, ensure terminal punctuation."""
        import re

        s = sentence.strip()
        # Remove stray bullets/dashes at start
        s = re.sub(r"^[-•\*\u2022]\s*", "", s)
        # Collapse spaces around punctuation
        s = re.sub(r"\s+([,;:])", r"\1", s)
        s = re.sub(r"\s+([\.!?])$", r"\1", s)
        # Ensure a terminal period if none present and sentence is long enough
        if len(s) > 40 and not re.search(r"[\.!?]$", s):
            s = s + "."
        return s


