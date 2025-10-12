from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Literal

SUMMARY_RATIOS: Mapping[str, float] = {
    "high": 0.40,
    "medium": 0.20,
    "low": 0.10,
}

OFFLINE_MODELS: List[Mapping[str, object]] = [
    {
        "id": "distilbart-cnn",
        "label": "DistilBART-CNN (Abstractive)",
        "provider": "transformers",
        "model": "sshleifer/distilbart-cnn-12-6",
        "tokenizer": "sshleifer/distilbart-cnn-12-6",
        "max_input_tokens": 1024,
        "max_summary_tokens": 256,
        "chunk_overlap": 128,
        "description": "Fast abstractive summariser tuned on CNN/DailyMail reports.",
    },
    {
        "id": "t5-base",
        "label": "T5-Base (Text-to-Text)",
        "provider": "transformers",
        "model": "t5-base",
        "tokenizer": "t5-base",
        "max_input_tokens": 512,
        "max_summary_tokens": 200,
        "chunk_overlap": 96,
        "description": "Versatile sequence-to-sequence transformer ideal for concise briefs.",
    },
    {
        "id": "pegasus-xsum",
        "label": "Pegasus-XSum (Extreme Summarisation)",
        "provider": "transformers",
        "model": "google/pegasus-xsum",
        "tokenizer": "google/pegasus-xsum",
        "max_input_tokens": 1024,
        "max_summary_tokens": 128,
        "chunk_overlap": 128,
        "description": "High-precision abstractive summariser trained for short factual summaries.",
    },
    {
        "id": "longt5",
        "label": "LongT5 Local-Base (Long Context)",
        "provider": "transformers",
        "model": "google/long-t5-local-base",
        "tokenizer": "google/long-t5-local-base",
        "max_input_tokens": 4096,
        "max_summary_tokens": 512,
        "chunk_overlap": 512,
        "description": "Handles very long dossiers with local attention windows.",
    },
]

ONLINE_MODELS: List[Mapping[str, object]] = [
    {
        "id": "gpt-4o-mini",
        "label": "OpenAI GPT-4o Mini",
        "model": "gpt-4o-mini",
        "max_input_chars": 120_000,
    },
    {
        "id": "gpt-4o",
        "label": "OpenAI GPT-4o",
        "model": "gpt-4o",
        "max_input_chars": 120_000,
    },
    {
        "id": "gpt-4.1-mini",
        "label": "OpenAI GPT-4.1 Mini",
        "model": "gpt-4.1-mini",
        "max_input_chars": 120_000,
    },
    {
        "id": "gpt-4.1",
        "label": "OpenAI GPT-4.1",
        "model": "gpt-4.1",
        "max_input_chars": 120_000,
    },
    {
        "id": "o3-mini",
        "label": "OpenAI o3 Mini",
        "model": "o3-mini",
        "max_input_chars": 120_000,
    },
]

OFFLINE_MODEL_BY_ID: Dict[str, Mapping[str, object]] = {cfg["id"]: cfg for cfg in OFFLINE_MODELS}
ONLINE_MODEL_BY_ID: Dict[str, Mapping[str, object]] = {cfg["id"]: cfg for cfg in ONLINE_MODELS}

try:
    from transformers import AutoTokenizer, pipeline  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for offline usage
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizerBase

DetailLevel = Literal['high', 'medium', 'low']


def _chunk_text(
    text: str,
    tokenizer: "PreTrainedTokenizerBase",
    max_tokens: int,
    overlap: int,
) -> Iterable[str]:
    """Yield text chunks constrained by token budget with optional overlap."""
    token_ids = tokenizer.encode(text, truncation=False)
    if len(token_ids) <= max_tokens:
        yield text
        return

    step = max(1, max_tokens - overlap)
    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start:start + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            yield chunk_text


class Summarizer:
    """Coordinates offline Hugging Face models and online OpenAI chat models."""

    _offline_cache: Dict[str, Tuple[object, object]] = {}

    def __init__(self, mode: Literal['offline', 'online'], api_key: Optional[str] = None):
        self.mode = mode
        self.api_key = api_key
        self.client = None

        if mode == 'online':
            if not api_key:
                raise ValueError("API key required for online mode")
            from openai import OpenAI  # lazy import so offline mode has no dependency
            self.client = OpenAI(api_key=api_key)

    def summarize(
        self,
        text: str,
        detail_level: DetailLevel,
        engine_id: Optional[str] = None,
        progress_callback=None
    ) -> str:
        if not text or not text.strip():
            raise ValueError("No text provided for summarisation")

        if self.mode == 'offline':
            chosen_id = engine_id or OFFLINE_MODELS[0]["id"]
            return self._summarize_offline(text, detail_level, chosen_id, progress_callback)

        chosen_id = engine_id or ONLINE_MODELS[0]["id"]
        return self._summarize_online(text, detail_level, chosen_id, progress_callback)

    # ------------------------------------------------------------------
    # Offline transformer-based summarisation
    # ------------------------------------------------------------------
    def _summarize_offline(
        self,
        text: str,
        detail_level: DetailLevel,
        engine_id: str,
        progress_callback=None
    ) -> str:
        config = OFFLINE_MODEL_BY_ID.get(engine_id)
        if not config:
            raise ValueError(f"Unknown offline summarisation engine '{engine_id}'")

        if config.get("provider") != "transformers":
            raise ValueError(f"Unsupported offline provider '{config.get('provider')}'")

        if AutoTokenizer is None or pipeline is None:
            raise ImportError(
                "Offline models require the 'transformers' stack. Install dependencies with\n"
                "    pip install transformers sentencepiece torch accelerate"
            )

        if progress_callback:
            progress_callback(5, f"Loading {config['label']} model...")

        summarizer_pipe, tokenizer = self._load_transformer_pipeline(config)
        max_tokens = int(config.get("max_input_tokens", 1024))
        overlap = int(config.get("chunk_overlap", max_tokens // 6))
        chunks = list(_chunk_text(text, tokenizer, max_tokens, overlap))
        if not chunks:
            raise RuntimeError("Unable to build chunks for offline summarisation")

        ratio = SUMMARY_RATIOS[detail_level]
        max_summary_tokens = int(config.get("max_summary_tokens", max_tokens // 4))
        summaries: List[str] = []

        for index, chunk in enumerate(chunks, start=1):
            if progress_callback:
                progress_callback(
                    min(85, 10 + int((index - 1) / max(len(chunks), 1) * 65)),
                    f"Summarising chunk {index}/{len(chunks)}..."
                )

            chunk_tokens = len(tokenizer.encode(chunk, truncation=False))
            if chunk_tokens == 0:
                continue

            desired = max(48, int(math.ceil(chunk_tokens * ratio)))
            desired = min(desired, max_summary_tokens)
            min_tokens = max(32, int(desired * 0.6))
            if min_tokens >= desired:
                desired = min_tokens + 8

            response = summarizer_pipe(
                chunk,
                max_length=desired,
                min_length=min_tokens,
                do_sample=False,
                num_beams=4,
            )
            summary_text = response[0]["summary_text"].strip()
            if summary_text:
                summaries.append(summary_text)

        if not summaries:
            raise RuntimeError("Offline summarisation produced no output")

        combined = "\n\n".join(summaries).strip()
        if len(chunks) > 1 and progress_callback:
            progress_callback(90, "Consolidating chunk summaries...")

        combined_tokens = len(tokenizer.encode(combined, truncation=False))
        if len(chunks) > 1 and combined_tokens > max_summary_tokens:
            desired = min(max_summary_tokens, max(64, int(combined_tokens * ratio)))
            min_tokens = max(40, int(desired * 0.6))
            response = summarizer_pipe(
                combined,
                max_length=desired,
                min_length=min_tokens,
                do_sample=False,
                num_beams=4,
            )
            combined = response[0]["summary_text"].strip()

        if progress_callback:
            progress_callback(100, "Offline summary complete")

        if len(combined.split()) < 20:
            raise RuntimeError("Offline summary is too short; try a higher detail level")

        return combined

    def _load_transformer_pipeline(self, config: Mapping[str, object]):
        engine_id = str(config["id"])
        cached = self._offline_cache.get(engine_id)
        if cached:
            return cached

        model_name = str(config["model"])
        tokenizer_name = str(config.get("tokenizer", model_name))

        summarizer_pipe = pipeline(
            "summarization",
            model=model_name,
            tokenizer=tokenizer_name,
            truncation=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._offline_cache[engine_id] = (summarizer_pipe, tokenizer)
        return summarizer_pipe, tokenizer

    # ------------------------------------------------------------------
    # Online OpenAI chat-based summarisation
    # ------------------------------------------------------------------
    def _summarize_online(
        self,
        text: str,
        detail_level: DetailLevel,
        engine_id: str,
        progress_callback=None
    ) -> str:
        if not self.client:
            raise RuntimeError("Online summarisation requires an authenticated OpenAI client")

        config = ONLINE_MODEL_BY_ID.get(engine_id)
        if not config:
            raise ValueError(f"Unknown online summarisation engine '{engine_id}'")

        if progress_callback:
            progress_callback(15, "Preparing request for AI service...")

        max_chars = int(config.get("max_input_chars", 120_000))
        if len(text) > max_chars:
            if progress_callback:
                progress_callback(25, f"Truncating text to {max_chars} characters for API limits...")
            text = text[:max_chars]

        detail_instruction = {
            "high": "Provide a comprehensive strategic summary covering context, actors, and implications.",
            "medium": "Provide a balanced summary covering main points and supporting facts.",
            "low": "Provide a concise executive brief with only mission-critical points.",
        }[detail_level]

        model_name = str(config["model"])

        if progress_callback:
            progress_callback(40, "Contacting OpenAI...")

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at summarising military intelligence reports with precision and clarity.",
                    },
                    {
                        "role": "user",
                        "content": f"{detail_instruction}\n\nText to summarise:\n\n{text}",
                    },
                ],
                temperature=0.3,
                timeout=180,
            )
        except Exception as api_error:  # pragma: no cover - depends on external service
            message = str(api_error)
            lowered = message.lower()
            if "authentication" in lowered or "api key" in lowered:
                raise Exception("Invalid API key. Please verify your OpenAI credentials.") from api_error
            if "quota" in lowered or "rate limit" in lowered:
                raise Exception("OpenAI quota exceeded or rate limited. Try later or switch to offline models.") from api_error
            if "timeout" in lowered:
                raise Exception("OpenAI request timed out. Reduce input size or use offline mode.") from api_error
            raise Exception(f"OpenAI API error: {message}") from api_error

        if progress_callback:
            progress_callback(90, "Processing AI response...")

        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            raise Exception("No response received from AI service")

        summary = response.choices[0].message.content.strip()
        if len(summary) < 20:
            raise Exception("AI service returned an empty or too-short summary")

        if progress_callback:
            progress_callback(100, "Online summary complete")

        return summary
