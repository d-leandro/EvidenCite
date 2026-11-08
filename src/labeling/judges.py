"""Cliente de inferencia estructurada para jueces LLM con caché append-only."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from src.data.schemas import DatasetInstance
from src.data.taxonomy import CANONICAL_9_CLASSES
from src.labeling.prompts import PROMPT_VERSION, PROMPT_VERSIONS, build_judge_messages
from src.labeling.schemas import JudgeConfig, JudgeVote, PanelVote

logger = logging.getLogger(__name__)

_LABEL_LOOKUP = {re.sub(r"[\s_-]+", "", label.lower()): label for label in CANONICAL_9_CLASSES}


def clean_markdown_json(raw_text: str) -> str:
    """Remueve bloques delimitadores ```json ... ``` si el LLM emitió markdown."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0).strip() if match else text


def parse_vote_text(raw_text: str) -> JudgeVote:
    """Parseo tolerante de una respuesta en texto plano hacia JudgeVote (fallback de RF-04)."""
    data = json.loads(clean_markdown_json(raw_text))
    if not isinstance(data, dict):
        raise ValueError("La respuesta no es un objeto JSON")
    raw_label = str(data.get("label", "")).strip()
    label = _LABEL_LOOKUP.get(re.sub(r"[\s_-]+", "", raw_label.lower()))
    if label is None:
        raise ValueError(f"Etiqueta fuera de la taxonomía canónica: {raw_label!r}")
    rationale = str(data.get("rationale", "")).strip()[:300]
    return JudgeVote(rationale=rationale, label=label)


class Judge:
    """Encapsula la interacción con un modelo individual del panel."""

    def __init__(
        self,
        config: JudgeConfig,
        cache_dir: Path,
        client: Optional[Any] = None,
        force: bool = False,
        cache_only: bool = False,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"Versión de prompt desconocida: {prompt_version!r}")
        self.config = config
        self.prompt_version = prompt_version
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{self.config.id}.jsonl"
        self.cache_only = cache_only
        self._cache: Dict[str, PanelVote] = {}
        self._abstentions: Dict[str, PanelVote] = {}
        self._lock = threading.Lock()
        if not force:
            self._load_cache()

        if client is None:
            api_key = os.getenv(self.config.api_key_env, "")
            client = OpenAI(
                base_url=self.config.base_url,
                api_key=api_key or "dummy_key",
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
            )
        self.client = client

    def _cache_key(self, instance_id: str) -> str:
        return f"{instance_id}::{self.config.model_name}::{self.prompt_version}"

    def _load_cache(self) -> None:
        if not self.cache_file.exists():
            return
        with open(self.cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    vote = PanelVote.model_validate(record["vote"])
                except Exception:
                    continue
                if vote.label is not None:
                    self._cache[record["cache_key"]] = vote
                else:
                    self._abstentions[record["cache_key"]] = vote
        logger.info("Juez [%s]: %d votos recuperados desde caché", self.config.id, len(self._cache))

    def _append_to_cache(self, key: str, vote: PanelVote) -> None:
        with self._lock:
            self._cache[key] = vote
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"cache_key": key, "vote": vote.model_dump()}, ensure_ascii=False) + "\n")

    def is_cached(self, instance_id: str) -> bool:
        return self._cache_key(instance_id) in self._cache

    def _request(self, messages: List[Dict[str, str]]) -> Tuple[JudgeVote, int, int]:
        """Intenta structured outputs nativo y, si falla, una llamada en texto plano con parseo tolerante."""
        completions = self.client.chat.completions
        parse = getattr(completions, "parse", None) or self.client.beta.chat.completions.parse
        try:
            completion = parse(
                model=self.config.model_name,
                messages=messages,
                response_format=JudgeVote,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
            )
            message = completion.choices[0].message
            parsed = getattr(message, "parsed", None)
            vote = parsed if isinstance(parsed, JudgeVote) else parse_vote_text(message.content or "")
        except Exception as primary_error:
            logger.debug("Juez [%s]: structured outputs falló (%s); se intenta texto plano", self.config.id, primary_error)
            completion = completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
            )
            vote = parse_vote_text(completion.choices[0].message.content or "")

        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return vote, prompt_tokens, completion_tokens

    def vote(self, instance: DatasetInstance) -> PanelVote:
        """Emite el voto con estructuración Pydantic, fallback tolerante y registro en caché."""
        key = self._cache_key(instance.instance_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self.cache_only:
            return self._abstentions.get(key) or PanelVote(
                judge_id=self.config.id,
                model_name=self.config.model_name,
                provider=self.config.provider,
                error="Sin voto en caché",
            )

        messages = build_judge_messages(instance, self.prompt_version)
        t0 = time.perf_counter()
        try:
            parsed, prompt_tokens, completion_tokens = self._request(messages)
            vote = PanelVote(
                judge_id=self.config.id,
                model_name=self.config.model_name,
                provider=self.config.provider,
                label=parsed.label,
                rationale=parsed.rationale,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            logger.warning("Juez [%s]: abstención en %s (%s)", self.config.id, instance.instance_id, e)
            vote = PanelVote(
                judge_id=self.config.id,
                model_name=self.config.model_name,
                provider=self.config.provider,
                label=None,
                rationale=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=str(e)[:500],
            )

        self._append_to_cache(key, vote)
        return vote

    def vote_all(
        self,
        instances: List[DatasetInstance],
        progress_desc: Optional[str] = None,
        log_every: int = 200,
    ) -> Dict[str, PanelVote]:
        """Ejecuta votaciones con ThreadPoolExecutor (max_concurrency) preservando orden."""
        desc = progress_desc or self.config.id
        total = len(instances)
        pending = sum(1 for inst in instances if not self.is_cached(inst.instance_id))
        logger.info("Juez [%s]: %d instancias (%d en caché, %d por consultar)", desc, total, total - pending, pending)

        results: Dict[str, PanelVote] = {}
        done = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as pool:
            for inst, vote in zip(instances, pool.map(self.vote, instances)):
                results[inst.instance_id] = vote
                done += 1
                if done % log_every == 0 or done == total:
                    logger.info(
                        "Juez [%s]: %d/%d (%.1f%%) en %.1fs",
                        desc, done, total, 100.0 * done / total, time.perf_counter() - t0,
                    )
        return results


def build_judges(
    configs: List[JudgeConfig],
    cache_dir: Path,
    force: bool = False,
    cache_only: bool = False,
    prompt_version: str = PROMPT_VERSION,
) -> List[Judge]:
    """Fabrica la lista de jueces configurados inicializando sus cachés correspondientes."""
    return [
        Judge(cfg, cache_dir, force=force, cache_only=cache_only, prompt_version=prompt_version)
        for cfg in configs
    ]
