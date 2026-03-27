"""
Reliable dataset-backed benchmarks for local model evaluation.
"""

from __future__ import annotations

import ast
import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


@dataclass(frozen=True)
class BenchmarkCatalogEntry:
    """Metadata for both runnable and tracked-only benchmarks."""

    key: str
    display_name: str
    description: str
    category: str
    status: str
    harness: str
    local_runnable: bool
    dataset_name: Optional[str] = None
    config_name: Optional[str] = None
    split: Optional[str] = None
    labs: Tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert catalog metadata to a JSON-friendly shape."""
        return {
            "key": self.key,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "status": self.status,
            "harness": self.harness,
            "local_runnable": self.local_runnable,
            "dataset_name": self.dataset_name,
            "config_name": self.config_name,
            "split": self.split,
            "labs": list(self.labs),
            "notes": self.notes,
        }

MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _safe_load_dataset(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
) -> Iterable[Dict[str, Any]]:
    """Load a dataset in streaming mode to keep local runs lightweight."""
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {
        "split": split,
        "streaming": True,
    }

    try:
        if config_name:
            return load_dataset(dataset_name, config_name, **kwargs)
        return load_dataset(dataset_name, **kwargs)
    except Exception as exc:
        message = str(exc)
        if "trust_remote_code=True" not in message and "contains custom code" not in message:
            raise

    kwargs["trust_remote_code"] = True
    if config_name:
        return load_dataset(dataset_name, config_name, **kwargs)
    return load_dataset(dataset_name, **kwargs)


def _take_samples(dataset: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Take a bounded number of items from a streaming dataset."""
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if idx >= limit:
            break
        rows.append(row)
    return rows


def _take_filtered_samples(
    dataset: Iterable[Dict[str, Any]],
    limit: int,
    predicate: Callable[[Dict[str, Any]], bool],
) -> List[Dict[str, Any]]:
    """Take a bounded number of rows that satisfy a filter predicate."""
    rows: List[Dict[str, Any]] = []
    for row in dataset:
        if not predicate(row):
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _extract_boxed_content(text: str) -> str:
    """Extract the contents of a LaTeX boxed answer."""
    marker = "\\boxed{"
    start = text.find(marker)
    if start == -1:
        return ""

    index = start + len(marker)
    depth = 1
    content: List[str] = []

    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        content.append(char)
        index += 1

    return "".join(content).strip()


def _extract_final_number(text: str) -> str:
    """Extract a final numeric answer from a response."""
    if not text:
        return ""

    boxed = _extract_boxed_content(text)
    if boxed:
        text = boxed

    explicit_patterns = [
        r"final answer\s*[:=]\s*([^\n]+)",
        r"answer\s*[:=]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", candidate.replace(",", ""))
            if numbers:
                return numbers[-1]

    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else ""


def _extract_final_text_answer(text: str) -> str:
    """Extract a best-effort final text answer from a response."""
    if not text:
        return ""

    boxed = _extract_boxed_content(text)
    if boxed:
        return boxed

    explicit_patterns = [
        r"final answer\s*[:=]\s*([^\n]+)",
        r"answer\s*[:=]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _normalize_text_answer(value: str) -> str:
    """Normalize a text answer for exact-comparison fallbacks."""
    cleaned = value.strip().strip(".")
    cleaned = cleaned.replace("$", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.strip("{}()[]")
    return cleaned.lower()


def _strip_code_fences(text: str) -> str:
    """Remove surrounding fenced-code markers when models wrap raw answers."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _normalize_multiline_text(text: str) -> str:
    """Normalize multiline answers while preserving meaningful newlines."""
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*(assistant|answer|final answer)\s*:\s*", "", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = _strip_code_fences(cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.strip().strip('"').strip("'")
    lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip()


def _extract_last_list_literal(text: str) -> str:
    """Extract the last simple list literal from a response."""
    final_matches = re.findall(r"final answer\s*:\s*(\[[^\n]*\])", text, flags=re.IGNORECASE)
    if final_matches:
        return final_matches[-1]

    generic_matches = re.findall(r"(\[[^\n]*\])", text)
    return generic_matches[-1] if generic_matches else ""


def _parse_string_list(text: str) -> List[str]:
    """Parse a flat list literal like `[a, b]` into string items."""
    literal = _extract_last_list_literal(text)
    if not literal:
        return []

    try:
        parsed = ast.literal_eval(literal)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass

    inner = literal.strip()[1:-1].strip()
    if not inner:
        return []

    items: List[str] = []
    for part in inner.split(","):
        item = part.strip().strip('"').strip("'")
        if item:
            items.append(item)
    return items


def _normalize_string_list(items: Sequence[str]) -> List[str]:
    """Normalize list items for exact set-style comparisons."""
    normalized: List[str] = []
    seen = set()
    for item in items:
        token = str(item).strip().strip('"').strip("'")
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return sorted(normalized)


def _render_chat_prompt(serialized_prompt: str) -> str:
    """Render a JSON chat transcript as a plain-text prompt for local models."""
    try:
        messages = json.loads(serialized_prompt)
    except json.JSONDecodeError:
        return serialized_prompt

    if not isinstance(messages, list):
        return serialized_prompt

    rendered: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        rendered.append(f"{role}: {content}")

    if not rendered:
        return serialized_prompt

    rendered.append("Assistant:")
    return "\n\n".join(rendered)


def _normalize_choice_response(response: str, labels: Sequence[str]) -> str:
    """Normalize a multiple-choice response to one of the expected labels."""
    if not response:
        return ""

    canonical_labels = [label.strip().upper() for label in labels]
    response_upper = response.strip().upper()

    for label in canonical_labels:
        if re.search(rf"\b{re.escape(label)}\b", response_upper):
            return label

    letters = [label for label in canonical_labels if len(label) == 1 and label.isalpha()]
    if letters:
        letter_match = re.search(r"\b([A-Z])\b", response_upper)
        if letter_match and letter_match.group(1) in letters:
            return letter_match.group(1)

    numbers = [label for label in canonical_labels if label.isdigit()]
    if numbers:
        number_match = re.search(r"\b(\d+)\b", response_upper)
        if number_match and number_match.group(1) in numbers:
            return number_match.group(1)

    if response_upper and response_upper[0] in canonical_labels:
        return response_upper[0]

    return ""


def _normalize_boolean_response(response: str) -> str:
    """Normalize a boolean or yes/no style answer."""
    if not response:
        return ""

    response_lower = response.strip().lower()
    response_lower = _extract_final_text_answer(response_lower)

    positive = ["yes", "true", "correct", "1"]
    negative = ["no", "false", "incorrect", "0"]

    for token in positive:
        if re.search(rf"\b{re.escape(token)}\b", response_lower):
            return "yes" if token in {"yes", "correct", "1"} else "true"

    for token in negative:
        if re.search(rf"\b{re.escape(token)}\b", response_lower):
            return "no" if token in {"no", "incorrect", "0"} else "false"

    return ""


def _numeric_equal(left: str, right: str, tolerance: float = 1e-6) -> bool:
    """Compare numeric strings with a small tolerance."""
    try:
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)
    except (TypeError, ValueError):
        return left.strip() == right.strip()


def _math_equal(left: str, right: str) -> bool:
    """Best-effort exact comparison for MATH-style answers."""
    if _numeric_equal(left, right, tolerance=1e-6):
        return True
    return _normalize_text_answer(left) == _normalize_text_answer(right)


def summarize_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-sample records into benchmark-level metrics."""
    sample_count = len(samples)
    correct_count = sum(1 for sample in samples if sample.get("is_correct"))
    total_errors = sum(1 for sample in samples if sample.get("error"))
    total_prompt_tokens = sum(int(sample.get("prompt_tokens") or 0) for sample in samples)
    total_completion_tokens = sum(int(sample.get("completion_tokens") or 0) for sample in samples)
    total_tokens = sum(int(sample.get("total_tokens") or 0) for sample in samples)
    latencies = [float(sample.get("latency_sec") or 0.0) for sample in samples]
    tps = [
        float(sample.get("tokens_per_second") or 0.0)
        for sample in samples
        if float(sample.get("tokens_per_second") or 0.0) > 0
    ]

    return {
        "sample_count": sample_count,
        "correct_count": correct_count,
        "accuracy": round(correct_count / sample_count, 4) if sample_count else 0.0,
        "error_count": total_errors,
        "avg_latency_sec": round(sum(latencies) / sample_count, 4) if sample_count else 0.0,
        "max_latency_sec": round(max(latencies), 4) if latencies else 0.0,
        "min_latency_sec": round(min(latencies), 4) if latencies else 0.0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_second": round(sum(tps) / len(tps), 4) if tps else 0.0,
        "local_cost_estimate": 0.0,
    }


@dataclass
class BenchmarkExecution:
    """Result of one benchmark against one model."""

    benchmark_result: Dict[str, Any]
    samples: List[Dict[str, Any]]


@dataclass
class LocalGenerationConfig:
    """Lightweight generation config used by the local pipeline."""

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None


class DatasetBenchmark(ABC):
    """Base class for local dataset-backed benchmarks."""

    key: str
    display_name: str
    dataset_name: str
    config_name: Optional[str]
    split: str
    description: str
    category: str = "general"
    max_tokens: int = 128

    def load_rows(self, limit: int) -> List[Dict[str, Any]]:
        """Load a bounded sample from the benchmark dataset."""
        dataset = _safe_load_dataset(self.dataset_name, self.config_name, self.split)
        return _take_samples(dataset, limit)

    @abstractmethod
    def build_prompt(self, row: Dict[str, Any]) -> str:
        """Build the model prompt for a dataset row."""

    @abstractmethod
    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        """Extract the benchmark reference answer."""

    @abstractmethod
    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        """Parse the model response into a comparable answer."""

    @abstractmethod
    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Return benchmark-specific sample metadata."""

    def is_correct(self, predicted: str, expected: str) -> bool:
        """Compare a prediction and reference answer."""
        return predicted == expected

    async def evaluate(
        self,
        model: Any,
        model_info: Dict[str, Any],
        run_id: str,
        num_samples: int,
        temperature: float = 0.0,
        progress_callback: ProgressCallback = None,
    ) -> BenchmarkExecution:
        """Evaluate a model on this benchmark and return summary plus samples."""
        rows = self.load_rows(num_samples)

        generation_config = LocalGenerationConfig(
            temperature=temperature,
            max_tokens=self.max_tokens,
            top_p=0.9,
            stop_sequences=[],
        )

        samples: List[Dict[str, Any]] = []
        running_correct = 0
        running_errors = 0

        for index, row in enumerate(rows):
            prompt = self.build_prompt(row)
            expected = self.extract_expected_answer(row)
            generation = await model.generate_with_metadata(prompt, generation_config)
            predicted = self.parse_prediction(generation.text, row)
            correct = self.is_correct(predicted, expected)
            error = generation.raw.get("error") if generation.raw else None

            if correct:
                running_correct += 1
            if error:
                running_errors += 1

            sample = {
                "run_id": run_id,
                "benchmark_name": self.key,
                "sample_index": index,
                "model_name": model_info["name"],
                "provider": model_info["provider"],
                "prompt": prompt,
                "response_text": generation.text,
                "expected_answer": expected,
                "parsed_prediction": predicted,
                "is_correct": correct,
                "error": error,
                "started_at": generation.started_at,
                "ended_at": generation.ended_at,
                "latency_sec": round(generation.latency_sec, 6),
                "load_duration_sec": round(generation.load_duration_sec, 6),
                "prompt_eval_duration_sec": round(generation.prompt_eval_duration_sec, 6),
                "eval_duration_sec": round(generation.eval_duration_sec, 6),
                "total_duration_sec": round(generation.total_duration_sec, 6),
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.total_tokens,
                "tokens_per_second": round(generation.tokens_per_second, 6),
                "raw_provider_metrics": generation.raw,
            }
            sample.update(self.sample_metadata(row))
            samples.append(sample)

            if progress_callback:
                progress_callback(
                    {
                        "event": "sample_completed",
                        "run_id": run_id,
                        "model_name": model_info["name"],
                        "benchmark_name": self.key,
                        "sample_index": index + 1,
                        "total_samples": len(rows),
                        "correct_count": running_correct,
                        "error_count": running_errors,
                        "sample": sample,
                    }
                )

        metrics = summarize_samples(samples)
        benchmark_result = {
            "schema_version": "2.0",
            "run_id": run_id,
            "benchmark_name": self.key,
            "benchmark_display_name": self.display_name,
            "description": self.description,
            "dataset": {
                "name": self.dataset_name,
                "config_name": self.config_name,
                "split": self.split,
            },
            "model": model_info,
            "metrics": metrics,
            "status": "completed",
        }

        if progress_callback:
            progress_callback(
                {
                    "event": "benchmark_completed",
                    "run_id": run_id,
                    "model_name": model_info["name"],
                    "benchmark_name": self.key,
                    "benchmark_result": benchmark_result,
                }
            )

        return BenchmarkExecution(benchmark_result=benchmark_result, samples=samples)


class GSM8KBenchmark(DatasetBenchmark):
    key = "gsm8k"
    display_name = "GSM8K"
    dataset_name = "gsm8k"
    config_name = "main"
    split = "test"
    description = "Grade-school math word problems."
    category = "reasoning"
    max_tokens = 256

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return (
            "Solve the math problem. You can reason briefly, but end with a final line in the "
            "exact format 'Final answer: <number>'.\n\n"
            f"Question: {row['question']}\n"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return _extract_final_number(row["answer"])

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _extract_final_number(response_text)

    def is_correct(self, predicted: str, expected: str) -> bool:
        return _numeric_equal(predicted, expected, tolerance=1e-2)

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {"question": row["question"]}


class MMLUBenchmark(DatasetBenchmark):
    key = "mmlu"
    display_name = "MMLU"
    dataset_name = "cais/mmlu"
    config_name = "all"
    split = "test"
    description = "Massive multitask language understanding multiple-choice questions."
    category = "knowledge"
    max_tokens = 32

    def build_prompt(self, row: Dict[str, Any]) -> str:
        choices = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(row["choices"]))
        return (
            "Answer the multiple-choice question. Reply with only one letter: A, B, C, or D.\n\n"
            f"Question: {row['question']}\n"
            f"{choices}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return chr(65 + int(row["answer"]))

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _normalize_choice_response(response_text, ["A", "B", "C", "D"])

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": row["question"],
            "subject": row.get("subject", "unknown"),
            "choices": row["choices"],
        }


class MMLUProBenchmark(DatasetBenchmark):
    key = "mmlu_pro"
    display_name = "MMLU-Pro"
    dataset_name = "TIGER-Lab/MMLU-Pro"
    config_name = "default"
    split = "test"
    description = "Harder multiple-choice MMLU-Pro benchmark."
    category = "knowledge"
    max_tokens = 48

    def build_prompt(self, row: Dict[str, Any]) -> str:
        labels = [chr(65 + idx) for idx in range(len(row["options"]))]
        choices = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, row["options"]))
        return (
            "Answer the multiple-choice question. Reply with only the option letter.\n\n"
            f"Question: {row['question']}\n"
            f"{choices}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return str(row["answer"]).upper()

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        labels = [chr(65 + idx) for idx in range(len(row["options"]))]
        return _normalize_choice_response(response_text, labels)

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": row["question"],
            "category": row.get("category", "unknown"),
            "source": row.get("src", "unknown"),
            "choices": row["options"],
        }


class HendrycksMathBenchmark(DatasetBenchmark):
    key = "math"
    display_name = "MATH"
    dataset_name = "EleutherAI/hendrycks_math"
    config_name = "all"
    split = "test"
    description = "Competition-level mathematics from the Hendrycks MATH benchmark."
    category = "reasoning"
    max_tokens = 384

    def load_rows(self, limit: int) -> List[Dict[str, Any]]:
        """Round-robin across the seven MATH configs for balanced sampling."""
        iterators: List[Iterator[Dict[str, Any]]] = [
            iter(_safe_load_dataset(self.dataset_name, config_name, self.split))
            for config_name in MATH_CONFIGS
        ]
        config_names = list(MATH_CONFIGS)
        rows: List[Dict[str, Any]] = []
        index = 0

        while iterators and len(rows) < limit:
            iterator = iterators[index]
            config_name = config_names[index]
            try:
                row = dict(next(iterator))
                row["_math_config"] = config_name
                rows.append(row)
                index = (index + 1) % len(iterators)
            except StopIteration:
                iterators.pop(index)
                config_names.pop(index)
                if iterators:
                    index %= len(iterators)

        return rows

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return (
            "Solve the math problem. You may reason step by step, but end with a final line in the "
            "exact format 'Final answer: <answer>'.\n\n"
            f"Problem: {row['problem']}\n"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return _extract_final_text_answer(row["solution"])

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _extract_final_text_answer(response_text)

    def is_correct(self, predicted: str, expected: str) -> bool:
        return _math_equal(predicted, expected)

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "problem": row["problem"],
            "level": row.get("level"),
            "type": row.get("type"),
            "math_config": row.get("_math_config"),
        }


class AIME2024Benchmark(DatasetBenchmark):
    key = "aime_2024"
    display_name = "AIME 2024"
    dataset_name = "Maxwell-Jia/AIME_2024"
    config_name = None
    split = "train"
    description = "American Invitational Mathematics Examination 2024 problems."
    category = "reasoning"
    max_tokens = 160

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return (
            "Solve the AIME problem. You may reason briefly, but end with a final line in the "
            "exact format 'Final answer: <integer>'.\n\n"
            f"Problem: {row['Problem']}\n"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return str(row["Answer"])

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _extract_final_number(response_text)

    def is_correct(self, predicted: str, expected: str) -> bool:
        return _numeric_equal(predicted, expected, tolerance=0.0)

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "problem": row["Problem"],
            "problem_id": row.get("ID"),
        }


class AIME2025Benchmark(DatasetBenchmark):
    key = "aime_2025"
    display_name = "AIME 2025"
    dataset_name = "MathArena/aime_2025"
    config_name = None
    split = "train"
    description = "American Invitational Mathematics Examination 2025 problems."
    category = "reasoning"
    max_tokens = 160

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return (
            "Solve the AIME problem. You may reason briefly, but end with a final line in the "
            "exact format 'Final answer: <integer>'.\n\n"
            f"Problem: {row['problem']}\n"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return str(row["answer"])

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _extract_final_number(response_text)

    def is_correct(self, predicted: str, expected: str) -> bool:
        return _numeric_equal(predicted, expected, tolerance=0.0)

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "problem": row["problem"],
            "problem_idx": row.get("problem_idx"),
            "problem_type": row.get("problem_type"),
        }


class GraphwalksBenchmark(DatasetBenchmark):
    dataset_name = "openai/graphwalks"
    config_name = None
    split = "train"
    category = "long_context"
    max_tokens = 192

    def __init__(
        self,
        key: str,
        display_name: str,
        description: str,
        problem_type: str,
        min_prompt_chars: int,
        max_prompt_chars: int,
    ) -> None:
        self.key = key
        self.display_name = display_name
        self.description = description
        self.problem_type = problem_type
        self.min_prompt_chars = min_prompt_chars
        self.max_prompt_chars = max_prompt_chars

    def load_rows(self, limit: int) -> List[Dict[str, Any]]:
        dataset = _safe_load_dataset(self.dataset_name, self.config_name, self.split)
        return _take_filtered_samples(
            dataset,
            limit,
            lambda row: (
                str(row.get("problem_type", "")).lower() == self.problem_type
                and self.min_prompt_chars <= int(row.get("prompt_chars") or 0) <= self.max_prompt_chars
            ),
        )

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return row["prompt"]

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return json.dumps(_normalize_string_list(row["answer_nodes"]))

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return json.dumps(_normalize_string_list(_parse_string_list(response_text)))

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt_chars": row.get("prompt_chars"),
            "problem_type": row.get("problem_type"),
            "date_added": row.get("date_added"),
            "answer_nodes": row.get("answer_nodes"),
        }


class MRCRBenchmark(DatasetBenchmark):
    dataset_name = "openai/mrcr"
    config_name = None
    split = "train"
    category = "long_context"
    max_tokens = 2048

    def __init__(
        self,
        key: str,
        display_name: str,
        description: str,
        n_needles: int,
        min_chars: int,
        max_chars: int,
    ) -> None:
        self.key = key
        self.display_name = display_name
        self.description = description
        self.n_needles = n_needles
        self.min_chars = min_chars
        self.max_chars = max_chars

    def load_rows(self, limit: int) -> List[Dict[str, Any]]:
        dataset = _safe_load_dataset(self.dataset_name, self.config_name, self.split)
        return _take_filtered_samples(
            dataset,
            limit,
            lambda row: (
                int(row.get("n_needles") or 0) == self.n_needles
                and self.min_chars <= int(row.get("n_chars") or 0) < self.max_chars
            ),
        )

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return _render_chat_prompt(row["prompt"])

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return _normalize_multiline_text(row["answer"])

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _normalize_multiline_text(response_text)

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "n_needles": row.get("n_needles"),
            "desired_msg_index": row.get("desired_msg_index"),
            "total_messages": row.get("total_messages"),
            "n_chars": row.get("n_chars"),
            "date_added": row.get("date_added"),
            "random_string_to_prepend": row.get("random_string_to_prepend"),
        }


class ARCChallengeBenchmark(DatasetBenchmark):
    key = "arc_challenge"
    display_name = "ARC Challenge"
    dataset_name = "ai2_arc"
    config_name = "ARC-Challenge"
    split = "validation"
    description = "AI2 Reasoning Challenge science questions."
    category = "reasoning"
    max_tokens = 48

    def build_prompt(self, row: Dict[str, Any]) -> str:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        valid = ", ".join(labels)
        return (
            f"Answer the science multiple-choice question. Reply with only one option label: {valid}.\n\n"
            f"Question: {row['question']}\n"
            f"{choices}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return str(row["answerKey"]).upper()

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _normalize_choice_response(
            response_text,
            [str(label).upper() for label in row["choices"]["label"]],
        )

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": row["question"],
            "choices": row["choices"],
        }


class ARCEasyBenchmark(ARCChallengeBenchmark):
    key = "arc_easy"
    display_name = "ARC Easy"
    config_name = "ARC-Easy"
    description = "AI2 Reasoning Challenge easy split."


class HellaSwagBenchmark(DatasetBenchmark):
    key = "hellaswag"
    display_name = "HellaSwag"
    dataset_name = "hellaswag"
    config_name = None
    split = "validation"
    description = "Commonsense completion and next-step inference."
    category = "reasoning"
    max_tokens = 48

    def build_prompt(self, row: Dict[str, Any]) -> str:
        endings = "\n".join(f"{chr(65 + idx)}. {ending}" for idx, ending in enumerate(row["endings"]))
        return (
            "Choose the best continuation. Reply with only one letter: A, B, C, or D.\n\n"
            f"Context: {row['ctx']}\n"
            f"{endings}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return chr(65 + int(row["label"]))

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _normalize_choice_response(response_text, ["A", "B", "C", "D"])

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "context": row["ctx"],
            "activity_label": row.get("activity_label"),
            "endings": row["endings"],
        }


class WinograndeBenchmark(DatasetBenchmark):
    key = "winogrande"
    display_name = "Winogrande"
    dataset_name = "winogrande"
    config_name = "winogrande_xl"
    split = "validation"
    description = "Pronoun and commonsense reasoning with two choices."
    category = "reasoning"
    max_tokens = 24

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return (
            "Choose the best replacement for the blank. Reply with only A or B.\n\n"
            f"Sentence: {row['sentence']}\n"
            f"A. {row['option1']}\n"
            f"B. {row['option2']}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return "A" if str(row["answer"]) == "1" else "B"

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _normalize_choice_response(response_text, ["A", "B"])

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sentence": row["sentence"],
            "option1": row["option1"],
            "option2": row["option2"],
        }


class BoolQBenchmark(DatasetBenchmark):
    key = "boolq"
    display_name = "BoolQ"
    dataset_name = "boolq"
    config_name = None
    split = "validation"
    description = "Passage-grounded yes/no questions."
    category = "knowledge"
    max_tokens = 24

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return (
            "Read the passage and answer the question. Reply with only Yes or No.\n\n"
            f"Passage: {row['passage']}\n\n"
            f"Question: {row['question']}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return "yes" if bool(row["answer"]) else "no"

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        normalized = _normalize_boolean_response(response_text)
        return "yes" if normalized in {"yes", "true"} else "no" if normalized in {"no", "false"} else ""

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": row["question"],
            "passage": row["passage"],
        }


class CommonsenseQABenchmark(DatasetBenchmark):
    key = "commonsense_qa"
    display_name = "CommonsenseQA"
    dataset_name = "commonsense_qa"
    config_name = None
    split = "validation"
    description = "Commonsense reasoning multiple-choice questions."
    category = "reasoning"
    max_tokens = 32

    def build_prompt(self, row: Dict[str, Any]) -> str:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        return (
            "Answer the commonsense question. Reply with only the option letter.\n\n"
            f"Question: {row['question']}\n"
            f"{choices}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return str(row["answerKey"]).upper()

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _normalize_choice_response(
            response_text,
            [str(label).upper() for label in row["choices"]["label"]],
        )

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": row["question"],
            "question_concept": row.get("question_concept"),
            "choices": row["choices"],
        }


class PIQABenchmark(DatasetBenchmark):
    key = "piqa"
    display_name = "PIQA"
    dataset_name = "piqa"
    config_name = None
    split = "validation"
    description = "Physical commonsense reasoning with two candidate solutions."
    category = "reasoning"
    max_tokens = 24

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return (
            "Choose the more sensible physical-world solution. Reply with only A or B.\n\n"
            f"Goal: {row['goal']}\n"
            f"A. {row['sol1']}\n"
            f"B. {row['sol2']}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return "A" if int(row["label"]) == 0 else "B"

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _normalize_choice_response(response_text, ["A", "B"])

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "goal": row["goal"],
            "sol1": row["sol1"],
            "sol2": row["sol2"],
        }


class SocialIQABenchmark(DatasetBenchmark):
    key = "social_iqa"
    display_name = "Social IQa"
    dataset_name = "social_i_qa"
    config_name = None
    split = "validation"
    description = "Social commonsense reasoning with three answer choices."
    category = "reasoning"
    max_tokens = 32

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return (
            "Answer the social reasoning question. Reply with only A, B, or C.\n\n"
            f"Context: {row['context']}\n"
            f"Question: {row['question']}\n"
            f"A. {row['answerA']}\n"
            f"B. {row['answerB']}\n"
            f"C. {row['answerC']}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return {"1": "A", "2": "B", "3": "C"}[str(row["label"])]

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _normalize_choice_response(response_text, ["A", "B", "C"])

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "context": row["context"],
            "question": row["question"],
            "answer_a": row["answerA"],
            "answer_b": row["answerB"],
            "answer_c": row["answerC"],
        }


class OpenBookQABenchmark(DatasetBenchmark):
    key = "openbookqa"
    display_name = "OpenBookQA"
    dataset_name = "allenai/openbookqa"
    config_name = "main"
    split = "test"
    description = "Elementary science questions with four options."
    category = "knowledge"
    max_tokens = 32

    def build_prompt(self, row: Dict[str, Any]) -> str:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        return (
            "Answer the science question. Reply with only the option letter.\n\n"
            f"Question: {row['question_stem']}\n"
            f"{choices}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return str(row["answerKey"]).upper()

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        return _normalize_choice_response(
            response_text,
            [str(label).upper() for label in row["choices"]["label"]],
        )

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": row["question_stem"],
            "choices": row["choices"],
        }


class TruthfulQAMC1Benchmark(DatasetBenchmark):
    key = "truthfulqa_mc1"
    display_name = "TruthfulQA MC1"
    dataset_name = "truthful_qa"
    config_name = "multiple_choice"
    split = "validation"
    description = "Truthfulness multiple-choice benchmark using MC1 targets."
    category = "knowledge"
    max_tokens = 40

    def build_prompt(self, row: Dict[str, Any]) -> str:
        choices = row["mc1_targets"]["choices"]
        labels = [chr(65 + idx) for idx in range(len(choices))]
        rendered = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
        return (
            "Choose the most truthful answer. Reply with only the option letter.\n\n"
            f"Question: {row['question']}\n"
            f"{rendered}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        labels = row["mc1_targets"]["labels"]
        return chr(65 + labels.index(1))

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        choices = row["mc1_targets"]["choices"]
        labels = [chr(65 + idx) for idx in range(len(choices))]
        return _normalize_choice_response(response_text, labels)

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": row["question"],
            "choices": row["mc1_targets"]["choices"],
        }


class BBHBooleanExpressionsBenchmark(DatasetBenchmark):
    key = "bbh_boolean_expressions"
    display_name = "BBH Boolean Expressions"
    dataset_name = "lukaemon/bbh"
    config_name = "boolean_expressions"
    split = "test"
    description = "Big-Bench Hard boolean expressions task."
    category = "reasoning"
    max_tokens = 24

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return (
            "Evaluate the boolean expression. Reply with only True or False.\n\n"
            f"Expression: {row['input']}\n"
            "Answer:"
        )

    def extract_expected_answer(self, row: Dict[str, Any]) -> str:
        return str(row["target"]).strip().lower()

    def parse_prediction(self, response_text: str, row: Dict[str, Any]) -> str:
        normalized = _normalize_boolean_response(response_text)
        return "true" if normalized in {"true", "yes"} else "false" if normalized in {"false", "no"} else ""

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {"input": row["input"]}


SUPPORTED_BENCHMARKS: Dict[str, DatasetBenchmark] = {
    "gsm8k": GSM8KBenchmark(),
    "mmlu": MMLUBenchmark(),
    "mmlu_pro": MMLUProBenchmark(),
    "math": HendrycksMathBenchmark(),
    "aime_2024": AIME2024Benchmark(),
    "aime_2025": AIME2025Benchmark(),
    "arc_challenge": ARCChallengeBenchmark(),
    "arc_easy": ARCEasyBenchmark(),
    "hellaswag": HellaSwagBenchmark(),
    "winogrande": WinograndeBenchmark(),
    "boolq": BoolQBenchmark(),
    "commonsense_qa": CommonsenseQABenchmark(),
    "piqa": PIQABenchmark(),
    "social_iqa": SocialIQABenchmark(),
    "openbookqa": OpenBookQABenchmark(),
    "truthfulqa_mc1": TruthfulQAMC1Benchmark(),
    "bbh_boolean_expressions": BBHBooleanExpressionsBenchmark(),
    "graphwalks_bfs_0_128k": GraphwalksBenchmark(
        key="graphwalks_bfs_0_128k",
        display_name="Graphwalks BFS 0K-128K",
        description="OpenAI Graphwalks breadth-first search tasks with prompts up to 128K characters.",
        problem_type="bfs",
        min_prompt_chars=0,
        max_prompt_chars=128000,
    ),
    "graphwalks_bfs_256k_1m": GraphwalksBenchmark(
        key="graphwalks_bfs_256k_1m",
        display_name="Graphwalks BFS 256K-1M",
        description="OpenAI Graphwalks breadth-first search tasks with 256K-1M character prompts.",
        problem_type="bfs",
        min_prompt_chars=256000,
        max_prompt_chars=1000000,
    ),
    "graphwalks_parents_0_128k": GraphwalksBenchmark(
        key="graphwalks_parents_0_128k",
        display_name="Graphwalks Parents 0K-128K",
        description="OpenAI Graphwalks parent-retrieval tasks with prompts up to 128K characters.",
        problem_type="parents",
        min_prompt_chars=0,
        max_prompt_chars=128000,
    ),
    "graphwalks_parents_256k_1m": GraphwalksBenchmark(
        key="graphwalks_parents_256k_1m",
        display_name="Graphwalks Parents 256K-1M",
        description="OpenAI Graphwalks parent-retrieval tasks with 256K-1M character prompts.",
        problem_type="parents",
        min_prompt_chars=256000,
        max_prompt_chars=1000000,
    ),
    "mrcr_v2_8needle_4k_8k": MRCRBenchmark(
        key="mrcr_v2_8needle_4k_8k",
        display_name="OpenAI MRCR v2 8-needle 4K-8K",
        description="OpenAI MRCR v2 retrieval benchmark with 8 needles and 4K-8K contexts.",
        n_needles=8,
        min_chars=4096,
        max_chars=8192,
    ),
    "mrcr_v2_8needle_8k_16k": MRCRBenchmark(
        key="mrcr_v2_8needle_8k_16k",
        display_name="OpenAI MRCR v2 8-needle 8K-16K",
        description="OpenAI MRCR v2 retrieval benchmark with 8 needles and 8K-16K contexts.",
        n_needles=8,
        min_chars=8192,
        max_chars=16384,
    ),
    "mrcr_v2_8needle_16k_32k": MRCRBenchmark(
        key="mrcr_v2_8needle_16k_32k",
        display_name="OpenAI MRCR v2 8-needle 16K-32K",
        description="OpenAI MRCR v2 retrieval benchmark with 8 needles and 16K-32K contexts.",
        n_needles=8,
        min_chars=16384,
        max_chars=32768,
    ),
    "mrcr_v2_8needle_32k_64k": MRCRBenchmark(
        key="mrcr_v2_8needle_32k_64k",
        display_name="OpenAI MRCR v2 8-needle 32K-64K",
        description="OpenAI MRCR v2 retrieval benchmark with 8 needles and 32K-64K contexts.",
        n_needles=8,
        min_chars=32768,
        max_chars=65536,
    ),
    "mrcr_v2_8needle_64k_128k": MRCRBenchmark(
        key="mrcr_v2_8needle_64k_128k",
        display_name="OpenAI MRCR v2 8-needle 64K-128K",
        description="OpenAI MRCR v2 retrieval benchmark with 8 needles and 64K-128K contexts.",
        n_needles=8,
        min_chars=65536,
        max_chars=131072,
    ),
    "mrcr_v2_8needle_128k_256k": MRCRBenchmark(
        key="mrcr_v2_8needle_128k_256k",
        display_name="OpenAI MRCR v2 8-needle 128K-256K",
        description="OpenAI MRCR v2 retrieval benchmark with 8 needles and 128K-256K contexts.",
        n_needles=8,
        min_chars=131072,
        max_chars=262144,
    ),
    "mrcr_v2_8needle_256k_512k": MRCRBenchmark(
        key="mrcr_v2_8needle_256k_512k",
        display_name="OpenAI MRCR v2 8-needle 256K-512K",
        description="OpenAI MRCR v2 retrieval benchmark with 8 needles and 256K-512K contexts.",
        n_needles=8,
        min_chars=262144,
        max_chars=524288,
    ),
    "mrcr_v2_8needle_512k_1m": MRCRBenchmark(
        key="mrcr_v2_8needle_512k_1m",
        display_name="OpenAI MRCR v2 8-needle 512K-1M",
        description="OpenAI MRCR v2 retrieval benchmark with 8 needles and 512K-1M contexts.",
        n_needles=8,
        min_chars=524288,
        max_chars=1048577,
    ),
}

BENCHMARK_SUITES: Dict[str, Dict[str, Any]] = {
    "quick_suite": {
        "display_name": "Quick Suite",
        "description": "Fast sanity check with three widely used benchmarks.",
        "benchmarks": ["gsm8k", "mmlu", "hellaswag"],
    },
    "core_suite": {
        "display_name": "Core Suite",
        "description": "Balanced default suite for serious local benchmarking.",
        "benchmarks": ["gsm8k", "mmlu", "arc_challenge", "hellaswag", "winogrande", "boolq", "commonsense_qa", "piqa"],
    },
    "knowledge_suite": {
        "display_name": "Knowledge Suite",
        "description": "Knowledge-heavy evaluation with factual and multiple-choice tasks.",
        "benchmarks": ["mmlu", "mmlu_pro", "boolq", "truthfulqa_mc1", "openbookqa", "commonsense_qa"],
    },
    "commonsense_suite": {
        "display_name": "Commonsense Suite",
        "description": "Broad commonsense and physical/social reasoning coverage.",
        "benchmarks": ["hellaswag", "winogrande", "commonsense_qa", "openbookqa", "piqa", "social_iqa"],
    },
    "reasoning_suite": {
        "display_name": "Reasoning Suite",
        "description": "Math, reasoning, and BBH-style tasks.",
        "benchmarks": ["gsm8k", "math", "aime_2024", "aime_2025", "arc_challenge", "winogrande", "piqa", "bbh_boolean_expressions"],
    },
    "competition_math_suite": {
        "display_name": "Competition Math Suite",
        "description": "Math-heavy benchmarking with GSM8K, Hendrycks MATH, and AIME.",
        "benchmarks": ["gsm8k", "math", "aime_2024", "aime_2025"],
    },
    "long_context_suite": {
        "display_name": "Long Context Suite",
        "description": "Public long-context retrieval and graph reasoning benchmarks aligned with frontier reporting.",
        "benchmarks": [
            "graphwalks_bfs_0_128k",
            "graphwalks_bfs_256k_1m",
            "graphwalks_parents_0_128k",
            "graphwalks_parents_256k_1m",
            "mrcr_v2_8needle_4k_8k",
            "mrcr_v2_8needle_8k_16k",
            "mrcr_v2_8needle_16k_32k",
            "mrcr_v2_8needle_32k_64k",
            "mrcr_v2_8needle_64k_128k",
            "mrcr_v2_8needle_128k_256k",
            "mrcr_v2_8needle_256k_512k",
            "mrcr_v2_8needle_512k_1m",
        ],
    },
    "openai_public_suite": {
        "display_name": "OpenAI Public Suite",
        "description": "Runnable public benchmarks that map closely to recent OpenAI benchmark disclosures.",
        "benchmarks": [
            "aime_2024",
            "aime_2025",
            "graphwalks_bfs_0_128k",
            "graphwalks_parents_0_128k",
            "mrcr_v2_8needle_4k_8k",
            "mrcr_v2_8needle_8k_16k",
            "mrcr_v2_8needle_16k_32k",
        ],
    },
    "frontier_report_suite": {
        "display_name": "Frontier Report Suite",
        "description": "Local-only subset aligned with commonly reported public frontier benchmarks.",
        "benchmarks": [
            "gsm8k",
            "mmlu",
            "mmlu_pro",
            "math",
            "aime_2024",
            "aime_2025",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "boolq",
            "piqa",
            "graphwalks_bfs_0_128k",
            "graphwalks_parents_0_128k",
            "mrcr_v2_8needle_4k_8k",
        ],
    },
    "serious_suite": {
        "display_name": "Serious Suite",
        "description": "Broad local-only benchmark set for leaderboard-grade comparisons.",
        "benchmarks": [
            "gsm8k",
            "mmlu",
            "mmlu_pro",
            "math",
            "aime_2024",
            "aime_2025",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "boolq",
            "commonsense_qa",
            "piqa",
            "social_iqa",
            "openbookqa",
            "truthfulqa_mc1",
            "bbh_boolean_expressions",
            "graphwalks_bfs_0_128k",
            "graphwalks_parents_0_128k",
            "mrcr_v2_8needle_4k_8k",
        ],
    },
    "all_benchmarks": {
        "display_name": "All Benchmarks",
        "description": "Everything implemented in the local benchmark runner.",
        "benchmarks": list(SUPPORTED_BENCHMARKS.keys()),
    },
}

SUITE_ALIASES = {
    "quick": "quick_suite",
    "core": "core_suite",
    "knowledge": "knowledge_suite",
    "commonsense": "commonsense_suite",
    "reasoning": "reasoning_suite",
    "competition_math": "competition_math_suite",
    "long_context": "long_context_suite",
    "openai_public": "openai_public_suite",
    "frontier": "frontier_report_suite",
    "serious": "serious_suite",
    "all": "all_benchmarks",
}

DEFAULT_BENCHMARKS = list(BENCHMARK_SUITES["core_suite"]["benchmarks"])

SUPPORTED_BENCHMARK_LABS: Dict[str, Tuple[str, ...]] = {
    "aime_2024": ("openai", "google"),
    "aime_2025": ("openai", "google"),
    "graphwalks_bfs_0_128k": ("openai",),
    "graphwalks_bfs_256k_1m": ("openai",),
    "graphwalks_parents_0_128k": ("openai",),
    "graphwalks_parents_256k_1m": ("openai",),
    "mrcr_v2_8needle_4k_8k": ("openai",),
    "mrcr_v2_8needle_8k_16k": ("openai",),
    "mrcr_v2_8needle_16k_32k": ("openai",),
    "mrcr_v2_8needle_32k_64k": ("openai",),
    "mrcr_v2_8needle_64k_128k": ("openai",),
    "mrcr_v2_8needle_128k_256k": ("openai",),
    "mrcr_v2_8needle_256k_512k": ("openai",),
    "mrcr_v2_8needle_512k_1m": ("openai",),
}

SUPPORTED_BENCHMARK_NOTES: Dict[str, str] = {
    "aime_2024": "Public dataset adapter for the 2024 AIME competition problems.",
    "aime_2025": "Public dataset adapter for the 2025 AIME competition problems.",
    "graphwalks_bfs_0_128k": "Public OpenAI graph reasoning dataset sliced to the benchmarked prompt-length band.",
    "graphwalks_bfs_256k_1m": "Public OpenAI graph reasoning dataset sliced to the benchmarked prompt-length band.",
    "graphwalks_parents_0_128k": "Public OpenAI graph reasoning dataset sliced to the benchmarked prompt-length band.",
    "graphwalks_parents_256k_1m": "Public OpenAI graph reasoning dataset sliced to the benchmarked prompt-length band.",
    "mrcr_v2_8needle_4k_8k": "Public OpenAI multi-round coreference retrieval dataset sliced to the benchmarked context band.",
    "mrcr_v2_8needle_8k_16k": "Public OpenAI multi-round coreference retrieval dataset sliced to the benchmarked context band.",
    "mrcr_v2_8needle_16k_32k": "Public OpenAI multi-round coreference retrieval dataset sliced to the benchmarked context band.",
    "mrcr_v2_8needle_32k_64k": "Public OpenAI multi-round coreference retrieval dataset sliced to the benchmarked context band.",
    "mrcr_v2_8needle_64k_128k": "Public OpenAI multi-round coreference retrieval dataset sliced to the benchmarked context band.",
    "mrcr_v2_8needle_128k_256k": "Public OpenAI multi-round coreference retrieval dataset sliced to the benchmarked context band.",
    "mrcr_v2_8needle_256k_512k": "Public OpenAI multi-round coreference retrieval dataset sliced to the benchmarked context band.",
    "mrcr_v2_8needle_512k_1m": "Public OpenAI multi-round coreference retrieval dataset sliced to the benchmarked context band.",
}


SUPPORTED_BENCHMARK_CATALOG: Dict[str, BenchmarkCatalogEntry] = {
    key: BenchmarkCatalogEntry(
        key=benchmark.key,
        display_name=benchmark.display_name,
        description=benchmark.description,
        category=benchmark.category,
        status="runnable_local",
        harness="single_turn_local_prompt",
        local_runnable=True,
        dataset_name=benchmark.dataset_name,
        config_name=benchmark.config_name,
        split=benchmark.split,
        labs=SUPPORTED_BENCHMARK_LABS.get(key, ()),
        notes=SUPPORTED_BENCHMARK_NOTES.get(key, ""),
    )
    for key, benchmark in SUPPORTED_BENCHMARKS.items()
}

TRACKED_BENCHMARK_CATALOG: Dict[str, BenchmarkCatalogEntry] = {
    "gpqa_diamond": BenchmarkCatalogEntry(
        key="gpqa_diamond",
        display_name="GPQA Diamond",
        description="Graduate-level science QA benchmark used by major labs.",
        category="knowledge",
        status="planned_local_adapter",
        harness="multiple_choice_dataset_adapter",
        local_runnable=False,
        labs=("openai", "google", "meta"),
    ),
    "swe_bench_verified": BenchmarkCatalogEntry(
        key="swe_bench_verified",
        display_name="SWE-bench Verified",
        description="Real software engineering benchmark over issue-fix tasks.",
        category="coding",
        status="external_agent_harness",
        harness="repo_task_execution_harness",
        local_runnable=False,
        labs=("openai", "anthropic", "google"),
        notes="Requires repository setup, test execution, and agent-style patch evaluation.",
    ),
    "codeforces": BenchmarkCatalogEntry(
        key="codeforces",
        display_name="Codeforces",
        description="Competitive programming benchmark used in reasoning model reports.",
        category="coding",
        status="external_agent_harness",
        harness="competitive_programming_harness",
        local_runnable=False,
        labs=("openai",),
    ),
    "mmmu": BenchmarkCatalogEntry(
        key="mmmu",
        display_name="MMMU",
        description="Massive multimodal benchmark over charts, diagrams, and academic questions.",
        category="multimodal",
        status="external_multimodal_harness",
        harness="vision_language_eval_harness",
        local_runnable=False,
        labs=("openai", "meta"),
    ),
    "video_mme": BenchmarkCatalogEntry(
        key="video_mme",
        display_name="Video-MME",
        description="Video understanding benchmark cited for multimodal frontier models.",
        category="multimodal",
        status="external_multimodal_harness",
        harness="video_eval_harness",
        local_runnable=False,
        labs=("openai",),
    ),
    "healthbench": BenchmarkCatalogEntry(
        key="healthbench",
        display_name="HealthBench",
        description="Medical conversation benchmark introduced by OpenAI.",
        category="safety",
        status="restricted_or_specialized",
        harness="specialized_medical_grader",
        local_runnable=False,
        labs=("openai",),
    ),
    "healthbench_hard": BenchmarkCatalogEntry(
        key="healthbench_hard",
        display_name="HealthBench Hard",
        description="Hard subset of HealthBench for unsaturated medical evaluation.",
        category="safety",
        status="restricted_or_specialized",
        harness="specialized_medical_grader",
        local_runnable=False,
        labs=("openai",),
    ),
    "tau_bench": BenchmarkCatalogEntry(
        key="tau_bench",
        display_name="TauBench",
        description="Tool-use and policy-following benchmark over realistic service domains.",
        category="agentic",
        status="external_agent_harness",
        harness="tool_use_environment",
        local_runnable=False,
        labs=("anthropic",),
    ),
    "humanitys_last_exam": BenchmarkCatalogEntry(
        key="humanitys_last_exam",
        display_name="Humanity's Last Exam",
        description="Expert-level benchmark used to stress frontier reasoning models.",
        category="reasoning",
        status="planned_local_adapter",
        harness="expert_qa_dataset_adapter",
        local_runnable=False,
        labs=("google",),
    ),
    "multi_challenge": BenchmarkCatalogEntry(
        key="multi_challenge",
        display_name="MultiChallenge",
        description="Instruction-following benchmark reported in OpenAI GPT-4.1 launch materials.",
        category="instruction_following",
        status="planned_local_adapter",
        harness="instruction_following_harness",
        local_runnable=False,
        labs=("openai",),
    ),
}

BENCHMARK_CATALOG: Dict[str, BenchmarkCatalogEntry] = {
    **SUPPORTED_BENCHMARK_CATALOG,
    **TRACKED_BENCHMARK_CATALOG,
}


def get_supported_benchmark(name: str) -> DatasetBenchmark:
    """Return a supported benchmark or raise a clear error."""
    if name not in SUPPORTED_BENCHMARKS:
        if name in BENCHMARK_CATALOG:
            entry = BENCHMARK_CATALOG[name]
            raise ValueError(
                f"Benchmark '{name}' is tracked by smaLLMs but is not locally runnable yet "
                f"(status: {entry.status}, harness: {entry.harness})."
            )
        supported = ", ".join(sorted(SUPPORTED_BENCHMARKS))
        raise ValueError(f"Unsupported benchmark '{name}'. Supported benchmarks: {supported}")
    return SUPPORTED_BENCHMARKS[name]


def expand_benchmark_selection(selection: Optional[Sequence[str]]) -> List[str]:
    """Expand a mix of benchmark names and suite names into concrete benchmark keys."""
    if not selection:
        return list(DEFAULT_BENCHMARKS)

    expanded: List[str] = []
    for item in selection:
        key = SUITE_ALIASES.get(item, item)
        if key in BENCHMARK_SUITES:
            expanded.extend(BENCHMARK_SUITES[key]["benchmarks"])
        elif key in SUPPORTED_BENCHMARKS:
            expanded.append(key)
        elif key in TRACKED_BENCHMARK_CATALOG:
            entry = TRACKED_BENCHMARK_CATALOG[key]
            raise ValueError(
                f"Benchmark '{item}' is tracked but not locally runnable yet "
                f"(status: {entry.status}, harness: {entry.harness})."
            )
        else:
            supported = ", ".join(sorted(list(BENCHMARK_CATALOG) + list(BENCHMARK_SUITES)))
            raise ValueError(f"Unknown benchmark or suite '{item}'. Supported names: {supported}")

    deduped: List[str] = []
    for item in expanded:
        if item not in deduped:
            deduped.append(item)
    return deduped


def list_supported_benchmarks() -> List[Dict[str, Any]]:
    """Return benchmark metadata for the local pipeline."""
    return [
        SUPPORTED_BENCHMARK_CATALOG[key].to_dict()
        for key in SUPPORTED_BENCHMARKS
    ]


def list_benchmark_catalog(runnable_only: bool = False) -> List[Dict[str, Any]]:
    """Return the full benchmark catalog, including tracked frontier evals."""
    entries = [
        entry.to_dict()
        for entry in BENCHMARK_CATALOG.values()
        if not runnable_only or entry.local_runnable
    ]
    entries.sort(key=lambda item: (not item["local_runnable"], item["category"], item["key"]))
    return entries


def list_benchmark_suites() -> List[Dict[str, Any]]:
    """Return suite metadata for the local pipeline."""
    return [
        {
            "key": key,
            "display_name": payload["display_name"],
            "description": payload["description"],
            "benchmarks": payload["benchmarks"],
        }
        for key, payload in BENCHMARK_SUITES.items()
    ]
