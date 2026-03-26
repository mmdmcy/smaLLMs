"""
Reliable dataset-backed benchmarks for local model evaluation.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _safe_load_dataset(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
) -> Iterable[Dict[str, Any]]:
    """Load a dataset in streaming mode to keep local runs lightweight."""
    from datasets import load_dataset

    if config_name:
        return load_dataset(dataset_name, config_name, split=split, streaming=True)
    return load_dataset(dataset_name, split=split, streaming=True)


def _take_samples(dataset: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Take a bounded number of items from a streaming dataset."""
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if idx >= limit:
            break
        rows.append(row)
    return rows


def _extract_final_number(text: str) -> str:
    """Extract a final numeric answer from a response."""
    if not text:
        return ""

    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        text = boxed.group(1)

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


def _numeric_equal(left: str, right: str, tolerance: float = 1e-6) -> bool:
    """Compare numeric strings with a small tolerance."""
    try:
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)
    except (TypeError, ValueError):
        return left.strip() == right.strip()


def summarize_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-sample records into benchmark-level metrics."""
    sample_count = len(samples)
    correct_count = sum(1 for sample in samples if sample.get("is_correct"))
    total_errors = sum(1 for sample in samples if sample.get("error"))
    total_prompt_tokens = sum(int(sample.get("prompt_tokens") or 0) for sample in samples)
    total_completion_tokens = sum(int(sample.get("completion_tokens") or 0) for sample in samples)
    total_tokens = sum(int(sample.get("total_tokens") or 0) for sample in samples)
    latencies = [float(sample.get("latency_sec") or 0.0) for sample in samples]
    tps = [float(sample.get("tokens_per_second") or 0.0) for sample in samples if float(sample.get("tokens_per_second") or 0.0) > 0]

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
    stop_sequences: List[str] = None


class DatasetBenchmark(ABC):
    """Base class for local dataset-backed benchmarks."""

    key: str
    display_name: str
    dataset_name: str
    config_name: Optional[str]
    split: str
    description: str
    max_tokens: int = 128

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
    ) -> BenchmarkExecution:
        """Evaluate a model on this benchmark and return summary plus samples."""
        dataset = _safe_load_dataset(self.dataset_name, self.config_name, self.split)
        rows = _take_samples(dataset, num_samples)

        generation_config = LocalGenerationConfig(
            temperature=temperature,
            max_tokens=self.max_tokens,
            top_p=0.9,
            stop_sequences=[],
        )

        samples: List[Dict[str, Any]] = []

        for index, row in enumerate(rows):
            prompt = self.build_prompt(row)
            expected = self.extract_expected_answer(row)
            generation = await model.generate_with_metadata(prompt, generation_config)
            predicted = self.parse_prediction(generation.text, row)
            correct = self.is_correct(predicted, expected)

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
                "error": generation.raw.get("error") if generation.raw else None,
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

        return BenchmarkExecution(benchmark_result=benchmark_result, samples=samples)


class GSM8KBenchmark(DatasetBenchmark):
    key = "gsm8k"
    display_name = "GSM8K"
    dataset_name = "gsm8k"
    config_name = "main"
    split = "test"
    description = "Grade-school math word problems."
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


class ARCBenchmark(DatasetBenchmark):
    key = "arc"
    display_name = "ARC Challenge"
    dataset_name = "ai2_arc"
    config_name = "ARC-Challenge"
    split = "validation"
    description = "AI2 Reasoning Challenge science questions."
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
        return _normalize_choice_response(response_text, [str(label).upper() for label in row["choices"]["label"]])

    def sample_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": row["question"],
            "choices": row["choices"],
        }


class HellaSwagBenchmark(DatasetBenchmark):
    key = "hellaswag"
    display_name = "HellaSwag"
    dataset_name = "hellaswag"
    config_name = None
    split = "validation"
    description = "Commonsense completion and next-step inference."
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


SUPPORTED_BENCHMARKS: Dict[str, DatasetBenchmark] = {
    "gsm8k": GSM8KBenchmark(),
    "mmlu": MMLUBenchmark(),
    "arc": ARCBenchmark(),
    "hellaswag": HellaSwagBenchmark(),
}

DEFAULT_BENCHMARKS = ["gsm8k", "mmlu", "arc", "hellaswag"]


def get_supported_benchmark(name: str) -> DatasetBenchmark:
    """Return a supported benchmark or raise a clear error."""
    if name not in SUPPORTED_BENCHMARKS:
        supported = ", ".join(sorted(SUPPORTED_BENCHMARKS))
        raise ValueError(f"Unsupported benchmark '{name}'. Supported benchmarks: {supported}")
    return SUPPORTED_BENCHMARKS[name]


def list_supported_benchmarks() -> List[Dict[str, Any]]:
    """Return benchmark metadata for the local pipeline."""
    return [
        {
            "key": benchmark.key,
            "display_name": benchmark.display_name,
            "description": benchmark.description,
            "dataset_name": benchmark.dataset_name,
            "config_name": benchmark.config_name,
            "split": benchmark.split,
        }
        for benchmark in SUPPORTED_BENCHMARKS.values()
    ]
