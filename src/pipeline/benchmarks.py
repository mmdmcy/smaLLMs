"""
Reliable dataset-backed benchmarks for local model evaluation.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]

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
    "arc_challenge": ARCChallengeBenchmark(),
    "arc_easy": ARCEasyBenchmark(),
    "hellaswag": HellaSwagBenchmark(),
    "winogrande": WinograndeBenchmark(),
    "boolq": BoolQBenchmark(),
    "commonsense_qa": CommonsenseQABenchmark(),
    "openbookqa": OpenBookQABenchmark(),
    "truthfulqa_mc1": TruthfulQAMC1Benchmark(),
    "bbh_boolean_expressions": BBHBooleanExpressionsBenchmark(),
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
        "benchmarks": ["gsm8k", "mmlu", "arc_challenge", "hellaswag", "winogrande", "boolq", "commonsense_qa"],
    },
    "knowledge_suite": {
        "display_name": "Knowledge Suite",
        "description": "Knowledge-heavy evaluation with factual and multiple-choice tasks.",
        "benchmarks": ["mmlu", "mmlu_pro", "boolq", "truthfulqa_mc1", "openbookqa", "commonsense_qa"],
    },
    "reasoning_suite": {
        "display_name": "Reasoning Suite",
        "description": "Math, reasoning, and BBH-style tasks.",
        "benchmarks": ["gsm8k", "math", "arc_challenge", "winogrande", "bbh_boolean_expressions"],
    },
    "serious_suite": {
        "display_name": "Serious Suite",
        "description": "Broad local-only benchmark set for leaderboard-grade comparisons.",
        "benchmarks": [
            "gsm8k",
            "mmlu",
            "mmlu_pro",
            "math",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "boolq",
            "commonsense_qa",
            "openbookqa",
            "truthfulqa_mc1",
            "bbh_boolean_expressions",
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
    "reasoning": "reasoning_suite",
    "serious": "serious_suite",
    "all": "all_benchmarks",
}

DEFAULT_BENCHMARKS = list(BENCHMARK_SUITES["core_suite"]["benchmarks"])


def get_supported_benchmark(name: str) -> DatasetBenchmark:
    """Return a supported benchmark or raise a clear error."""
    if name not in SUPPORTED_BENCHMARKS:
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
        else:
            supported = ", ".join(sorted(list(SUPPORTED_BENCHMARKS) + list(BENCHMARK_SUITES)))
            raise ValueError(f"Unknown benchmark or suite '{item}'. Supported names: {supported}")

    deduped: List[str] = []
    for item in expanded:
        if item not in deduped:
            deduped.append(item)
    return deduped


def list_supported_benchmarks() -> List[Dict[str, Any]]:
    """Return benchmark metadata for the local pipeline."""
    return [
        {
            "key": benchmark.key,
            "display_name": benchmark.display_name,
            "description": benchmark.description,
            "category": benchmark.category,
            "dataset_name": benchmark.dataset_name,
            "config_name": benchmark.config_name,
            "split": benchmark.split,
        }
        for benchmark in SUPPORTED_BENCHMARKS.values()
    ]


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
