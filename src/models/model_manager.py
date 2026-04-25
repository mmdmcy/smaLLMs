"""
Model management layer for smaLLMs platform.
Provides unified interface for different model providers and endpoints.
"""

import asyncio
import time
import base64
import json
import glob
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timezone

try:
    import aiohttp
except ModuleNotFoundError:
    aiohttp = None

try:
    from huggingface_hub import InferenceClient
except ModuleNotFoundError:
    InferenceClient = None

@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    provider: str
    size_gb: float
    parameters: str
    architecture: str
    license: str
    cost_per_token: float = 0.0
    max_context: int = 2048
    supports_streaming: bool = True
    supports_vision: bool = False
    model_type: str = "text"  # text, vision, code
    local_path: Optional[str] = None
    family: str = "unknown"
    quantization: str = "unknown"

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = None


@dataclass
class GenerationResult:
    """Normalized generation output with provider metadata."""

    text: str
    prompt: str
    started_at: str
    ended_at: str
    latency_sec: float
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    load_duration_sec: float = 0.0
    prompt_eval_duration_sec: float = 0.0
    eval_duration_sec: float = 0.0
    total_duration_sec: float = 0.0
    tokens_per_second: float = 0.0
    eval_tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    used_raw_fallback: bool = False
    raw_fallback_attempted: bool = False
    raw: Optional[Dict[str, Any]] = None


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _is_wsl() -> bool:
    """Return True when running inside WSL."""
    release = platform.release().lower()
    return bool(os.getenv("WSL_DISTRO_NAME")) or "microsoft" in release or "wsl" in release


def _find_windows_curl() -> Optional[str]:
    """Locate Windows curl.exe when running from WSL."""
    if not _is_wsl():
        return None

    for candidate in [
        "/mnt/c/Windows/System32/curl.exe",
        "/mnt/c/Windows/Sysnative/curl.exe",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


def _find_ollama_command(config: Dict[str, Any]) -> Optional[List[str]]:
    """Locate an Ollama CLI command for local fallback use."""
    configured = config.get("ollama", {}).get("command")
    if configured:
        if isinstance(configured, str):
            return [configured]
        if isinstance(configured, list):
            return configured

    local = shutil.which("ollama")
    if local:
        return [local]

    for pattern in [
        "/mnt/c/Users/*/AppData/Local/Programs/Ollama/ollama.exe",
        "/mnt/c/Program Files/Ollama/ollama.exe",
    ]:
        matches = sorted(glob.glob(pattern))
        if matches:
            return [matches[0]]

    return None


def _strip_ansi_sequences(text: str) -> str:
    """Strip ANSI control codes from CLI output."""
    return ANSI_ESCAPE_RE.sub("", text).replace("\r", "")


def _parse_size_to_gb(size_str: str) -> float:
    """Parse size strings like '2.1 GB' to a float in GB."""
    cleaned = size_str.strip().upper().replace(" ", "")
    try:
        if cleaned.endswith("GB"):
            return float(cleaned[:-2])
        if cleaned.endswith("MB"):
            return float(cleaned[:-2]) / 1024
        if cleaned.endswith("KB"):
            return float(cleaned[:-2]) / (1024 * 1024)
        if cleaned.endswith("B"):
            return float(cleaned[:-1]) / (1024 ** 3)
    except ValueError:
        return 0.0
    return 0.0


def _detect_vision_support_from_name(model_name: str) -> bool:
    """Best-effort vision detection from the model name."""
    vision_keywords = ["llava", "vision", "vl", "multimodal", "bakllava", "moondream", "minicpm-v"]
    return any(keyword in model_name.lower() for keyword in vision_keywords)


def _parse_ollama_list_output(stdout: str) -> List[Dict[str, Any]]:
    """Parse `ollama list` output into the repo's model metadata shape."""
    models: List[Dict[str, Any]] = []
    lines = [line.rstrip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return models

    for line in lines[1:]:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        name = parts[0]
        size_token = next((part for part in parts if part.upper().endswith(("GB", "MB", "KB", "B"))), "0B")
        supports_vision = _detect_vision_support_from_name(name)
        models.append(
            {
                "name": name,
                "provider": "ollama",
                "size_gb": _parse_size_to_gb(size_token),
                "supports_vision": supports_vision,
                "model_type": "vision" if supports_vision else "text",
                "cost_per_token": 0.0,
                "available": True,
                "family": "unknown",
                "quantization": "unknown",
                "parameters": "unknown",
            }
        )

    return models


def _parse_duration_to_sec(value: str) -> float:
    """Parse Ollama CLI duration strings to seconds."""
    raw = value.strip().lower()
    match = re.match(r"([0-9.]+)\s*(ns|us|µs|ms|s|m)$", raw)
    if not match:
        return 0.0

    amount = float(match.group(1))
    unit = match.group(2)
    if unit == "ns":
        return amount / 1_000_000_000
    if unit in {"us", "µs"}:
        return amount / 1_000_000
    if unit == "ms":
        return amount / 1_000
    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60
    return 0.0


def _parse_ollama_verbose_output(stdout: str) -> Dict[str, Any]:
    """Parse `ollama run --verbose` output into response text plus metrics."""
    cleaned = _strip_ansi_sequences(stdout)
    metrics_start = cleaned.find("total duration:")
    response_text = cleaned.strip()
    metrics_text = ""

    if metrics_start != -1:
        response_text = cleaned[:metrics_start].strip()
        metrics_text = cleaned[metrics_start:]

    metrics: Dict[str, Any] = {"response": response_text}
    patterns = {
        "total_duration": r"total duration:\s*([0-9.a-zA-Zµ]+)",
        "load_duration": r"load duration:\s*([0-9.a-zA-Zµ]+)",
        "prompt_eval_count": r"prompt eval count:\s*(\d+)",
        "prompt_eval_duration": r"prompt eval duration:\s*([0-9.a-zA-Zµ]+)",
        "eval_count": r"eval count:\s*(\d+)",
        "eval_duration": r"eval duration:\s*([0-9.a-zA-Zµ]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, metrics_text, re.IGNORECASE)
        if not match:
            continue

        value = match.group(1)
        if key.endswith("_duration"):
            metrics[key] = int(_parse_duration_to_sec(value) * 1_000_000_000)
        else:
            metrics[key] = int(value)

    return metrics

class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(self, model_name: str, config: Dict, metadata: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config
        self.metadata = metadata or {}
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
    
    @abstractmethod
    async def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def batch_generate(self, prompts: List[str], generation_config: GenerationConfig) -> List[str]:
        """Generate text for multiple prompts."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the model."""
        pass

    async def generate_with_metadata(
        self,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        """Generate text plus normalized metadata."""
        started = datetime.now(timezone.utc).isoformat()
        start_time = time.time()
        text = await self.generate(prompt, generation_config)
        latency_sec = time.time() - start_time
        ended = datetime.now(timezone.utc).isoformat()
        model_info = self.get_model_info()
        return GenerationResult(
            text=text,
            prompt=prompt,
            started_at=started,
            ended_at=ended,
            latency_sec=latency_sec,
            total_duration_sec=latency_sec,
            provider=model_info.provider,
        )

    async def batch_generate_with_metadata(
        self,
        prompts: List[str],
        generation_config: GenerationConfig,
    ) -> List[GenerationResult]:
        """Sequential metadata-aware generation fallback."""
        results = []
        for prompt in prompts:
            results.append(await self.generate_with_metadata(prompt, generation_config))
        return results

class HuggingFaceModel(BaseModel):
    """Hugging Face Inference Providers model implementation."""
    
    def __init__(self, model_name: str, config: Dict, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config, metadata=metadata)
        if InferenceClient is None:
            raise RuntimeError(
                "huggingface_hub is not installed. Install the full requirements or avoid Hugging Face models."
            )
        hf_config = config.get('huggingface', {})
        self.token = hf_config.get('token')
        
        # Use the modern InferenceClient for Inference Providers
        self.client = InferenceClient(
            api_key=self.token,
            provider="auto"  # Let HF choose the best provider automatically
        )
        self.max_retries = hf_config.get('inference_endpoints', {}).get('max_retries', 3)

    async def generate_with_metadata(
        self,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        """Generate text using HF chat completions with usage metadata when available."""
        started = datetime.now(timezone.utc).isoformat()
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=generation_config.max_tokens,
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                    stop=generation_config.stop_sequences or [],
                )

                if completion.choices and len(completion.choices) > 0:
                    content = completion.choices[0].message.content or ""
                    ended = datetime.now(timezone.utc).isoformat()
                    latency_sec = time.time() - start_time
                    usage = getattr(completion, "usage", None)
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else 0

                    return GenerationResult(
                        text=content.strip(),
                        prompt=prompt,
                        started_at=started,
                        ended_at=ended,
                        latency_sec=latency_sec,
                        total_duration_sec=latency_sec,
                        provider="huggingface",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        tokens_per_second=(completion_tokens / latency_sec) if latency_sec > 0 else 0.0,
                        raw=completion.model_dump() if hasattr(completion, "model_dump") else None,
                    )

                self.logger.warning(f"Model {self.model_name} returned no choices in attempt {attempt + 1}")
                return GenerationResult(
                    text="",
                    prompt=prompt,
                    started_at=started,
                    ended_at=datetime.now(timezone.utc).isoformat(),
                    latency_sec=time.time() - start_time,
                    total_duration_sec=time.time() - start_time,
                    provider="huggingface",
                )

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                if error_type == "StopIteration":
                    self.logger.error(
                        f"Model {self.model_name} doesn't support chat completions API (StopIteration). "
                        "Use instruction-tuned models only."
                    )
                    break

                self.logger.warning(
                    f"Chat API attempt {attempt + 1} failed for {self.model_name}: "
                    f"{error_type}: {error_msg}"
                )
                if attempt == self.max_retries - 1:
                    break
                await asyncio.sleep(2 ** attempt)

        ended = datetime.now(timezone.utc).isoformat()
        latency_sec = time.time() - start_time
        return GenerationResult(
            text="",
            prompt=prompt,
            started_at=started,
            ended_at=ended,
            latency_sec=latency_sec,
            total_duration_sec=latency_sec,
            provider="huggingface",
            raw={"error": f"Failed after {self.max_retries} attempts"},
        )
    
    async def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        """Generate text using HF Inference Providers Chat Completions API only."""
        result = await self.generate_with_metadata(prompt, generation_config)
        return result.text
    
    async def batch_generate(self, prompts: List[str], generation_config: GenerationConfig) -> List[str]:
        """Generate text for multiple prompts with rate limiting."""
        batch_results = await self.batch_generate_with_metadata(prompts, generation_config)
        return [result.text for result in batch_results]

    async def batch_generate_with_metadata(
        self,
        prompts: List[str],
        generation_config: GenerationConfig,
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts with rate limiting."""
        batch_size = self.config.get('huggingface', {}).get('inference_endpoints', {}).get('batch_size', 10)
        
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_tasks = [self.generate_with_metadata(prompt, generation_config) for prompt in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in batch
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch generation error: {str(result)}")
                    results.append(
                        GenerationResult(
                            text="",
                            prompt="",
                            started_at=datetime.now(timezone.utc).isoformat(),
                            ended_at=datetime.now(timezone.utc).isoformat(),
                            latency_sec=0.0,
                            total_duration_sec=0.0,
                            provider="huggingface",
                            raw={"error": str(result)},
                        )
                    )
                else:
                    results.append(result)
            
            # Small delay between batches to respect rate limits
            if i + batch_size < len(prompts):
                await asyncio.sleep(1)
        
        return results
    
    def get_model_info(self) -> ModelInfo:
        """Get model information from HF model card."""
        # This would typically fetch from HF API, simplified for now
        model_sizes = {
            "1b": 1.0, "2b": 2.0, "3b": 3.0, "7b": 7.0, "8b": 8.0,
            "13b": 13.0, "14b": 14.0, "15b": 15.0, "20b": 20.0
        }
        
        # Extract size from model name (simplified)
        size = 1.0
        for size_str, size_val in model_sizes.items():
            if size_str in self.model_name.lower():
                size = size_val
                break
        
        return ModelInfo(
            name=self.model_name,
            provider="huggingface",
            size_gb=size * 0.5,  # Rough estimate: 0.5GB per billion parameters
            parameters=f"{size}B",
            architecture="transformer",
            license="unknown",
            cost_per_token=0.0001 if size < 7 else 0.0002,  # Rough HF pricing
            max_context=2048,
            family=self.metadata.get("family", "unknown"),
            quantization=self.metadata.get("quantization", "unknown"),
        )

class OllamaModel(BaseModel):
    """Local Ollama model implementation with vision support."""
    
    def __init__(self, model_name: str, config: Dict, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config, metadata=metadata)
        self.base_url = config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        self.session = None
        self.transport: Optional[str] = None
        self.ollama_command = _find_ollama_command(config)
        self.windows_curl = _find_windows_curl()
        self.supports_vision = self._detect_vision_support()
    
    def _detect_vision_support(self) -> bool:
        """Detect if model supports vision based on name."""
        return _detect_vision_support_from_name(self.model_name)
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is not installed. Install local requirements for direct HTTP model access.")
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def unload_model(self):
        """Unload the model from Ollama memory to free up space."""
        try:
            payload = {"model": self.model_name, "keep_alive": 0}
            transport = await self._detect_transport()

            if transport == "http":
                session = await self._get_session()
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    response.raise_for_status()
            elif transport == "windows_curl" and self.windows_curl:
                await asyncio.to_thread(
                    subprocess.run,
                    [
                        self.windows_curl,
                        "-s",
                        "-X",
                        "POST",
                        f"{self.base_url}/api/generate",
                        "-H",
                        "Content-Type: application/json",
                        "--data-binary",
                        json.dumps(payload),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            elif transport == "cli" and self.ollama_command:
                await asyncio.to_thread(
                    subprocess.run,
                    [*self.ollama_command, "stop", self.model_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )

            self.logger.info(f"Unloaded model from memory: {self.model_name}")
        except Exception as e:
            self.logger.warning(f"Could not unload model {self.model_name}: {e}")
    
    async def close(self):
        """Close the aiohttp session and unload model."""
        # First unload the model from Ollama memory
        await self.unload_model()
        
        # Then close the session
        if self.session:
            await self.session.close()
            self.session = None
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for vision models."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {e}")
            return ""

    def _ollama_think_value(self) -> Optional[Union[bool, str]]:
        """Return the Ollama thinking setting for normal benchmark requests."""
        ollama_config = self.config.get("ollama", {})
        if "think" in ollama_config:
            return ollama_config["think"]
        if not ollama_config.get("disable_thinking", True):
            return None

        lowered = self.model_name.lower()
        if "gpt-oss" in lowered or "gptoss" in lowered:
            return ollama_config.get("gpt_oss_think_level", "low")
        return False

    def _effective_max_tokens(
        self,
        generation_config: GenerationConfig,
        think_value: Optional[Union[bool, str]],
    ) -> int:
        """Give forced-thinking models enough room to reach the answer channel."""
        requested_tokens = generation_config.max_tokens
        if isinstance(think_value, str):
            minimum = int(self.config.get("ollama", {}).get("thinking_min_tokens", 128))
            return max(requested_tokens, minimum)
        return requested_tokens

    def _generation_options(
        self,
        generation_config: GenerationConfig,
        think_value: Optional[Union[bool, str]] = None,
    ) -> Dict[str, Any]:
        """Build common Ollama generation options."""
        return {
            "temperature": generation_config.temperature,
            "num_predict": self._effective_max_tokens(generation_config, think_value),
            "top_p": generation_config.top_p,
            "stop": generation_config.stop_sequences or [],
        }
    
    def _duration_ns_to_sec(self, value: Any) -> float:
        """Convert Ollama nanosecond duration values to seconds."""
        try:
            if value is None:
                return 0.0
            return float(value) / 1_000_000_000
        except (TypeError, ValueError):
            return 0.0

    def _metric_int(self, value: Any) -> int:
        """Convert provider counters and durations to integers safely."""
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _normalized_total_duration_ns(self, result: Dict[str, Any]) -> int:
        """Return a total duration, falling back to the sum of known phases."""
        reported_total = self._metric_int(result.get("total_duration"))
        if reported_total > 0:
            return reported_total

        return (
            self._metric_int(result.get("load_duration"))
            + self._metric_int(result.get("prompt_eval_duration"))
            + self._metric_int(result.get("eval_duration"))
        )

    def _snapshot_attempt_metrics(self, result: Dict[str, Any], transport: str, raw_mode: bool) -> Dict[str, Any]:
        """Capture a compact per-attempt metric snapshot for debugging and aggregation."""
        return {
            "transport": transport,
            "raw_mode": raw_mode,
            "error": result.get("error"),
            "response_chars": len(str(result.get("response") or "")),
            "thinking_chars": len(str(result.get("thinking") or "")),
            "prompt_eval_count": self._metric_int(result.get("prompt_eval_count")),
            "eval_count": self._metric_int(result.get("eval_count")),
            "load_duration": self._metric_int(result.get("load_duration")),
            "prompt_eval_duration": self._metric_int(result.get("prompt_eval_duration")),
            "eval_duration": self._metric_int(result.get("eval_duration")),
            "total_duration": self._normalized_total_duration_ns(result),
        }

    def _aggregate_attempt_metrics(self, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-attempt Ollama metrics into total work done for one sample."""
        if not attempts:
            return {}

        return {
            "attempt_count": len(attempts),
            "prompt_eval_count": sum(self._metric_int(attempt.get("prompt_eval_count")) for attempt in attempts),
            "eval_count": sum(self._metric_int(attempt.get("eval_count")) for attempt in attempts),
            "load_duration": sum(self._metric_int(attempt.get("load_duration")) for attempt in attempts),
            "prompt_eval_duration": sum(self._metric_int(attempt.get("prompt_eval_duration")) for attempt in attempts),
            "eval_duration": sum(self._metric_int(attempt.get("eval_duration")) for attempt in attempts),
            "total_duration": sum(self._metric_int(attempt.get("total_duration")) for attempt in attempts),
        }

    async def _detect_transport(self) -> str:
        """Detect the best available transport for talking to Ollama."""
        if self.transport:
            return self.transport

        if aiohttp is not None:
            try:
                session = await self._get_session()
                async with session.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        self.transport = "http"
                        return self.transport
            except Exception:
                pass

        if self.windows_curl:
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [self.windows_curl, "-s", f"{self.base_url}/api/tags"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip().startswith("{"):
                    self.transport = "windows_curl"
                    return self.transport
            except Exception:
                pass

        if self.ollama_command:
            self.transport = "cli"
            return self.transport

        raise RuntimeError("Could not connect to Ollama. Start Ollama or configure an accessible local endpoint.")

    def _build_payload(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        image_path: Optional[str] = None,
        raw_mode: bool = False,
    ) -> Dict[str, Any]:
        """Build a standard Ollama generate payload."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": self._generation_options(generation_config),
            "stream": False,
            "keep_alive": "10m",
        }

        if raw_mode:
            payload["think"] = False
            payload["raw"] = True

        if image_path and self.supports_vision:
            base64_image = self._encode_image(image_path)
            if base64_image:
                payload["images"] = [base64_image]

        return payload

    def _build_chat_payload(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build an Ollama chat payload for normal benchmark generation."""
        think_value = self._ollama_think_value()
        message: Dict[str, Any] = {"role": "user", "content": prompt}
        if image_path and self.supports_vision:
            base64_image = self._encode_image(image_path)
            if base64_image:
                message["images"] = [base64_image]

        payload = {
            "model": self.model_name,
            "messages": [message],
            "options": self._generation_options(generation_config, think_value),
            "stream": False,
            "keep_alive": "10m",
        }
        if think_value is not None:
            payload["think"] = think_value

        return payload

    def _normalize_chat_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Expose chat responses through the same keys used by generate responses."""
        if result.get("error"):
            return result

        message = result.get("message") or {}
        normalized = dict(result)
        normalized["response"] = str(message.get("content") or result.get("response") or "")
        thinking = message.get("thinking") or result.get("thinking")
        if thinking:
            normalized["thinking"] = str(thinking)
        return normalized

    def _timeout_config(self) -> Dict[str, int]:
        """Return timeout and retry settings adapted to model size."""
        base_timeout = self.config.get('ollama', {}).get('timeout', 600)
        max_retries = self.config.get('ollama', {}).get('max_retries', 5)

        model_size_factor = 1.5
        lowered = self.model_name.lower()
        if '3b' in lowered or '7b' in lowered:
            model_size_factor = 3.0
        elif '13b' in lowered or 'large' in lowered or '20b' in lowered:
            model_size_factor = 5.0
        elif '1b' in lowered or '1.5b' in lowered or '0.8b' in lowered:
            model_size_factor = 2.0

        return {
            "base_timeout": int(base_timeout),
            "max_retries": int(max_retries),
            "model_size_factor": model_size_factor,
        }

    async def _request_http(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        current_timeout: int,
    ) -> Dict[str, Any]:
        """Call an Ollama HTTP endpoint through the local Python process."""
        if aiohttp is None:
            return {"response": "", "error": "aiohttp_not_installed"}
        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(total=current_timeout)
        async with session.post(f"{self.base_url}{endpoint}", json=payload, timeout=timeout) as response:
            if response.status == 404:
                self.logger.error(f"Model {self.model_name} not found in Ollama. Is it pulled?")
                return {"response": "", "error": "model_not_found"}
            if response.status == 500:
                error_text = await response.text()
                self.logger.error(f"Ollama server error: {error_text}")
                return {"response": "", "error": error_text}

            response.raise_for_status()
            return await response.json()

    async def _request_generate_http(self, payload: Dict[str, Any], current_timeout: int) -> Dict[str, Any]:
        """Call the Ollama generate API through the local Python process."""
        return await self._request_http("/api/generate", payload, current_timeout)

    async def _request_chat_http(self, payload: Dict[str, Any], current_timeout: int) -> Dict[str, Any]:
        """Call the Ollama chat API through the local Python process."""
        return self._normalize_chat_result(await self._request_http("/api/chat", payload, current_timeout))

    async def _request_windows_curl(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        current_timeout: int,
    ) -> Dict[str, Any]:
        """Call an Ollama HTTP endpoint through Windows curl.exe when running in WSL."""
        if not self.windows_curl:
            return {"response": "", "error": "windows_curl_not_found"}

        result = await asyncio.to_thread(
            subprocess.run,
            [
                self.windows_curl,
                "-s",
                "-X",
                "POST",
                f"{self.base_url}{endpoint}",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                json.dumps(payload),
            ],
            capture_output=True,
            text=True,
            timeout=current_timeout,
            check=False,
        )
        if result.returncode != 0:
            return {"response": "", "error": result.stderr.strip() or "windows_curl_failed"}

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"response": "", "error": "invalid_json_from_windows_curl", "raw_stdout": result.stdout}

    async def _request_generate_windows_curl(self, payload: Dict[str, Any], current_timeout: int) -> Dict[str, Any]:
        """Call the Ollama generate API through Windows curl.exe when running in WSL."""
        return await self._request_windows_curl("/api/generate", payload, current_timeout)

    async def _request_chat_windows_curl(self, payload: Dict[str, Any], current_timeout: int) -> Dict[str, Any]:
        """Call the Ollama chat API through Windows curl.exe when running in WSL."""
        return self._normalize_chat_result(
            await self._request_windows_curl("/api/chat", payload, current_timeout)
        )

    async def _request_generate_cli(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        current_timeout: int,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the Ollama CLI as the last local fallback."""
        if not self.ollama_command:
            return {"response": "", "error": "ollama_cli_not_found"}
        if image_path:
            return {"response": "", "error": "ollama_cli_image_generation_not_supported"}

        command = [
            *self.ollama_command,
            "run",
            self.model_name,
            "--verbose",
            "--nowordwrap",
            prompt,
        ]
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            capture_output=True,
            text=True,
            timeout=current_timeout,
            check=False,
        )
        if result.returncode != 0:
            return {"response": "", "error": result.stderr.strip() or "ollama_cli_failed"}

        parsed = _parse_ollama_verbose_output(result.stdout)
        parsed["transport"] = "cli"
        return parsed

    async def _request_generate(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a raw Ollama response payload."""
        timeout_config = self._timeout_config()
        base_timeout = timeout_config["base_timeout"]
        max_retries = timeout_config["max_retries"]
        model_size_factor = timeout_config["model_size_factor"]

        for attempt in range(max_retries + 1):
            try:
                current_timeout = int(base_timeout * model_size_factor * (1 + attempt * 1.0))
                transport = await self._detect_transport()
                self.logger.debug(
                    f"Ollama request attempt {attempt + 1}/{max_retries + 1} with {current_timeout}s "
                    f"timeout for {self.model_name} via {transport}"
                )

                attempt_metrics: List[Dict[str, Any]] = []
                if transport == "http":
                    payload = self._build_chat_payload(prompt, generation_config, image_path=image_path)
                    result = await self._request_chat_http(payload, current_timeout)
                elif transport == "windows_curl":
                    payload = self._build_chat_payload(prompt, generation_config, image_path=image_path)
                    result = await self._request_chat_windows_curl(payload, current_timeout)
                else:
                    result = await self._request_generate_cli(
                        prompt,
                        generation_config,
                        current_timeout,
                        image_path=image_path,
                    )
                attempt_metrics.append(self._snapshot_attempt_metrics(result, transport=transport, raw_mode=False))

                response_text = (result.get("response") or result.get("thinking") or "").strip()
                should_retry_raw = (
                    transport in {"http", "windows_curl"}
                    and not result.get("error")
                    and not str(result.get("response") or "").strip()
                    and (
                        str(result.get("thinking") or "").strip()
                        or int(result.get("eval_count") or 0) > 0
                    )
                )
                if should_retry_raw:
                    self.logger.info(
                        "Retrying %s with Ollama raw mode because the normal response channel was empty.",
                        self.model_name,
                    )
                    raw_payload = self._build_payload(
                        prompt,
                        generation_config,
                        image_path=image_path,
                        raw_mode=True,
                    )
                    if transport == "http":
                        raw_result = await self._request_generate_http(raw_payload, current_timeout)
                    else:
                        raw_result = await self._request_generate_windows_curl(raw_payload, current_timeout)
                    attempt_metrics.append(self._snapshot_attempt_metrics(raw_result, transport=transport, raw_mode=True))

                    raw_response_text = (raw_result.get("response") or raw_result.get("thinking") or "").strip()
                    if raw_response_text:
                        raw_result["raw_fallback_attempted"] = True
                        raw_result["used_raw_fallback"] = True
                        result = raw_result
                        response_text = raw_response_text
                    else:
                        result["raw_fallback_attempted"] = True

                aggregate_metrics = self._aggregate_attempt_metrics(attempt_metrics)
                if aggregate_metrics:
                    result.update(aggregate_metrics)
                    result["attempts"] = attempt_metrics
                    result["transport"] = transport
                    result["raw_fallback_attempted"] = bool(result.get("raw_fallback_attempted"))
                    result["used_raw_fallback"] = bool(result.get("used_raw_fallback"))

                if result.get("error"):
                    self.logger.warning(f"Ollama generation returned an error for {self.model_name}: {result['error']}")
                elif not response_text:
                    self.logger.warning(f"Empty response from Ollama for {self.model_name}: {result}")
                else:
                    self.logger.info(
                        f"Ollama response for {self.model_name}: response_length={len(response_text)} via {transport}"
                    )
                return result

            except asyncio.TimeoutError:
                if attempt < max_retries:
                    wait_time = 10 * (attempt + 1)
                    self.logger.warning(f"Ollama generation timeout for model {self.model_name} (timeout: {current_timeout}s), retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Ollama generation timeout for model {self.model_name} after {max_retries + 1} attempts with final timeout of {current_timeout}s")
                    return {"response": "", "error": "timeout"}
            except Exception as e:
                if aiohttp is not None and isinstance(e, aiohttp.ClientError):
                    self.logger.error(f"Ollama connection error: {str(e)}")
                    self.transport = None
                    return {"response": "", "error": str(e)}
                self.logger.error(f"Ollama generation error: {str(e)}")
                self.transport = None
                return {"response": "", "error": str(e)}
        
        return {"response": "", "error": "unknown"}

    async def generate_with_metadata(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        image_path: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text using local Ollama instance with normalized metadata."""
        started = datetime.now(timezone.utc).isoformat()
        start_time = time.time()
        raw = await self._request_generate(prompt, generation_config, image_path=image_path)
        latency_sec = time.time() - start_time
        ended = datetime.now(timezone.utc).isoformat()

        prompt_tokens = int(raw.get("prompt_eval_count") or 0)
        completion_tokens = int(raw.get("eval_count") or 0)
        eval_duration_sec = self._duration_ns_to_sec(raw.get("eval_duration"))
        prompt_eval_duration_sec = self._duration_ns_to_sec(raw.get("prompt_eval_duration"))
        load_duration_sec = self._duration_ns_to_sec(raw.get("load_duration"))
        total_duration_sec = self._duration_ns_to_sec(raw.get("total_duration")) or latency_sec
        end_to_end_tokens_per_second = (completion_tokens / latency_sec) if latency_sec > 0 else 0.0
        eval_tokens_per_second = (completion_tokens / eval_duration_sec) if eval_duration_sec > 0 else 0.0
        prompt_tokens_per_second = (prompt_tokens / prompt_eval_duration_sec) if prompt_eval_duration_sec > 0 else 0.0

        return GenerationResult(
            text=(raw.get("response") or raw.get("thinking") or "").strip(),
            prompt=prompt,
            started_at=started,
            ended_at=ended,
            latency_sec=latency_sec,
            total_duration_sec=total_duration_sec,
            provider="ollama",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            load_duration_sec=load_duration_sec,
            prompt_eval_duration_sec=prompt_eval_duration_sec,
            eval_duration_sec=eval_duration_sec,
            tokens_per_second=end_to_end_tokens_per_second,
            eval_tokens_per_second=eval_tokens_per_second,
            prompt_tokens_per_second=prompt_tokens_per_second,
            used_raw_fallback=bool(raw.get("used_raw_fallback")),
            raw_fallback_attempted=bool(raw.get("raw_fallback_attempted") or raw.get("used_raw_fallback")),
            raw=raw,
        )

    async def generate(self, prompt: str, generation_config: GenerationConfig, image_path: Optional[str] = None) -> str:
        """Generate text using local Ollama instance with optional image support."""
        result = await self.generate_with_metadata(prompt, generation_config, image_path=image_path)
        return result.text
    
    async def batch_generate(self, prompts: List[str], generation_config: GenerationConfig, image_paths: Optional[List[str]] = None) -> List[str]:
        """Generate text for multiple prompts with optional images - resource-aware sequential processing."""
        batch_results = await self.batch_generate_with_metadata(prompts, generation_config, image_paths=image_paths)
        return [result.text for result in batch_results]

    async def batch_generate_with_metadata(
        self,
        prompts: List[str],
        generation_config: GenerationConfig,
        image_paths: Optional[List[str]] = None,
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts with optional images - resource-aware sequential processing."""
        results = []
        
        self.logger.info(f"Starting batch generation for {len(prompts)} prompts with model {self.model_name}")
        
        # Get resource-aware batch configuration with generous defaults for slow hardware
        batch_delay = self.config.get('ollama', {}).get('batch_delay', 5.0)  # Longer delay for slow laptops
        checkpoint_interval = self.config.get('ollama', {}).get('checkpoint_interval', 5)  # More frequent updates
        
        # Process sequentially with intelligent resource management
        for i, prompt in enumerate(prompts):
            image_path = None
            if image_paths and i < len(image_paths):
                image_path = image_paths[i]
            
            try:
                self.logger.debug(f"Generating response {i+1}/{len(prompts)} for model {self.model_name}")
                result = await self.generate_with_metadata(prompt, generation_config, image_path)
                
                if not result.text:
                    self.logger.warning(f"Empty result for prompt {i+1} with model {self.model_name}")
                else:
                    self.logger.debug(f"Got response {i+1}/{len(prompts)}: '{result.text[:50]}...'")
                
                results.append(result)
                
                # Progress checkpoint every N items
                if (i + 1) % checkpoint_interval == 0:
                    success_count = sum(1 for r in results if r.text)
                    self.logger.info(f"Progress checkpoint: {i+1}/{len(prompts)} completed, {success_count} successful")
                
                # Resource-friendly delay between requests
                if i < len(prompts) - 1:  # Don't delay after the last request
                    await asyncio.sleep(batch_delay)
                    
            except Exception as e:
                self.logger.warning(f"Error in batch generation item {i+1}: {e}")
                results.append(
                    GenerationResult(
                        text="",
                        prompt=prompt,
                        started_at=datetime.now(timezone.utc).isoformat(),
                        ended_at=datetime.now(timezone.utc).isoformat(),
                        latency_sec=0.0,
                        total_duration_sec=0.0,
                        provider="ollama",
                        raw={"error": str(e)},
                    )
                )
                
                # On error, add extra delay to let slow system recover
                await asyncio.sleep(batch_delay * 3)  # Even longer recovery time for slow hardware
        
        success_count = sum(1 for r in results if r.text)
        success_rate = success_count / len(results) * 100 if results else 0
        self.logger.info(f"Batch generation completed: {success_count}/{len(results)} successful ({success_rate:.1f}%)")
        return results
    
    def get_model_info(self) -> ModelInfo:
        """Get local model information."""
        size_gb = float(self.metadata.get("size_gb", 2.0) or 2.0)
        parameters = self.metadata.get("parameters") or self.metadata.get("parameter_size") or "unknown"
        return ModelInfo(
            name=self.model_name,
            provider="ollama",
            size_gb=size_gb,
            parameters=parameters,
            architecture="transformer",
            license="unknown",
            cost_per_token=0.0,  # Local execution is free
            max_context=int(self.metadata.get("context_length") or self.metadata.get("max_context") or 2048),
            supports_vision=self.supports_vision,
            model_type="vision" if self.supports_vision else "text",
            family=self.metadata.get("family", "unknown"),
            quantization=self.metadata.get("quantization", "unknown"),
        )
    
    async def close(self):
        """Close the aiohttp session and unload the current model."""
        await self.unload_model()
        if self.session:
            await self.session.close()
            self.session = None

class LMStudioModel(BaseModel):
    """Local LM Studio model implementation with vision support."""
    
    def __init__(self, model_name: str, config: Dict, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config, metadata=metadata)
        self.base_url = config.get('lm_studio', {}).get('base_url', 'http://localhost:1234')
        self.session = None
        self.supports_vision = self._detect_vision_support()
    
    def _detect_vision_support(self) -> bool:
        """Detect if model supports vision based on name."""
        vision_keywords = ['llava', 'vision', 'vl', 'multimodal', 'qwen2-vl', 'minicpm-v']
        return any(keyword in self.model_name.lower() for keyword in vision_keywords)
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is not installed. LM Studio support needs local requirements.")
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def unload_model(self):
        """Unload the model from LM Studio memory to free up space."""
        try:
            session = await self._get_session()
            # LM Studio doesn't have a direct unload API, but we can try to switch to a minimal model
            # or just close the connection to let it timeout
            self.logger.info(f"Signaling model unload for LM Studio: {self.model_name}")
        except Exception as e:
            self.logger.warning(f"Could not unload LM Studio model {self.model_name}: {e}")
    
    async def close(self):
        """Close the aiohttp session and unload model."""
        # First unload the model 
        await self.unload_model()
        
        # Then close the session
        if self.session:
            await self.session.close()
            self.session = None
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for vision models."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {e}")
            return ""
    
    async def generate_with_metadata(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        image_path: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text using LM Studio's OpenAI-compatible API."""
        session = await self._get_session()
        started = datetime.now(timezone.utc).isoformat()
        start_time = time.time()
        
        # Prepare messages for chat completion
        messages = []
        
        if image_path and self.supports_vision:
            # Vision model with image input
            base64_image = self._encode_image(image_path)
            if base64_image:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                })
            else:
                # Fallback to text-only if image encoding fails
                messages.append({"role": "user", "content": prompt})
        else:
            # Regular text model
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": generation_config.temperature,
            "max_tokens": generation_config.max_tokens,
            "top_p": generation_config.top_p,
            "frequency_penalty": generation_config.frequency_penalty,
            "presence_penalty": generation_config.presence_penalty,
            "stop": generation_config.stop_sequences or []
        }
        
        try:
            async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("choices") and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        usage = result.get("usage") or {}
                        latency_sec = time.time() - start_time
                        ended = datetime.now(timezone.utc).isoformat()
                        completion_tokens = int(usage.get("completion_tokens") or 0)
                        return GenerationResult(
                            text=content.strip() if content else "",
                            prompt=prompt,
                            started_at=started,
                            ended_at=ended,
                            latency_sec=latency_sec,
                            total_duration_sec=latency_sec,
                            provider="lm_studio",
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=completion_tokens,
                            total_tokens=int(usage.get("total_tokens") or 0),
                            tokens_per_second=(completion_tokens / latency_sec) if latency_sec > 0 else 0.0,
                            raw=result,
                        )
                    else:
                        self.logger.warning(f"LM Studio returned no choices for {self.model_name}")
                        return GenerationResult(
                            text="",
                            prompt=prompt,
                            started_at=started,
                            ended_at=datetime.now(timezone.utc).isoformat(),
                            latency_sec=time.time() - start_time,
                            total_duration_sec=time.time() - start_time,
                            provider="lm_studio",
                            raw=result,
                        )
                else:
                    error_text = await response.text()
                    self.logger.error(f"LM Studio API error {response.status}: {error_text}")
                    return GenerationResult(
                        text="",
                        prompt=prompt,
                        started_at=started,
                        ended_at=datetime.now(timezone.utc).isoformat(),
                        latency_sec=time.time() - start_time,
                        total_duration_sec=time.time() - start_time,
                        provider="lm_studio",
                        raw={"error": error_text, "status": response.status},
                    )
        
        except Exception as e:
            self.logger.error(f"LM Studio generation error: {str(e)}")
            return GenerationResult(
                text="",
                prompt=prompt,
                started_at=started,
                ended_at=datetime.now(timezone.utc).isoformat(),
                latency_sec=time.time() - start_time,
                total_duration_sec=time.time() - start_time,
                provider="lm_studio",
                raw={"error": str(e)},
            )

        ended = datetime.now(timezone.utc).isoformat()
        latency_sec = time.time() - start_time
        return GenerationResult(
            text="",
            prompt=prompt,
            started_at=started,
            ended_at=ended,
            latency_sec=latency_sec,
            total_duration_sec=latency_sec,
            provider="lm_studio",
        )

    async def generate(self, prompt: str, generation_config: GenerationConfig, image_path: Optional[str] = None) -> str:
        """Generate text using LM Studio's OpenAI-compatible API."""
        result = await self.generate_with_metadata(prompt, generation_config, image_path=image_path)
        return result.text
    
    async def batch_generate(self, prompts: List[str], generation_config: GenerationConfig, image_paths: Optional[List[str]] = None) -> List[str]:
        """Generate text for multiple prompts - sequential for local models."""
        batch_results = await self.batch_generate_with_metadata(prompts, generation_config, image_paths=image_paths)
        return [result.text for result in batch_results]

    async def batch_generate_with_metadata(
        self,
        prompts: List[str],
        generation_config: GenerationConfig,
        image_paths: Optional[List[str]] = None,
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts - sequential for local models."""
        results = []
        
        self.logger.info(f"Starting LM Studio batch generation for {len(prompts)} prompts with model {self.model_name}")
        
        # Process sequentially for local LM Studio models to avoid overwhelming the server
        for i, prompt in enumerate(prompts):
            image_path = None
            if image_paths and i < len(image_paths):
                image_path = image_paths[i]
            
            try:
                self.logger.debug(f"LM Studio generating response {i+1}/{len(prompts)} for model {self.model_name}")
                result = await self.generate_with_metadata(prompt, generation_config, image_path)
                
                if not result.text:
                    self.logger.warning(f"Empty result for LM Studio prompt {i+1} with model {self.model_name}")
                else:
                    self.logger.debug(f"LM Studio got response {i+1}/{len(prompts)}: '{result.text[:50]}...'")
                
                results.append(result)
                
                # Small delay between requests to be gentle on local server
                if i < len(prompts) - 1:  # Don't delay after the last request
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.warning(f"Error in LM Studio batch generation item {i+1}: {e}")
                results.append(
                    GenerationResult(
                        text="",
                        prompt=prompt,
                        started_at=datetime.now(timezone.utc).isoformat(),
                        ended_at=datetime.now(timezone.utc).isoformat(),
                        latency_sec=0.0,
                        total_duration_sec=0.0,
                        provider="lm_studio",
                        raw={"error": str(e)},
                    )
                )
        
        success_count = sum(1 for r in results if r.text)
        self.logger.info(f"LM Studio batch generation completed: {success_count}/{len(results)} successful responses")
        return results
    
    def get_model_info(self) -> ModelInfo:
        """Get local model information."""
        return ModelInfo(
            name=self.model_name,
            provider="lm_studio",
            size_gb=float(self.metadata.get("size_gb", 0.0) or 0.0),
            parameters=self.metadata.get("parameters", "unknown"),
            architecture="transformer",
            license="unknown",
            cost_per_token=0.0,  # Local execution is free
            max_context=int(self.metadata.get("context_length") or self.metadata.get("max_context") or 4096),
            supports_vision=self.supports_vision,
            model_type="vision" if self.supports_vision else "text",
            family=self.metadata.get("family", "unknown"),
            quantization=self.metadata.get("quantization", "unknown"),
        )
    
    async def close(self):
        """Close the aiohttp session and signal model unload."""
        await self.unload_model()
        if self.session:
            await self.session.close()
            self.session = None

class ModelManager:
    """Manager for all model providers and instances."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self.logger = logging.getLogger(__name__)
        self._local_models_cache = None
    
    async def discover_local_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all available local models from Ollama and LM Studio."""
        discovered = {
            "ollama": [],
            "lm_studio": []
        }
        
        # Discover Ollama models
        try:
            ollama_models = await self._discover_ollama_models()
            discovered["ollama"] = ollama_models
            self.logger.info(f"Found {len(ollama_models)} Ollama models")
        except Exception as e:
            self.logger.warning(f"Could not discover Ollama models: {e}")
        
        # Discover LM Studio models
        try:
            lm_studio_models = await self._discover_lm_studio_models()
            discovered["lm_studio"] = lm_studio_models
            self.logger.info(f"Found {len(lm_studio_models)} LM Studio models")
        except Exception as e:
            self.logger.warning(f"Could not discover LM Studio models: {e}")
        
        self._local_models_cache = discovered
        return discovered
    
    async def _discover_ollama_models(self) -> List[Dict[str, Any]]:
        """Discover models available in Ollama using API first, CLI fallback."""
        base_url = self.config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        windows_curl = _find_windows_curl()
        ollama_command = _find_ollama_command(self.config)

        def parse_api_models(data: Dict[str, Any]) -> List[Dict[str, Any]]:
            parsed: List[Dict[str, Any]] = []
            for model in data.get('models', []):
                name = model.get('name', 'unknown')
                details = model.get('details') or {}
                supports_vision = _detect_vision_support_from_name(name)
                parsed.append(
                    {
                        'name': name,
                        'provider': 'ollama',
                        'size_gb': round((model.get('size', 0) or 0) / (1024 ** 3), 3),
                        'supports_vision': supports_vision,
                        'model_type': 'vision' if supports_vision else 'text',
                        'cost_per_token': 0.0,
                        'available': True,
                        'family': details.get('family', 'unknown'),
                        'quantization': details.get('quantization_level', 'unknown'),
                        'parameters': details.get('parameter_size', 'unknown'),
                        'context_length': details.get('context_length'),
                    }
                )
            return parsed

        if aiohttp is not None:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(f"{base_url}/api/tags") as response:
                        if response.status == 200:
                            models = parse_api_models(await response.json())
                            self.logger.info(f"Found {len(models)} Ollama models via API")
                            if models:
                                return models
            except asyncio.TimeoutError:
                self.logger.info("Ollama API timed out during discovery, falling back to CLI")
            except Exception as e:
                if isinstance(e, aiohttp.ClientConnectorError):
                    self.logger.info("Ollama API not reachable during discovery, falling back to CLI")
                else:
                    self.logger.warning(f"Could not discover Ollama models via API: {e}")

        if windows_curl:
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [windows_curl, "-s", f"{base_url}/api/tags"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    models = parse_api_models(json.loads(result.stdout))
                    self.logger.info(f"Found {len(models)} Ollama models via Windows curl bridge")
                    if models:
                        return models
            except Exception as e:
                self.logger.warning(f"Could not discover Ollama models via Windows curl bridge: {e}")

        if not ollama_command:
            return []

        try:
            result = subprocess.run(
                [*ollama_command, 'list'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                models = _parse_ollama_list_output(result.stdout)
                self.logger.info(f"Found {len(models)} Ollama models via 'ollama list'")
                return models
            else:
                self.logger.warning(f"'ollama list' failed with return code {result.returncode}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.warning("'ollama list' command timed out")
        except FileNotFoundError:
            self.logger.info("Ollama not found in PATH - skipping Ollama discovery")
        except Exception as e:
            self.logger.error(f"Error running 'ollama list': {e}")
        
        return []
    
    async def _discover_lm_studio_models(self) -> List[Dict[str, Any]]:
        """Discover models available in LM Studio (only if server is running)."""
        base_url = self.config.get('lm_studio', {}).get('base_url', 'http://localhost:1234')
        models = []
        
        if aiohttp is None:
            self.logger.info("aiohttp not installed - skipping LM Studio discovery")
            return models

        try:
            # Quick connection test first
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{base_url}/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        for model in data.get('data', []):
                            name = model.get('id', 'unknown')
                            
                            # Detect vision support
                            supports_vision = any(keyword in name.lower() 
                                                for keyword in ['llava', 'vision', 'vl', 'multimodal', 'qwen2-vl', 'minicpm-v'])
                            
                            models.append({
                                'name': name,
                                'provider': 'lm_studio',
                                'size_gb': 0,  # LM Studio doesn't provide size info easily
                                'supports_vision': supports_vision,
                                'model_type': 'vision' if supports_vision else 'text',
                                'cost_per_token': 0.0,
                                'created': model.get('created', 0),
                                'available': True,
                                'family': model.get('owned_by', 'unknown'),
                                'quantization': model.get('quantization', 'unknown'),
                                'parameters': model.get('parameters', 'unknown'),
                            })
                        
                        self.logger.info(f"Found {len(models)} LM Studio models")
                    else:
                        self.logger.warning(f"LM Studio API returned status {response.status}")
                        
        except asyncio.TimeoutError:
            self.logger.info("LM Studio server not responding (timeout) - skipping LM Studio discovery")
        except Exception as e:
            if isinstance(e, aiohttp.ClientConnectorError):
                self.logger.info("LM Studio server not running - skipping LM Studio discovery")
            else:
                self.logger.warning(f"Could not discover LM Studio models: {e}")
        
        return models
    
    async def get_all_local_models(self) -> List[Dict[str, Any]]:
        """Get all local models from both providers."""
        if self._local_models_cache is None:
            await self.discover_local_models()
        
        all_models = []
        for provider, models in self._local_models_cache.items():
            all_models.extend(models)
        
        return all_models
    
    async def get_local_models_by_type(self, model_type: str = "all") -> List[Dict[str, Any]]:
        """Get local models filtered by type (text, vision, or all)."""
        all_models = await self.get_all_local_models()
        
        if model_type == "all":
            return all_models
        elif model_type == "vision":
            return [m for m in all_models if m.get('supports_vision', False)]
        elif model_type == "text":
            return [m for m in all_models if not m.get('supports_vision', False)]
        else:
            return all_models
    
    async def get_model(self, model_name: str) -> BaseModel:
        """Get or create a model instance."""
        if model_name in self.models:
            return self.models[model_name]
        
        # Determine provider based on model name or configuration
        provider = self._detect_provider(model_name)
        if provider in {"ollama", "lm_studio"} and self._local_models_cache is None:
            await self.discover_local_models()
        metadata = self._get_discovered_model_metadata(model_name)
        
        if provider == "huggingface":
            model = HuggingFaceModel(model_name, self.config, metadata=metadata)
        elif provider == "ollama":
            model = OllamaModel(model_name, self.config, metadata=metadata)
        elif provider == "lm_studio":
            model = LMStudioModel(model_name, self.config, metadata=metadata)
        else:
            raise ValueError(f"Unknown provider for model: {model_name}")
        
        self.models[model_name] = model
        self.logger.info(f"Initialized model: {model_name} with provider: {provider}")
        return model

    def _get_discovered_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Return metadata from the local discovery cache when available."""
        if not self._local_models_cache:
            return {}

        for models in self._local_models_cache.values():
            for model in models:
                if model.get('name') == model_name:
                    return model
        return {}
    
    def _detect_provider(self, model_name: str) -> str:
        """Detect the provider for a model based on its name."""
        # Check if it's explicitly prefixed
        if model_name.startswith("ollama:"):
            return "ollama"
        elif model_name.startswith("lm_studio:"):
            return "lm_studio"
        elif model_name.startswith("hf:"):
            return "huggingface"
        
        # Check if it's in local models cache
        if self._local_models_cache:
            for provider, models in self._local_models_cache.items():
                if any(m['name'] == model_name for m in models):
                    return provider
        
        # Default heuristics
        if "/" in model_name and not model_name.startswith("./"):
            # Likely a HuggingFace model path
            return "huggingface"
        else:
            # Could be local, try Ollama first as it's more common
            return "ollama"
    
    async def list_available_models(self, include_local: bool = True, include_huggingface: bool = True) -> List[Dict[str, Any]]:
        """List all available models across providers."""
        models = []
        
        if include_local:
            local_models = await self.get_all_local_models()
            models.extend(local_models)
        
        if include_huggingface:
            # Add some common HuggingFace models
            hf_models = [
                {"name": "microsoft/DialoGPT-small", "provider": "huggingface", "size": "117M", "supports_vision": False},
                {"name": "microsoft/DialoGPT-medium", "provider": "huggingface", "size": "345M", "supports_vision": False},
                {"name": "google/gemma-2-2b-it", "provider": "huggingface", "size": "2B", "supports_vision": False},
                {"name": "Qwen/Qwen2.5-1.5B-Instruct", "provider": "huggingface", "size": "1.5B", "supports_vision": False},
                {"name": "meta-llama/Llama-3.2-1B-Instruct", "provider": "huggingface", "size": "1B", "supports_vision": False},
                {"name": "meta-llama/Llama-3.2-3B-Instruct", "provider": "huggingface", "size": "3B", "supports_vision": False},
                {"name": "Qwen/Qwen2.5-3B-Instruct", "provider": "huggingface", "size": "3B", "supports_vision": False},
                {"name": "microsoft/Phi-3-mini-4k-instruct", "provider": "huggingface", "size": "3.8B", "supports_vision": False}
            ]
            models.extend(hf_models)
        
        return models
    
    async def cleanup(self):
        """Cleanup all model instances."""
        for model in self.models.values():
            if hasattr(model, 'close'):
                await model.close()
        self.models.clear()

# Confirmed working instruction-tuned models (Chat Completion API compatible)
RECOMMENDED_MODELS = [
    # CONFIRMED WORKING - tested and verified
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    
    # LIKELY WORKING - instruction-tuned models that should work
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    
    # EXPERIMENTAL - may have API limitations but worth testing
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "mistralai/Mistral-7B-Instruct-v0.2",
]
