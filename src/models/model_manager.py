"""
Model management layer for smaLLMs platform.
Provides unified interface for different model providers and endpoints.
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from huggingface_hub import InferenceClient
import logging

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

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = None

class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(self, model_name: str, config: Dict):
        self.model_name = model_name
        self.config = config
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

class HuggingFaceModel(BaseModel):
    """Hugging Face Inference Providers model implementation."""
    
    def __init__(self, model_name: str, config: Dict):
        super().__init__(model_name, config)
        hf_config = config.get('huggingface', {})
        self.token = hf_config.get('token')
        
        # Use the modern InferenceClient for Inference Providers
        self.client = InferenceClient(
            api_key=self.token,
            provider="auto"  # Let HF choose the best provider automatically
        )
        self.max_retries = hf_config.get('inference_endpoints', {}).get('max_retries', 3)
    
    async def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        """Generate text using HF Inference Providers Chat Completions API only."""
        for attempt in range(self.max_retries):
            try:
                # Use the modern chat completions API
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=generation_config.max_tokens,
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                    stop=generation_config.stop_sequences or []
                )
                
                # Extract content from the response
                if completion.choices and len(completion.choices) > 0:
                    content = completion.choices[0].message.content
                    if content and content.strip():
                        return content.strip()
                    else:
                        self.logger.warning(f"Model {self.model_name} returned empty content in attempt {attempt + 1}")
                        if attempt == self.max_retries - 1:
                            self.logger.error(f"Model {self.model_name} consistently returns empty responses - not compatible with chat completions API")
                        return ""
                else:
                    self.logger.warning(f"Model {self.model_name} returned no choices in attempt {attempt + 1}")
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"Model {self.model_name} consistently returns no choices - API compatibility issue")
                    return ""
            
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                
                # StopIteration typically means the model doesn't support chat completions
                if error_type == "StopIteration":
                    self.logger.error(f"Model {self.model_name} doesn't support chat completions API (StopIteration). Use instruction-tuned models only.")
                    return ""
                
                self.logger.warning(f"Chat API attempt {attempt + 1} failed for {self.model_name}: {error_type}: {error_msg}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All {self.max_retries} attempts failed for {self.model_name}. Final error: {error_type}: {error_msg}")
                    return ""
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return ""
    
    async def batch_generate(self, prompts: List[str], generation_config: GenerationConfig) -> List[str]:
        """Generate text for multiple prompts with rate limiting."""
        batch_size = self.config.get('huggingface', {}).get('inference_endpoints', {}).get('batch_size', 10)
        
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_tasks = [self.generate(prompt, generation_config) for prompt in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in batch
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch generation error: {str(result)}")
                    results.append("")  # Empty string for failed generations
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
            max_context=2048
        )

class OllamaModel(BaseModel):
    """Local Ollama model implementation."""
    
    def __init__(self, model_name: str, config: Dict):
        super().__init__(model_name, config)
        self.base_url = config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        """Generate text using local Ollama instance."""
        session = await self._get_session()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": generation_config.temperature,
                "num_predict": generation_config.max_tokens,
                "top_p": generation_config.top_p,
                "stop": generation_config.stop_sequences or []
            },
            "stream": False
        }
        
        try:
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("response", "").strip()
        
        except Exception as e:
            self.logger.error(f"Ollama generation error: {str(e)}")
            raise e
    
    async def batch_generate(self, prompts: List[str], generation_config: GenerationConfig) -> List[str]:
        """Generate text for multiple prompts."""
        tasks = [self.generate(prompt, generation_config) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [result if not isinstance(result, Exception) else "" for result in results]
    
    def get_model_info(self) -> ModelInfo:
        """Get local model information."""
        return ModelInfo(
            name=self.model_name,
            provider="ollama",
            size_gb=2.0,  # Default estimate
            parameters="unknown",
            architecture="transformer",
            license="unknown",
            cost_per_token=0.0,  # Local execution is free
            max_context=2048
        )
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

class ModelManager:
    """Manager for all model providers and instances."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self.logger = logging.getLogger(__name__)
    
    async def get_model(self, model_name: str) -> BaseModel:
        """Get or create a model instance."""
        if model_name in self.models:
            return self.models[model_name]
        
        # Determine provider based on model name or configuration
        provider = self._detect_provider(model_name)
        
        if provider == "huggingface":
            model = HuggingFaceModel(model_name, self.config)
        elif provider == "ollama":
            model = OllamaModel(model_name, self.config)
        else:
            raise ValueError(f"Unknown provider for model: {model_name}")
        
        self.models[model_name] = model
        self.logger.info(f"Initialized model: {model_name} with provider: {provider}")
        return model
    
    def _detect_provider(self, model_name: str) -> str:
        """Detect the provider for a model based on its name."""
        # Simple heuristics - can be improved with a model registry
        if "ollama:" in model_name or model_name in ["gemma:2b", "phi3", "llama3.1"]:
            return "ollama"
        else:
            return "huggingface"
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models across providers."""
        # This would typically query each provider's API
        return [
            {"name": "microsoft/DialoGPT-small", "provider": "huggingface", "size": "117M"},
            {"name": "microsoft/DialoGPT-medium", "provider": "huggingface", "size": "345M"},
            {"name": "google/gemma-2b", "provider": "huggingface", "size": "2B"},
            {"name": "microsoft/phi-2", "provider": "huggingface", "size": "2.7B"},
            {"name": "deepseek-ai/deepseek-coder-6.7b-instruct", "provider": "huggingface", "size": "6.7B"},
            {"name": "mistralai/Mistral-7B-Instruct-v0.2", "provider": "huggingface", "size": "7B"},
            {"name": "deepseek-ai/DeepSeek-R1-14B-Chat", "provider": "huggingface", "size": "14B"}
        ]
    
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
