"""
Model management layer for smaLLMs platform.
Provides unified interface for different model providers and endpoints.
"""

import asyncio
import aiohttp
import time
import base64
import json
import subprocess
from pathlib import Path
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
    supports_vision: bool = False
    model_type: str = "text"  # text, vision, code
    local_path: Optional[str] = None

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
    """Local Ollama model implementation with vision support."""
    
    def __init__(self, model_name: str, config: Dict):
        super().__init__(model_name, config)
        self.base_url = config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        self.session = None
        self.supports_vision = self._detect_vision_support()
    
    def _detect_vision_support(self) -> bool:
        """Detect if model supports vision based on name."""
        vision_keywords = ['llava', 'vision', 'vl', 'multimodal', 'bakllava', 'moondream']
        return any(keyword in self.model_name.lower() for keyword in vision_keywords)
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def unload_model(self):
        """Unload the model from Ollama memory to free up space."""
        try:
            session = await self._get_session()
            # Use Ollama's proper unload API - set keep_alive to 0 to unload immediately
            payload = {"model": self.model_name, "keep_alive": 0}
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                # This should unload the model by setting keep_alive to 0
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
    
    async def generate(self, prompt: str, generation_config: GenerationConfig, image_path: Optional[str] = None) -> str:
        """Generate text using local Ollama instance with optional image support."""
        session = await self._get_session()
        
        # First check if model is available
        try:
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    tags_data = await response.json()
                    available_models = [model['name'] for model in tags_data.get('models', [])]
                    if self.model_name not in available_models:
                        self.logger.error(f"Model {self.model_name} not found. Available models: {available_models}")
                        return ""
        except Exception as e:
            self.logger.warning(f"Could not check available models: {e}")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": generation_config.temperature,
                "num_predict": generation_config.max_tokens,
                "top_p": generation_config.top_p,
                "stop": generation_config.stop_sequences or []
            },
            "stream": False,
            "keep_alive": "10m"  # Keep model loaded for 10 minutes to avoid reload delays
        }
        
        # Add image if provided and model supports vision
        if image_path and self.supports_vision:
            base64_image = self._encode_image(image_path)
            if base64_image:
                payload["images"] = [base64_image]
        
        # Smart retry logic with progressive timeouts for slow laptops
        base_timeout = self.config.get('ollama', {}).get('timeout', 600)  # Default 10 minutes for slow hardware
        max_retries = self.config.get('ollama', {}).get('max_retries', 5)
        
        # Adapt timeout based on model complexity (larger models need MUCH more time on slow hardware)
        model_size_factor = 1.5  # Base factor increased for slow laptops
        if '3b' in self.model_name.lower() or '7b' in self.model_name.lower():
            model_size_factor = 3.0  # Significantly more time for 3B/7B models
        elif '13b' in self.model_name.lower() or 'large' in self.model_name.lower():
            model_size_factor = 5.0  # Even more time for large models
        elif '1b' in self.model_name.lower() or '1.5b' in self.model_name.lower():
            model_size_factor = 2.0  # Extra time even for small models on slow hardware
        
        for attempt in range(max_retries + 1):
            try:
                # Progressive timeout: increase significantly with each retry for slow laptops
                current_timeout = int(base_timeout * model_size_factor * (1 + attempt * 1.0))  # Double the progression
                timeout = aiohttp.ClientTimeout(total=current_timeout)
                
                self.logger.debug(f"Ollama request attempt {attempt + 1}/{max_retries + 1} with {current_timeout}s timeout for {self.model_name}")
                
                async with session.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout) as response:
                    if response.status == 404:
                        self.logger.error(f"Model {self.model_name} not found in Ollama. Is it pulled?")
                        return ""
                    elif response.status == 500:
                        error_text = await response.text()
                        self.logger.error(f"Ollama server error: {error_text}")
                        return ""
                        
                    response.raise_for_status()
                    result = await response.json()
                    response_text = result.get("response", "").strip()
                    
                    # Add debugging to see what we're getting
                    self.logger.info(f"Ollama response for {self.model_name}: status={response.status}, response_length={len(response_text)}")
                    if not response_text:
                        self.logger.warning(f"Empty response from Ollama for {self.model_name}: {result}")
                    
                    return response_text
            
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    wait_time = 10 * (attempt + 1)  # Longer wait for slow hardware: 10s, 20s, 30s, 40s, 50s
                    self.logger.warning(f"Ollama generation timeout for model {self.model_name} (timeout: {current_timeout}s), retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Ollama generation timeout for model {self.model_name} after {max_retries + 1} attempts with final timeout of {current_timeout}s")
                    return ""
            except aiohttp.ClientError as e:
                self.logger.error(f"Ollama connection error: {str(e)}")
                return ""
            except Exception as e:
                self.logger.error(f"Ollama generation error: {str(e)}")
                return ""
        
        return ""
    
    async def batch_generate(self, prompts: List[str], generation_config: GenerationConfig, image_paths: Optional[List[str]] = None) -> List[str]:
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
                result = await self.generate(prompt, generation_config, image_path)
                
                if not result:
                    self.logger.warning(f"Empty result for prompt {i+1} with model {self.model_name}")
                else:
                    self.logger.debug(f"Got response {i+1}/{len(prompts)}: '{result[:50]}...'")
                
                results.append(result)
                
                # Progress checkpoint every N items
                if (i + 1) % checkpoint_interval == 0:
                    success_count = sum(1 for r in results if r)
                    self.logger.info(f"Progress checkpoint: {i+1}/{len(prompts)} completed, {success_count} successful")
                
                # Resource-friendly delay between requests
                if i < len(prompts) - 1:  # Don't delay after the last request
                    await asyncio.sleep(batch_delay)
                    
            except Exception as e:
                self.logger.warning(f"Error in batch generation item {i+1}: {e}")
                results.append("")
                
                # On error, add extra delay to let slow system recover
                await asyncio.sleep(batch_delay * 3)  # Even longer recovery time for slow hardware
        
        success_count = sum(1 for r in results if r)
        success_rate = success_count / len(results) * 100 if results else 0
        self.logger.info(f"Batch generation completed: {success_count}/{len(results)} successful ({success_rate:.1f}%)")
        return results
    
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
            max_context=2048,
            supports_vision=self.supports_vision,
            model_type="vision" if self.supports_vision else "text"
        )
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

class LMStudioModel(BaseModel):
    """Local LM Studio model implementation with vision support."""
    
    def __init__(self, model_name: str, config: Dict):
        super().__init__(model_name, config)
        self.base_url = config.get('lm_studio', {}).get('base_url', 'http://localhost:1234')
        self.session = None
        self.supports_vision = self._detect_vision_support()
    
    def _detect_vision_support(self) -> bool:
        """Detect if model supports vision based on name."""
        vision_keywords = ['llava', 'vision', 'vl', 'multimodal', 'qwen2-vl', 'minicpm-v']
        return any(keyword in self.model_name.lower() for keyword in vision_keywords)
    
    async def _get_session(self):
        """Get or create aiohttp session."""
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
    
    async def generate(self, prompt: str, generation_config: GenerationConfig, image_path: Optional[str] = None) -> str:
        """Generate text using LM Studio's OpenAI-compatible API."""
        session = await self._get_session()
        
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
                        return content.strip() if content else ""
                    else:
                        self.logger.warning(f"LM Studio returned no choices for {self.model_name}")
                        return ""
                else:
                    error_text = await response.text()
                    self.logger.error(f"LM Studio API error {response.status}: {error_text}")
                    return ""
        
        except Exception as e:
            self.logger.error(f"LM Studio generation error: {str(e)}")
            return ""
    
    async def batch_generate(self, prompts: List[str], generation_config: GenerationConfig, image_paths: Optional[List[str]] = None) -> List[str]:
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
                result = await self.generate(prompt, generation_config, image_path)
                
                if not result:
                    self.logger.warning(f"Empty result for LM Studio prompt {i+1} with model {self.model_name}")
                else:
                    self.logger.debug(f"LM Studio got response {i+1}/{len(prompts)}: '{result[:50]}...'")
                
                results.append(result)
                
                # Small delay between requests to be gentle on local server
                if i < len(prompts) - 1:  # Don't delay after the last request
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.warning(f"Error in LM Studio batch generation item {i+1}: {e}")
                results.append("")
        
        success_count = sum(1 for r in results if r)
        self.logger.info(f"LM Studio batch generation completed: {success_count}/{len(results)} successful responses")
        return results
    
    def get_model_info(self) -> ModelInfo:
        """Get local model information."""
        return ModelInfo(
            name=self.model_name,
            provider="lm_studio",
            size_gb=2.0,  # Default estimate
            parameters="unknown",
            architecture="transformer",
            license="unknown",
            cost_per_token=0.0,  # Local execution is free
            max_context=4096,  # LM Studio typically has higher context
            supports_vision=self.supports_vision,
            model_type="vision" if self.supports_vision else "text"
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
        """Discover models available in Ollama using command line."""
        models = []
        
        try:
            # Use subprocess to run 'ollama list' command
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                # Skip header line
                for line in lines[1:]:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0]
                            size_str = parts[2] if len(parts) > 2 else "0B"
                            
                            # Parse size (convert from "1.2GB" format to GB float)
                            size_gb = 0
                            try:
                                if 'GB' in size_str:
                                    size_gb = float(size_str.replace('GB', ''))
                                elif 'MB' in size_str:
                                    size_gb = float(size_str.replace('MB', '')) / 1024
                                elif 'KB' in size_str:
                                    size_gb = float(size_str.replace('KB', '')) / (1024 * 1024)
                            except:
                                size_gb = 0
                            
                            # Detect vision support
                            supports_vision = any(keyword in name.lower() 
                                                for keyword in ['llava', 'vision', 'vl', 'multimodal', 'bakllava', 'moondream'])
                            
                            models.append({
                                'name': name,
                                'provider': 'ollama',
                                'size_gb': size_gb,
                                'supports_vision': supports_vision,
                                'model_type': 'vision' if supports_vision else 'text',
                                'cost_per_token': 0.0,
                                'available': True
                            })
                            
                self.logger.info(f"Found {len(models)} Ollama models via 'ollama list'")
            else:
                self.logger.warning(f"'ollama list' failed with return code {result.returncode}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.warning("'ollama list' command timed out")
        except FileNotFoundError:
            self.logger.info("Ollama not found in PATH - skipping Ollama discovery")
        except Exception as e:
            self.logger.error(f"Error running 'ollama list': {e}")
        
        return models
    
    async def _discover_lm_studio_models(self) -> List[Dict[str, Any]]:
        """Discover models available in LM Studio (only if server is running)."""
        base_url = self.config.get('lm_studio', {}).get('base_url', 'http://localhost:1234')
        models = []
        
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
                                'available': True
                            })
                        
                        self.logger.info(f"Found {len(models)} LM Studio models")
                    else:
                        self.logger.warning(f"LM Studio API returned status {response.status}")
                        
        except asyncio.TimeoutError:
            self.logger.info("LM Studio server not responding (timeout) - skipping LM Studio discovery")
        except aiohttp.ClientConnectorError:
            self.logger.info("LM Studio server not running - skipping LM Studio discovery")
        except Exception as e:
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
        
        if provider == "huggingface":
            model = HuggingFaceModel(model_name, self.config)
        elif provider == "ollama":
            model = OllamaModel(model_name, self.config)
        elif provider == "lm_studio":
            model = LMStudioModel(model_name, self.config)
        else:
            raise ValueError(f"Unknown provider for model: {model_name}")
        
        self.models[model_name] = model
        self.logger.info(f"Initialized model: {model_name} with provider: {provider}")
        return model
    
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
