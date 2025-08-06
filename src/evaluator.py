"""
Core evaluation orchestrator for smaLLMs platform.
Handles the coordination of model evaluations across different benchmarks.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import yaml

from models.model_manager import ModelManager
from benchmarks.benchmark_registry import BenchmarkRegistry
from metrics.result_aggregator import ResultAggregator
from utils.storage import ResultStorage

@dataclass
class EvaluationConfig:
    """Configuration for a single evaluation run."""
    model_name: str
    benchmark_name: str
    num_samples: int = 100
    few_shot: int = 5
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: int = 600  # Increased to 10 minutes for slow local models

@dataclass
class EvaluationResult:
    """Result of a single evaluation with comprehensive statistics."""
    model_name: str
    benchmark_name: str
    accuracy: float
    latency: float
    cost_estimate: float
    timestamp: str
    num_samples: int
    detailed_results: Optional[List[Dict]] = None
    error: Optional[str] = None
    
    # Enhanced statistics
    provider: str = "Unknown"
    model_size_gb: float = 0.0
    model_parameters: str = "Unknown"
    load_time: float = 0.0
    avg_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    error_count: int = 0
    total_requests: int = 0
    success_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all statistics."""
        return {
            'model': self.model_name,
            'model_name': self.model_name,
            'benchmark': self.benchmark_name,
            'benchmark_name': self.benchmark_name,
            'accuracy': self.accuracy,
            'duration': self.latency,
            'latency': self.latency,
            'cost': self.cost_estimate,
            'cost_estimate': self.cost_estimate,
            'timestamp': self.timestamp,
            'num_samples': self.num_samples,
            'error': self.error,
            
            # Enhanced statistics
            'provider': self.provider,
            'model_provider': self.provider,
            'model_size_gb': self.model_size_gb,
            'size_gb': self.model_size_gb,
            'model_parameters': self.model_parameters,
            'parameters': self.model_parameters,
            'load_time': self.load_time,
            'model_load_time': self.load_time,
            'avg_response_time': self.avg_response_time,
            'min_response_time': self.min_response_time,
            'max_response_time': self.max_response_time,
            'tokens_per_second': self.tokens_per_second,
            'throughput': self.tokens_per_second,
            'memory_usage_mb': self.memory_usage_mb,
            'memory_usage': self.memory_usage_mb,
            'error_count': self.error_count,
            'total_requests': self.total_requests,
            'success_count': self.success_count,
            'detailed_results': self.detailed_results
        }

class EvaluationOrchestrator:
    """
    Main orchestrator for running evaluations across models and benchmarks.
    Optimized for minimal resource usage and cloud-first approach.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.config = self._load_config(config_path)
        self.model_manager = ModelManager(self.config)
        self.benchmark_registry = BenchmarkRegistry(self.config)
        self.result_aggregator = ResultAggregator(self.config)
        self.storage = ResultStorage(self.config)
        
        # Session management for organized results
        self.session_id = None
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(
            self.config.get('evaluation', {}).get('max_concurrent_requests', 5)
        )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        # Ensure absolute path
        if not Path(config_path).is_absolute():
            config_path = Path(__file__).parent.parent / config_path
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    async def evaluate_single(self, config: EvaluationConfig) -> EvaluationResult:
        """Run a single model-benchmark evaluation with comprehensive statistics collection."""
        async with self.semaphore:
            self.logger.info(f"Starting evaluation: {config.model_name} on {config.benchmark_name}")
            
            start_time = time.time()
            load_start_time = time.time()
            
            try:
                # Load model and collect model info
                model = await self.model_manager.get_model(config.model_name)
                load_time = time.time() - load_start_time
                
                # Get model information
                model_info = model.get_model_info()
                
                benchmark = self.benchmark_registry.get_benchmark(config.benchmark_name)
                
                # Track response times during evaluation
                eval_start_time = time.time()
                response_times = []
                
                # Run evaluation with enhanced timing
                results = await benchmark.evaluate(
                    model=model,
                    num_samples=config.num_samples,
                    few_shot=config.few_shot,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout
                )
                
                eval_duration = time.time() - eval_start_time
                
                # Collect response time statistics from results
                for result_item in results:
                    if isinstance(result_item, dict) and 'response_time' in result_item:
                        response_times.append(result_item['response_time'])
                
                # Calculate comprehensive metrics
                accuracy = self.result_aggregator.calculate_accuracy(results)
                latency = time.time() - start_time
                cost_estimate = self.result_aggregator.estimate_cost(results, config.model_name)
                
                # Calculate enhanced statistics
                total_requests = len(results)
                error_count = sum(1 for r in results if isinstance(r, dict) and r.get('error'))
                success_count = total_requests - error_count
                
                # Response time statistics
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    min_response_time = min(response_times)
                    max_response_time = max(response_times)
                else:
                    avg_response_time = eval_duration / max(1, total_requests)
                    min_response_time = 0.0
                    max_response_time = 0.0
                
                # Calculate tokens per second (rough estimation)
                estimated_tokens = total_requests * config.max_tokens * 0.3  # Rough estimate
                tokens_per_second = estimated_tokens / max(eval_duration, 0.1)
                
                # Memory usage estimation (basic heuristic based on model size)
                memory_usage_mb = model_info.size_gb * 1024 * 1.2 if model_info.size_gb > 0 else 0.0
                
                result = EvaluationResult(
                    model_name=config.model_name,
                    benchmark_name=config.benchmark_name,
                    accuracy=accuracy,
                    latency=latency,
                    cost_estimate=cost_estimate,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    num_samples=config.num_samples,
                    detailed_results=results[:5],  # Store only first 5 for space efficiency
                    
                    # Enhanced statistics
                    provider=model_info.provider,
                    model_size_gb=model_info.size_gb,
                    model_parameters=model_info.parameters,
                    load_time=load_time,
                    avg_response_time=avg_response_time,
                    min_response_time=min_response_time,
                    max_response_time=max_response_time,
                    tokens_per_second=tokens_per_second,
                    memory_usage_mb=memory_usage_mb,
                    error_count=error_count,
                    total_requests=total_requests,
                    success_count=success_count
                )
                
                # Save result with session
                await self.storage.save_result(result, session_id=self.session_id)
                
                self.logger.info(f"Completed: {config.model_name} on {config.benchmark_name} - Accuracy: {accuracy:.3f}, Provider: {model_info.provider}, Load Time: {load_time:.2f}s")
                return result
                
            except Exception as e:
                self.logger.error(f"Error evaluating {config.model_name} on {config.benchmark_name}: {str(e)}")
                
                # Try to get basic model info even on error
                provider = "Unknown"
                model_size_gb = 0.0
                model_parameters = "Unknown"
                
                try:
                    model = await self.model_manager.get_model(config.model_name)
                    model_info = model.get_model_info()
                    provider = model_info.provider
                    model_size_gb = model_info.size_gb
                    model_parameters = model_info.parameters
                except:
                    pass
                
                return EvaluationResult(
                    model_name=config.model_name,
                    benchmark_name=config.benchmark_name,
                    accuracy=0.0,
                    latency=time.time() - start_time,
                    cost_estimate=0.0,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    num_samples=0,
                    error=str(e),
                    provider=provider,
                    model_size_gb=model_size_gb,
                    model_parameters=model_parameters,
                    error_count=1,
                    total_requests=1,
                    success_count=0
                )
            finally:
                # Cleanup model sessions after each evaluation
                await self.model_manager.cleanup()
    
    async def cleanup(self):
        """Cleanup all resources."""
        await self.model_manager.cleanup()
    
    async def evaluate_batch(self, configs: List[EvaluationConfig]) -> List[EvaluationResult]:
        """Run multiple evaluations SEQUENTIALLY for slow laptop compatibility."""
        self.logger.info(f"Starting batch evaluation with {len(configs)} configurations (SEQUENTIAL for slow hardware)")
        
        # Create session for this batch
        self.session_id = self.storage.create_session()
        self.logger.info(f"Created evaluation session: {self.session_id}")
        
        try:
            # Run evaluations SEQUENTIALLY for slow laptops - one at a time
            results = []
            batch_delay = self.config.get('evaluation', {}).get('batch_delay', 5.0)  # Delay between evaluations
            
            for i, config in enumerate(configs):
                self.logger.info(f"Processing evaluation {i+1}/{len(configs)}: {config.model_name} on {config.benchmark_name}")
                
                try:
                    result = await self.evaluate_single(config)
                    results.append(result)
                    
                    # Progress update every few evaluations
                    if (i + 1) % 5 == 0:
                        success_count = sum(1 for r in results if not r.error)
                        self.logger.info(f"Progress: {i+1}/{len(configs)} completed, {success_count} successful")
                    
                    # Resource-friendly delay between evaluations (except after the last one)
                    if i < len(configs) - 1:
                        self.logger.debug(f"Waiting {batch_delay}s before next evaluation...")
                        await asyncio.sleep(batch_delay)
                        
                except Exception as e:
                    self.logger.error(f"Error in evaluation {i+1}: {e}")
                    # Create error result
                    error_result = EvaluationResult(
                        model_name=config.model_name,
                        benchmark_name=config.benchmark_name,
                        accuracy=0.0,
                        latency=0.0,
                        cost_estimate=0.0,
                        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                        num_samples=0,
                        error=str(e)
                    )
                    results.append(error_result)
                    
                    # Extra delay after errors to let slow system recover
                    await asyncio.sleep(batch_delay * 2)
            
            # Filter out exceptions and count successes
            valid_results = [r for r in results if isinstance(r, EvaluationResult)]
            success_count = sum(1 for r in valid_results if not r.error)
            
            self.logger.info(f"Completed SEQUENTIAL batch evaluation: {success_count}/{len(configs)} successful, {len(valid_results)} total results")
            return valid_results
        finally:
            # Final cleanup after batch
            await self.model_manager.cleanup()
    
    async def evaluate_model_comprehensive(self, model_name: str, benchmarks: List[str] = None) -> List[EvaluationResult]:
        """Run comprehensive evaluation of a model across multiple benchmarks."""
        if benchmarks is None:
            benchmarks = ["mmlu", "gsm8k", "math", "humaneval"]
        
        # Create session for this comprehensive evaluation
        self.session_id = self.storage.create_session()
        self.logger.info(f"Created comprehensive evaluation session: {self.session_id}")
        
        configs = []
        for benchmark in benchmarks:
            config = EvaluationConfig(
                model_name=model_name,
                benchmark_name=benchmark,
                num_samples=self.config.get('evaluation', {}).get('default_samples', 100)
            )
            configs.append(config)
        
        return await self.evaluate_batch(configs)
    
    async def run_leaderboard_update(self, models: List[str], benchmarks: List[str] = None) -> Dict[str, Any]:
        """Update leaderboard with latest results for specified models."""
        if benchmarks is None:
            benchmarks = ["mmlu", "gsm8k", "math", "humaneval"]
        
        # Create session for this leaderboard update
        self.session_id = self.storage.create_session()
        self.logger.info(f"Created leaderboard update session: {self.session_id}")
        
        all_configs = []
        for model in models:
            for benchmark in benchmarks:
                config = EvaluationConfig(
                    model_name=model,
                    benchmark_name=benchmark,
                    num_samples=self.config.get('evaluation', {}).get('default_samples', 100)
                )
                all_configs.append(config)
        
        results = await self.evaluate_batch(all_configs)
        
        # Generate leaderboard data
        leaderboard = self.result_aggregator.generate_leaderboard(results)
        
        # Save leaderboard
        await self.storage.save_leaderboard(leaderboard)
        
        return leaderboard
    
    def get_cached_results(self, model_name: str = None, benchmark_name: str = None) -> List[EvaluationResult]:
        """Retrieve cached evaluation results."""
        return self.storage.get_cached_results(model_name, benchmark_name)
    
    async def cleanup_old_results(self):
        """Clean up old cached results to save space."""
        await self.storage.cleanup_old_results()

# Convenience functions for quick evaluation
async def quick_eval(model_name: str, benchmark: str = "gsm8k", samples: int = 50) -> EvaluationResult:
    """Quick evaluation for testing purposes."""
    orchestrator = EvaluationOrchestrator()
    # Create session for quick eval
    orchestrator.session_id = orchestrator.storage.create_session()
    config = EvaluationConfig(
        model_name=model_name,
        benchmark_name=benchmark,
        num_samples=samples
    )
    return await orchestrator.evaluate_single(config)

async def compare_models(models: List[str], benchmark: str = "gsm8k", samples: int = 100) -> List[EvaluationResult]:
    """Compare multiple models on a single benchmark."""
    orchestrator = EvaluationOrchestrator()
    # Create session for model comparison
    orchestrator.session_id = orchestrator.storage.create_session()
    configs = [
        EvaluationConfig(model_name=model, benchmark_name=benchmark, num_samples=samples)
        for model in models
    ]
    return await orchestrator.evaluate_batch(configs)
