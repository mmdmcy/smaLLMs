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
    timeout: int = 120

@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    model_name: str
    benchmark_name: str
    accuracy: float
    latency: float
    cost_estimate: float
    timestamp: str
    num_samples: int
    detailed_results: Optional[List[Dict]] = None
    error: Optional[str] = None

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
        """Run a single model-benchmark evaluation."""
        async with self.semaphore:
            self.logger.info(f"Starting evaluation: {config.model_name} on {config.benchmark_name}")
            
            start_time = time.time()
            
            try:
                # Load model and benchmark
                model = await self.model_manager.get_model(config.model_name)
                benchmark = self.benchmark_registry.get_benchmark(config.benchmark_name)
                
                # Run evaluation
                results = await benchmark.evaluate(
                    model=model,
                    num_samples=config.num_samples,
                    few_shot=config.few_shot,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout
                )
                
                # Calculate metrics
                accuracy = self.result_aggregator.calculate_accuracy(results)
                latency = time.time() - start_time
                cost_estimate = self.result_aggregator.estimate_cost(results, config.model_name)
                
                result = EvaluationResult(
                    model_name=config.model_name,
                    benchmark_name=config.benchmark_name,
                    accuracy=accuracy,
                    latency=latency,
                    cost_estimate=cost_estimate,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    num_samples=config.num_samples,
                    detailed_results=results[:5]  # Store only first 5 for space efficiency
                )
                
                # Save result
                await self.storage.save_result(result)
                
                self.logger.info(f"Completed: {config.model_name} on {config.benchmark_name} - Accuracy: {accuracy:.3f}")
                return result
                
            except Exception as e:
                self.logger.error(f"Error evaluating {config.model_name} on {config.benchmark_name}: {str(e)}")
                return EvaluationResult(
                    model_name=config.model_name,
                    benchmark_name=config.benchmark_name,
                    accuracy=0.0,
                    latency=time.time() - start_time,
                    cost_estimate=0.0,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    num_samples=0,
                    error=str(e)
                )
    
    async def evaluate_batch(self, configs: List[EvaluationConfig]) -> List[EvaluationResult]:
        """Run multiple evaluations in parallel."""
        self.logger.info(f"Starting batch evaluation with {len(configs)} configurations")
        
        # Run evaluations concurrently
        tasks = [self.evaluate_single(config) for config in configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, EvaluationResult)]
        
        self.logger.info(f"Completed batch evaluation: {len(valid_results)}/{len(configs)} successful")
        return valid_results
    
    async def evaluate_model_comprehensive(self, model_name: str, benchmarks: List[str] = None) -> List[EvaluationResult]:
        """Run comprehensive evaluation of a model across multiple benchmarks."""
        if benchmarks is None:
            benchmarks = ["mmlu", "gsm8k", "math", "humaneval"]
        
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
    config = EvaluationConfig(
        model_name=model_name,
        benchmark_name=benchmark,
        num_samples=samples
    )
    return await orchestrator.evaluate_single(config)

async def compare_models(models: List[str], benchmark: str = "gsm8k", samples: int = 100) -> List[EvaluationResult]:
    """Compare multiple models on a single benchmark."""
    orchestrator = EvaluationOrchestrator()
    configs = [
        EvaluationConfig(model_name=model, benchmark_name=benchmark, num_samples=samples)
        for model in models
    ]
    return await orchestrator.evaluate_batch(configs)
