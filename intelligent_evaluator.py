#!/usr/bin/env python3
"""
smaLLMs Intelligent Evaluation Orchestrator
==========================================
Cost-optimized and rate-limit aware batch evaluation system.
Designed to maximize value while respecting API limits and costs.
"""

import asyncio
import time
import json
import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluator import EvaluationOrchestrator, EvaluationConfig
from beautiful_terminal import BeautifulTerminal, start_evaluation_display, update_evaluation_progress, evaluation_completed, evaluation_failed, start_evaluation

@dataclass
class IntelligentEvaluationConfig:
    """Smart configuration that optimizes for cost and efficiency."""
    models: List[str]
    benchmarks: List[str]
    
    # Cost optimization
    samples_per_eval: int = 25  # Start small, can increase
    progressive_sampling: bool = True  # Start with fewer samples, increase for promising models
    
    # Rate limiting
    requests_per_minute: int = 30  # Conservative rate limit
    concurrent_models: int = 1  # Only one model at a time to avoid limits
    delay_between_evals: float = 2.0  # Seconds between evaluations
    
    # Smart scheduling
    prioritize_fast_models: bool = True  # Test smaller models first
    early_stopping: bool = True  # Stop if model fails multiple times
    
    # Results management
    output_dir: str = "intelligent_results"
    save_after_each: bool = True  # Save immediately to avoid losing progress

@dataclass
class ModelPerformanceTracker:
    """Track model performance and reliability."""
    model_name: str
    successful_evals: int = 0
    failed_evals: int = 0
    avg_accuracy: float = 0.0
    avg_latency: float = 0.0
    total_cost: float = 0.0
    reliability_score: float = 1.0  # 0-1, decreases with failures
    
    def update_success(self, accuracy: float, latency: float, cost: float):
        """Update with successful evaluation."""
        self.successful_evals += 1
        total_evals = self.successful_evals
        self.avg_accuracy = ((self.avg_accuracy * (total_evals - 1)) + accuracy) / total_evals
        self.avg_latency = ((self.avg_latency * (total_evals - 1)) + latency) / total_evals
        self.total_cost += cost
        # Improve reliability score slightly with success
        self.reliability_score = min(1.0, self.reliability_score + 0.05)
    
    def update_failure(self):
        """Update with failed evaluation."""
        self.failed_evals += 1
        # Decrease reliability score with failures
        self.reliability_score = max(0.1, self.reliability_score - 0.2)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_evals + self.failed_evals
        return self.successful_evals / total if total > 0 else 0.0
    
    @property
    def should_continue(self) -> bool:
        """Determine if we should continue testing this model."""
        # Stop if reliability is too low or too many failures
        if self.reliability_score < 0.3:
            return False
        if self.failed_evals >= 3 and self.success_rate < 0.5:
            return False
        return True

class IntelligentEvaluationOrchestrator:
    """
    Smart evaluation orchestrator that optimizes for cost, efficiency, and reliability.
    """
    
    def __init__(self, config: IntelligentEvaluationConfig):
        self.config = config
        self.orchestrator = EvaluationOrchestrator()
        
        # Performance tracking
        self.model_trackers: Dict[str, ModelPerformanceTracker] = {}
        for model in config.models:
            self.model_trackers[model] = ModelPerformanceTracker(model)
        
        # Rate limiting
        self.last_request_time = 0.0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        
        # Organized results storage
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.date_folder = datetime.now().strftime("%Y-%m-%d")
        self.run_folder = f"run_{self.timestamp}"
        
        # Create unified organized directory structure
        self.results_base = Path("smaLLMs_results")  # Single unified folder
        self.date_path = self.results_base / self.date_folder
        self.run_path = self.date_path / self.run_folder
        
        # Create all necessary directories
        self.run_path.mkdir(parents=True, exist_ok=True)
        (self.run_path / "individual_results").mkdir(exist_ok=True)
        (self.run_path / "reports").mkdir(exist_ok=True)
        (self.run_path / "exports").mkdir(exist_ok=True)
        
        self._setup_logging()
        
        # Statistics
        self.total_cost = 0.0
        self.total_time = 0.0
        self.evaluations_completed = 0
        
    def _setup_logging(self):
        """Setup quiet logging for beautiful terminal."""
        # Create log directory if it doesn't exist
        log_dir = self.run_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup file logging only (no console output for clean terminal)
        log_file = log_dir / "intelligent_evaluation.log"
        
        # Configure logging for this evaluator only
        logger = logging.getLogger("IntelligentEvaluator")
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add file handler only
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        self.logger = logger
    
    async def run_intelligent_evaluation(self) -> Dict[str, Any]:
        """Run cost-optimized intelligent evaluation with beautiful terminal."""
        
        # Initialize beautiful terminal
        start_evaluation_display(self.config.models, self.config.benchmarks)
        
        # Sort models by estimated efficiency (smaller models first)
        sorted_models = self._sort_models_by_efficiency()
        
        all_results = []
        start_time = time.time()
        
        # Evaluate each model sequentially to avoid rate limits
        for model_idx, model in enumerate(sorted_models):
            model_results = await self._evaluate_model_intelligently(model)
            all_results.extend(model_results)
            
            # Check if we should continue with this model
            tracker = self.model_trackers[model]
            if not tracker.should_continue:
                # Mark remaining benchmarks as failed for this model
                for benchmark in self.config.benchmarks:
                    if not any(r.get('model') == model and r.get('benchmark') == benchmark for r in model_results):
                        evaluation_failed(model, benchmark)
                continue
            
            # Progressive delay between models to respect rate limits
            if model_idx < len(sorted_models) - 1:
                await asyncio.sleep(self.config.delay_between_evals * 2)
        
        self.total_time = time.time() - start_time
        
        # Compile intelligent results
        final_results = self._compile_intelligent_results(all_results)
        
        # Save comprehensive report
        await self._save_intelligent_report(final_results)
        
        # Print final summary in terminal
        print(f"\n\n🎯 EVALUATION COMPLETED!")
        print(f"✅ Total Evaluations: {len(all_results)}")
        print(f"💰 Total Cost: ${final_results['execution_summary']['total_cost']:.4f}")
        print(f"⏱️  Duration: {self.total_time/60:.1f} minutes")
        
        return final_results
    
    def _sort_models_by_efficiency(self) -> List[str]:
        """Sort models by estimated efficiency (smaller/faster first)."""
        # Estimate model efficiency based on known characteristics
        model_sizes = {}
        
        for model in self.config.models:
            # Estimate size from model name
            if "1B" in model or "1b" in model:
                model_sizes[model] = 1
            elif "2B" in model or "2b" in model:
                model_sizes[model] = 2
            elif "3B" in model or "3b" in model:
                model_sizes[model] = 3
            elif "7B" in model or "7b" in model:
                model_sizes[model] = 7
            else:
                model_sizes[model] = 5  # Default estimate
        
        # Sort by size (smaller first for faster testing)
        return sorted(self.config.models, key=lambda m: model_sizes.get(m, 5))
    
    async def _evaluate_model_intelligently(self, model: str) -> List[Dict]:
        """Evaluate a single model across all benchmarks intelligently."""
        model_results = []
        tracker = self.model_trackers[model]
        
        for benchmark_idx, benchmark in enumerate(self.config.benchmarks):
            # Mark as started
            start_evaluation(model, benchmark)
            
            # Determine sample size based on model performance so far
            samples = self._determine_sample_size(tracker)
            
            # Rate limiting check
            await self._ensure_rate_limit()
            
            try:
                # Create evaluation config
                eval_config = EvaluationConfig(
                    model_name=model,
                    benchmark_name=benchmark,
                    num_samples=samples,
                    temperature=0.0
                )
                
                # Run evaluation with timing
                eval_start = time.time()
                result = await self.orchestrator.evaluate_single(eval_config)
                eval_duration = time.time() - eval_start
                
                if result and not result.error:
                    # Success
                    tracker.update_success(result.accuracy, result.latency, result.cost_estimate)
                    self.total_cost += result.cost_estimate
                    self.evaluations_completed += 1
                    
                    evaluation_completed(model, benchmark, result.accuracy, 
                                       result.cost_estimate, eval_duration)
                    
                    eval_result = {
                        'model': model,
                        'benchmark': benchmark,
                        'accuracy': result.accuracy,
                        'latency': result.latency,
                        'cost': result.cost_estimate,
                        'samples': samples,
                        'duration': eval_duration,
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    model_results.append(eval_result)
                    
                    # Save intermediate result immediately
                    if self.config.save_after_each:
                        await self._save_intermediate_result(eval_result)
                
                else:
                    # Failure
                    error_msg = result.error if result else "No result returned"
                    tracker.update_failure()
                    
                    evaluation_failed(model, benchmark)
                    
                    # Check if we should stop this model
                    if not tracker.should_continue:
                        break
                
            except Exception as e:
                tracker.update_failure()
                evaluation_failed(model, benchmark)
                
                # Check if we should stop this model
                if not tracker.should_continue:
                    break
            
            # Delay between benchmark evaluations
            if benchmark_idx < len(self.config.benchmarks) - 1:
                await asyncio.sleep(self.config.delay_between_evals)
        
        return model_results
    
    def _determine_sample_size(self, tracker: ModelPerformanceTracker) -> int:
        """Intelligently determine sample size based on model performance."""
        if not self.config.progressive_sampling:
            return self.config.samples_per_eval
        
        # Start with base sample size
        base_samples = self.config.samples_per_eval
        
        # Increase samples for reliable, high-performing models
        if tracker.successful_evals >= 2:
            if tracker.avg_accuracy > 0.7 and tracker.reliability_score > 0.8:
                return min(base_samples * 2, 100)  # Cap at 100 for cost control
            elif tracker.avg_accuracy > 0.5 and tracker.reliability_score > 0.6:
                return min(base_samples + 15, 75)
        
        # Reduce samples for struggling models
        if tracker.reliability_score < 0.5:
            return max(base_samples // 2, 10)  # Minimum 10 samples
        
        return base_samples
    
    async def _ensure_rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.minute_start = current_time
        
        # Check if we need to wait
        if self.requests_this_minute >= self.config.requests_per_minute:
            wait_time = 60 - (current_time - self.minute_start)
            if wait_time > 0:
                self.logger.info(f"⏳ Rate limit reached, waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.requests_this_minute = 0
                self.minute_start = time.time()
        
        # Enforce minimum delay between requests
        time_since_last = current_time - self.last_request_time
        min_delay = 60 / self.config.requests_per_minute
        
        if time_since_last < min_delay:
            wait_time = min_delay - time_since_last
            await asyncio.sleep(wait_time)
        
        self.requests_this_minute += 1
        self.last_request_time = time.time()
    
    async def _save_intermediate_result(self, result: Dict):
        """Save individual evaluation result in organized structure."""
        model_name = result.get('model', 'unknown').replace('/', '_')
        benchmark_name = result.get('benchmark', 'unknown')
        timestamp = int(time.time())
        filename = f"{model_name}_{benchmark_name}_{timestamp}.json"
        
        filepath = self.run_path / "individual_results" / filename
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _compile_intelligent_results(self, all_results: List[Dict]) -> Dict:
        """Compile intelligent analysis of results."""
        successful_results = [r for r in all_results if r.get('status') == 'success']
        
        # Model performance analysis
        model_analysis = {}
        for model, tracker in self.model_trackers.items():
            model_results = [r for r in successful_results if r['model'] == model]
            
            if model_results:
                model_analysis[model] = {
                    'evaluations_completed': len(model_results),
                    'success_rate': tracker.success_rate,
                    'avg_accuracy': tracker.avg_accuracy,
                    'avg_latency': tracker.avg_latency,
                    'total_cost': tracker.total_cost,
                    'reliability_score': tracker.reliability_score,
                    'cost_per_eval': tracker.total_cost / len(model_results) if model_results else 0,
                    'value_score': tracker.avg_accuracy / (tracker.total_cost + 0.001),  # Accuracy per dollar
                    'benchmarks': {r['benchmark']: r['accuracy'] for r in model_results}
                }
        
        # Cost efficiency ranking
        cost_efficient_models = sorted(
            model_analysis.items(),
            key=lambda x: x[1]['value_score'],
            reverse=True
        )
        
        # Performance ranking
        performance_ranking = sorted(
            model_analysis.items(),
            key=lambda x: x[1]['avg_accuracy'],
            reverse=True
        )
        
        return {
            'evaluation_config': asdict(self.config),
            'execution_summary': {
                'total_evaluations': self.evaluations_completed,
                'total_cost': self.total_cost,
                'total_time_minutes': self.total_time / 60,
                'cost_per_minute': self.total_cost / (self.total_time / 60) if self.total_time > 0 else 0,
                'avg_cost_per_eval': self.total_cost / self.evaluations_completed if self.evaluations_completed > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'model_analysis': model_analysis,
            'rankings': {
                'cost_efficiency': cost_efficient_models[:5],  # Top 5
                'performance': performance_ranking[:5],  # Top 5
            },
            'detailed_results': all_results,
            'cost_breakdown': {
                model: tracker.total_cost 
                for model, tracker in self.model_trackers.items()
            }
        }
    
    async def _save_intelligent_report(self, results: Dict):
        """Save comprehensive intelligent report in organized structure."""
        # Save full JSON report
        json_filename = self.run_path / "reports" / "full_evaluation_report.json"
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save human-readable summary
        summary_filename = self.run_path / "reports" / "evaluation_summary.txt"
        with open(summary_filename, 'w') as f:
            f.write("smaLLMs Intelligent Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            exec_summary = results['execution_summary']
            f.write(f"Execution Summary:\n")
            f.write(f"Total Evaluations: {exec_summary['total_evaluations']}\n")
            f.write(f"Total Cost: ${exec_summary['total_cost']:.4f}\n")
            f.write(f"Duration: {exec_summary['total_time_minutes']:.1f} minutes\n")
            f.write(f"Cost Efficiency: ${exec_summary['avg_cost_per_eval']:.6f} per evaluation\n\n")
            
            f.write("Cost Efficiency Ranking:\n")
            f.write("-" * 30 + "\n")
            for i, (model, stats) in enumerate(results['rankings']['cost_efficiency'], 1):
                f.write(f"{i}. {model}: {stats['value_score']:.3f} accuracy per $\n")
                f.write(f"   Accuracy: {stats['avg_accuracy']:.3f}, Cost: ${stats['total_cost']:.4f}\n")
            
            f.write("\nPerformance Ranking:\n")
            f.write("-" * 30 + "\n")
            for i, (model, stats) in enumerate(results['rankings']['performance'], 1):
                f.write(f"{i}. {model}: {stats['avg_accuracy']:.3f} avg accuracy\n")
                f.write(f"   Success Rate: {stats['success_rate']:.1%}, Reliability: {stats['reliability_score']:.2f}\n")
        
        self.logger.info(f"Full report saved: {json_filename}")
        self.logger.info(f"Summary saved: {summary_filename}")
        self.logger.info(f"Evaluation results organized in: {self.run_path}")

# Predefined intelligent configurations - Using only CONFIRMED WORKING models
LIGHTNING_EVAL = IntelligentEvaluationConfig(
    models=[
        "google/gemma-2-2b-it",           # CONFIRMED WORKING
        "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
        "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
    ],
    benchmarks=["gsm8k", "mmlu"],
    samples_per_eval=10,  # Lightning fast
    requests_per_minute=20,
    concurrent_models=1,
    progressive_sampling=False,
    output_dir="smaLLMs_results"
)

QUICK_EVAL = IntelligentEvaluationConfig(
    models=[
        "google/gemma-2-2b-it",           # CONFIRMED WORKING
        "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
        "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
        "Qwen/Qwen2.5-3B-Instruct",      # CONFIRMED WORKING
        "meta-llama/Llama-3.2-3B-Instruct", # CONFIRMED WORKING
    ],
    benchmarks=["gsm8k", "mmlu"],
    samples_per_eval=25,
    requests_per_minute=20,
    concurrent_models=1,
    progressive_sampling=True,
    output_dir="smaLLMs_results"
)

STANDARD_EVAL = IntelligentEvaluationConfig(
    models=[
        "google/gemma-2-2b-it",           # CONFIRMED WORKING
        "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
        "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
        "Qwen/Qwen2.5-3B-Instruct",      # CONFIRMED WORKING
        "meta-llama/Llama-3.2-3B-Instruct", # CONFIRMED WORKING
        "Qwen/Qwen2.5-7B-Instruct",      # Should work (7B size)
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", # Should work (instruct)
        "microsoft/DialoGPT-medium",      # Alternative to problematic models
    ],
    benchmarks=["gsm8k", "mmlu", "math"],
    samples_per_eval=50,
    requests_per_minute=25,
    concurrent_models=1,
    progressive_sampling=True,
    output_dir="smaLLMs_results"
)

COMPREHENSIVE_EVAL = IntelligentEvaluationConfig(
    models=[
        "google/gemma-2-2b-it",           # CONFIRMED WORKING
        "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
        "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
        "Qwen/Qwen2.5-3B-Instruct",      # CONFIRMED WORKING
        "meta-llama/Llama-3.2-3B-Instruct", # CONFIRMED WORKING
        "Qwen/Qwen2.5-7B-Instruct",      # Should work (7B size)
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", # Should work (instruct)
        "google/gemma-2-9b-it",          # Confirmed working Gemma family
        "microsoft/DialoGPT-medium",      # Alternative to problematic models
        "mistralai/Mistral-7B-Instruct-v0.3", # Popular instruct model
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Small but reliable
        "microsoft/DialoGPT-large",       # Larger alternative
    ],
    benchmarks=["gsm8k", "mmlu", "math", "humaneval"],
    samples_per_eval=100,
    requests_per_minute=30,
    concurrent_models=1,
    progressive_sampling=True,
    output_dir="smaLLMs_results"
)

async def main():
    """Main entry point for intelligent evaluation."""
    print("smaLLMs Unified Evaluation Platform")
    print("=" * 60)
    print("Streamlined evaluation with only working models")
    print()
    print("Choose evaluation preset:")
    print("1. Lightning (3 models, 2 benchmarks, 10 samples) - ~2 min")
    print("2. Quick (5 models, 2 benchmarks, 25 samples) - ~8 min")  
    print("3. Standard (8 models, 3 benchmarks, 50 samples) - ~25 min")
    print("4. Comprehensive (12 models, 4 benchmarks, 100 samples) - ~60 min")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        config = LIGHTNING_EVAL
        estimated_cost = 0.03
    elif choice == "2":
        config = QUICK_EVAL
        estimated_cost = 0.08
    elif choice == "3":
        config = STANDARD_EVAL
        estimated_cost = 0.25
    elif choice == "4":
        config = COMPREHENSIVE_EVAL
        estimated_cost = 0.60
    else:
        print("Using Lightning configuration...")
        config = LIGHTNING_EVAL
        estimated_cost = 0.03
    
    print(f"\nConfiguration: {len(config.models)} models, {len(config.benchmarks)} benchmarks")
    print(f"Models: {', '.join(config.models[:3])}{'...' if len(config.models) > 3 else ''}")
    print(f"Benchmarks: {', '.join(config.benchmarks)}")
    print(f"Samples per evaluation: {config.samples_per_eval}")
    print(f"Rate limit: {config.requests_per_minute} requests/minute")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Results will be saved to: smaLLMs_results/")
    
    confirm = input("\nProceed with evaluation? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Run intelligent evaluation
    orchestrator = IntelligentEvaluationOrchestrator(config)
    results = await orchestrator.run_intelligent_evaluation()
    
    # Display final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    
    exec_summary = results['execution_summary']
    print(f"✅ Total Evaluations: {exec_summary['total_evaluations']}")
    print(f"💰 Total Cost: ${exec_summary['total_cost']:.4f}")
    print(f"⏱️  Duration: {exec_summary['total_time_minutes']:.1f} minutes")
    print(f"📊 Cost Efficiency: ${exec_summary['avg_cost_per_eval']:.6f} per evaluation")
    print(f"📁 Results saved to: smaLLMs_results/")
    
    if results['rankings']['cost_efficiency']:
        print(f"\n🏆 Most Cost Efficient: {results['rankings']['cost_efficiency'][0][0]}")
        print(f"🎯 Best Performance: {results['rankings']['performance'][0][0]}")
    
    print(f"\n💡 Run 'python simple_exporter.py' to create website export")

if __name__ == "__main__":
    asyncio.run(main())
