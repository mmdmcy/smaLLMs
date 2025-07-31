"""
Advanced metrics and result aggregation for smaLLMs platform.
Provides comprehensive analysis including enterprise-grade metrics.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats

@dataclass
class ModelPerformance:
    """Comprehensive performance metrics for a model."""
    model_name: str
    overall_score: float
    benchmark_scores: Dict[str, float]
    cost_efficiency: float
    avg_latency: float
    reliability_score: float
    safety_score: float
    timestamp: str

@dataclass
class BenchmarkStats:
    """Statistical analysis for a benchmark."""
    benchmark_name: str
    mean_score: float
    std_dev: float
    median_score: float
    percentile_25: float
    percentile_75: float
    num_models: int
    top_model: str
    top_score: float

class ResultAggregator:
    """Advanced result aggregation and analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Benchmark weights for overall score calculation
        self.benchmark_weights = {
            'mmlu': 0.3,      # General knowledge and reasoning
            'gsm8k': 0.25,    # Mathematical reasoning
            'math': 0.2,      # Advanced mathematics
            'humaneval': 0.25  # Code generation
        }
    
    def calculate_accuracy(self, results: List[Dict[str, Any]]) -> float:
        """Calculate accuracy from evaluation results."""
        if not results:
            return 0.0
        
        correct_count = sum(1 for result in results if result.get('is_correct', False))
        total_count = len(results)
        
        return correct_count / total_count if total_count > 0 else 0.0
    
    def calculate_comprehensive_metrics(self, results: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics beyond basic accuracy."""
        if not results:
            return {}
        
        # Basic accuracy
        accuracy = self.calculate_accuracy(results)
        
        # Response quality metrics
        avg_response_length = np.mean([len(r.get('response', '')) for r in results])
        
        # Consistency analysis (how often the model gives the same answer for similar questions)
        consistency_score = self._calculate_consistency(results)
        
        # Answer distribution analysis
        answer_distribution = self._analyze_answer_distribution(results)
        
        return {
            'accuracy': accuracy,
            'avg_response_length': avg_response_length,
            'consistency_score': consistency_score,
            'answer_distribution': answer_distribution,
            'total_samples': len(results),
            'error_rate': 1 - accuracy
        }
    
    def _calculate_consistency(self, results: List[Dict[str, Any]]) -> float:
        """Calculate how consistent the model's answers are."""
        # Simplified consistency metric
        # In practice, this would involve more sophisticated analysis
        predicted_answers = [r.get('predicted_answer', '') for r in results]
        
        if not predicted_answers:
            return 0.0
        
        # Calculate variance in answer patterns
        answer_lengths = [len(str(ans)) for ans in predicted_answers]
        
        if len(set(answer_lengths)) == 1:
            return 1.0  # Perfect consistency in length
        
        return 1.0 - (np.std(answer_lengths) / np.mean(answer_lengths)) if np.mean(answer_lengths) > 0 else 0.0
    
    def _analyze_answer_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of answers."""
        predicted_answers = [r.get('predicted_answer', '') for r in results]
        
        # Count answer patterns
        answer_counts = {}
        for answer in predicted_answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Calculate statistics
        total_answers = len(predicted_answers)
        most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ('', 0)
        
        return {
            'unique_answers': len(answer_counts),
            'most_common_answer': most_common[0],
            'most_common_frequency': most_common[1] / total_answers if total_answers > 0 else 0,
            'answer_diversity': len(answer_counts) / total_answers if total_answers > 0 else 0
        }
    
    def estimate_cost(self, results: List[Dict[str, Any]], model_name: str) -> float:
        """Estimate the cost of running the evaluation."""
        # Simplified cost estimation based on token usage
        total_prompt_tokens = sum(len(r.get('prompt', '').split()) for r in results)
        total_response_tokens = sum(len(r.get('response', '').split()) for r in results)
        
        # Cost per token estimates (would be loaded from model info)
        cost_per_token = self._get_model_cost_per_token(model_name)
        
        total_cost = (total_prompt_tokens + total_response_tokens) * cost_per_token
        return round(total_cost, 6)
    
    def _get_model_cost_per_token(self, model_name: str) -> float:
        """Get cost per token for a model."""
        # Simplified cost mapping - in practice would come from model manager
        cost_mapping = {
            'small': 0.0001,   # < 3B models
            'medium': 0.0002,  # 3B-7B models
            'large': 0.0005,   # 7B+ models
        }
        
        # Determine size category from model name
        if any(size in model_name.lower() for size in ['1b', '2b', '3b']):
            return cost_mapping['small']
        elif any(size in model_name.lower() for size in ['7b', '8b', '9b']):
            return cost_mapping['medium']
        else:
            return cost_mapping['large']
    
    def calculate_overall_score(self, benchmark_results: Dict[str, float]) -> float:
        """Calculate weighted overall score across benchmarks."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for benchmark, score in benchmark_results.items():
            weight = self.benchmark_weights.get(benchmark, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def generate_leaderboard(self, all_results: List[Any]) -> Dict[str, Any]:
        """Generate comprehensive leaderboard from all evaluation results."""
        # Group results by model
        model_results = {}
        for result in all_results:
            model_name = result.model_name
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)
        
        # Calculate performance for each model
        model_performances = []
        for model_name, results in model_results.items():
            benchmark_scores = {}
            total_latency = 0
            total_cost = 0
            
            # Group by benchmark
            benchmark_groups = {}
            for result in results:
                benchmark = result.benchmark_name
                if benchmark not in benchmark_groups:
                    benchmark_groups[benchmark] = []
                benchmark_groups[benchmark].append(result)
            
            # Calculate scores per benchmark
            for benchmark, bench_results in benchmark_groups.items():
                if bench_results:
                    avg_accuracy = np.mean([r.accuracy for r in bench_results])
                    benchmark_scores[benchmark] = avg_accuracy
                    total_latency += sum(r.latency for r in bench_results)
                    total_cost += sum(r.cost_estimate for r in bench_results)
            
            # Calculate overall metrics
            overall_score = self.calculate_overall_score(benchmark_scores)
            avg_latency = total_latency / len(results) if results else 0
            cost_efficiency = overall_score / (total_cost + 0.001)  # Avoid division by zero
            
            performance = ModelPerformance(
                model_name=model_name,
                overall_score=overall_score,
                benchmark_scores=benchmark_scores,
                cost_efficiency=cost_efficiency,
                avg_latency=avg_latency,
                reliability_score=self._calculate_reliability(results),
                safety_score=1.0,  # Placeholder for safety evaluation
                timestamp=datetime.now().isoformat()
            )
            model_performances.append(performance)
        
        # Sort by overall score
        model_performances.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Generate benchmark statistics
        benchmark_stats = self._generate_benchmark_stats(model_performances)
        
        return {
            'leaderboard': [self._performance_to_dict(perf) for perf in model_performances],
            'benchmark_stats': benchmark_stats,
            'last_updated': datetime.now().isoformat(),
            'total_models': len(model_performances),
            'total_evaluations': len(all_results)
        }
    
    def _calculate_reliability(self, results: List[Any]) -> float:
        """Calculate reliability score based on error rates and consistency."""
        if not results:
            return 0.0
        
        # Calculate success rate (no errors)
        success_rate = sum(1 for r in results if not r.error) / len(results)
        
        # Calculate latency consistency
        latencies = [r.latency for r in results if not r.error]
        if latencies:
            latency_cv = np.std(latencies) / np.mean(latencies) if np.mean(latencies) > 0 else 1
            latency_consistency = max(0, 1 - latency_cv)
        else:
            latency_consistency = 0
        
        # Combined reliability score
        return (success_rate * 0.7 + latency_consistency * 0.3)
    
    def _generate_benchmark_stats(self, performances: List[ModelPerformance]) -> Dict[str, BenchmarkStats]:
        """Generate statistical analysis for each benchmark."""
        benchmark_stats = {}
        
        # Get all unique benchmarks
        all_benchmarks = set()
        for perf in performances:
            all_benchmarks.update(perf.benchmark_scores.keys())
        
        for benchmark in all_benchmarks:
            scores = []
            models_with_scores = []
            
            for perf in performances:
                if benchmark in perf.benchmark_scores:
                    scores.append(perf.benchmark_scores[benchmark])
                    models_with_scores.append((perf.model_name, perf.benchmark_scores[benchmark]))
            
            if scores:
                # Calculate statistics
                scores_array = np.array(scores)
                
                # Find top model
                top_model_data = max(models_with_scores, key=lambda x: x[1])
                
                stats_obj = BenchmarkStats(
                    benchmark_name=benchmark,
                    mean_score=float(np.mean(scores_array)),
                    std_dev=float(np.std(scores_array)),
                    median_score=float(np.median(scores_array)),
                    percentile_25=float(np.percentile(scores_array, 25)),
                    percentile_75=float(np.percentile(scores_array, 75)),
                    num_models=len(scores),
                    top_model=top_model_data[0],
                    top_score=float(top_model_data[1])
                )
                
                benchmark_stats[benchmark] = stats_obj
        
        return {name: self._benchmark_stats_to_dict(stats) for name, stats in benchmark_stats.items()}
    
    def _performance_to_dict(self, performance: ModelPerformance) -> Dict[str, Any]:
        """Convert ModelPerformance to dictionary."""
        return {
            'model_name': performance.model_name,
            'overall_score': round(performance.overall_score, 4),
            'benchmark_scores': {k: round(v, 4) for k, v in performance.benchmark_scores.items()},
            'cost_efficiency': round(performance.cost_efficiency, 4),
            'avg_latency': round(performance.avg_latency, 2),
            'reliability_score': round(performance.reliability_score, 4),
            'safety_score': round(performance.safety_score, 4),
            'timestamp': performance.timestamp
        }
    
    def _benchmark_stats_to_dict(self, stats: BenchmarkStats) -> Dict[str, Any]:
        """Convert BenchmarkStats to dictionary."""
        return {
            'benchmark_name': stats.benchmark_name,
            'mean_score': round(stats.mean_score, 4),
            'std_dev': round(stats.std_dev, 4),
            'median_score': round(stats.median_score, 4),
            'percentile_25': round(stats.percentile_25, 4),
            'percentile_75': round(stats.percentile_75, 4),
            'num_models': stats.num_models,
            'top_model': stats.top_model,
            'top_score': round(stats.top_score, 4)
        }
    
    def compare_models(self, model_results: Dict[str, List[Any]], benchmark: str = None) -> Dict[str, Any]:
        """Compare multiple models across benchmarks."""
        comparison = {}
        
        for model_name, results in model_results.items():
            if benchmark:
                # Filter results for specific benchmark
                bench_results = [r for r in results if r.benchmark_name == benchmark]
            else:
                bench_results = results
            
            if bench_results:
                metrics = self.calculate_comprehensive_metrics(bench_results, model_name)
                comparison[model_name] = metrics
        
        # Add relative rankings
        if comparison:
            # Rank by accuracy
            sorted_models = sorted(comparison.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
            for i, (model_name, metrics) in enumerate(sorted_models):
                comparison[model_name]['rank'] = i + 1
                comparison[model_name]['percentile'] = (len(sorted_models) - i) / len(sorted_models) * 100
        
        return comparison
    
    def generate_model_report(self, model_name: str, results: List[Any]) -> Dict[str, Any]:
        """Generate comprehensive report for a single model."""
        if not results:
            return {'error': 'No results found for model'}
        
        # Group by benchmark
        benchmark_results = {}
        for result in results:
            benchmark = result.benchmark_name
            if benchmark not in benchmark_results:
                benchmark_results[benchmark] = []
            benchmark_results[benchmark].append(result)
        
        # Calculate metrics per benchmark
        benchmark_metrics = {}
        for benchmark, bench_results in benchmark_results.items():
            metrics = self.calculate_comprehensive_metrics(bench_results, model_name)
            benchmark_metrics[benchmark] = metrics
        
        # Overall statistics
        overall_accuracy = np.mean([metrics['accuracy'] for metrics in benchmark_metrics.values()])
        total_cost = sum(r.cost_estimate for r in results)
        avg_latency = np.mean([r.latency for r in results])
        
        return {
            'model_name': model_name,
            'overall_accuracy': round(overall_accuracy, 4),
            'total_cost': round(total_cost, 6),
            'avg_latency': round(avg_latency, 2),
            'benchmark_metrics': benchmark_metrics,
            'total_evaluations': len(results),
            'evaluation_period': {
                'start': min(r.timestamp for r in results),
                'end': max(r.timestamp for r in results)
            },
            'generated_at': datetime.now().isoformat()
        }
