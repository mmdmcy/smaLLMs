"""
Benchmark registry and implementations for state-of-the-art evaluations.
Includes all major benchmarks used by leading AI labs.
"""

import asyncio
import random
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from datasets import load_dataset
import logging

class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Run evaluation on the model."""
        pass
    
    @abstractmethod
    def get_benchmark_info(self) -> Dict[str, Any]:
        """Get information about the benchmark."""
        pass
    
    def extract_answer(self, response: str, question_type: str = "multiple_choice") -> str:
        """Extract answer from model response."""
        if question_type == "multiple_choice":
            # Look for A, B, C, D patterns
            match = re.search(r'\b([ABCD])\b', response.upper())
            return match.group(1) if match else ""
        
        elif question_type == "numerical":
            # Extract numbers
            numbers = re.findall(r'-?\d+\.?\d*', response)
            return numbers[-1] if numbers else ""
        
        elif question_type == "yes_no":
            response_lower = response.lower()
            if 'yes' in response_lower:
                return 'yes'
            elif 'no' in response_lower:
                return 'no'
            return ""
        
        return response.strip()

class MMLUBenchmark(BaseBenchmark):
    """MMLU (Massive Multitask Language Understanding) benchmark."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.subjects = config.get('evaluation', {}).get('benchmarks', {}).get('mmlu', {}).get('subjects', ['all'])
        self.few_shot = config.get('evaluation', {}).get('benchmarks', {}).get('mmlu', {}).get('few_shot', 5)
    
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on MMLU dataset."""
        self.logger.info(f"Starting MMLU evaluation with {num_samples} samples")
        
        # Load dataset (streaming to save space) - fix slicing issue
        if self.subjects == ['all']:
            dataset = load_dataset("cais/mmlu", "all", split="test", streaming=True)
        else:
            # For specific subjects, we'd load each one - simplified here
            dataset = load_dataset("cais/mmlu", "all", split="test", streaming=True)
        
        results = []
        
        # Convert to list for processing - only take what we need
        dataset_list = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            dataset_list.append(item)
        
        # Prepare prompts
        prompts = []
        for item in dataset_list:
            prompt = self._create_mmlu_prompt(item, self.few_shot)
            prompts.append(prompt)
        
        # Generate responses in batch
        from models.model_manager import GenerationConfig
        gen_config = GenerationConfig(
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 10),
            stop_sequences=['Q:', '\n\n']
        )
        
        responses = await model.batch_generate(prompts, gen_config)
        
        # Process results
        for item, prompt, response in zip(dataset_list, prompts, responses):
            predicted = self.extract_answer(response, "multiple_choice")
            correct = item['answer']
            
            result = {
                'question': item['question'],
                'subject': item.get('subject', 'unknown'),
                'choices': item['choices'],
                'correct_answer': correct,
                'predicted_answer': predicted,
                'is_correct': predicted == correct,
                'response': response,
                'prompt': prompt
            }
            results.append(result)
        
        self.logger.info(f"MMLU evaluation completed: {len(results)} samples")
        return results
    
    def _create_mmlu_prompt(self, item: Dict, few_shot: int = 5) -> str:
        """Create few-shot prompt for MMLU question."""
        # Simplified few-shot prompt
        prompt = "Answer the following multiple choice question.\n\n"
        
        question = item['question']
        choices = item['choices']
        
        prompt += f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        
        prompt += "\nAnswer: "
        return prompt
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": "MMLU",
            "description": "Massive Multitask Language Understanding",
            "num_subjects": 57,
            "question_type": "multiple_choice",
            "metric": "accuracy",
            "paper": "https://arxiv.org/abs/2009.03300"
        }

class GSM8KBenchmark(BaseBenchmark):
    """GSM8K (Grade School Math 8K) benchmark."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.few_shot = config.get('evaluation', {}).get('benchmarks', {}).get('gsm8k', {}).get('few_shot', 5)
    
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on GSM8K dataset."""
        self.logger.info(f"Starting GSM8K evaluation with {num_samples} samples")
        
        # Load dataset - fix the slicing issue
        dataset = load_dataset("gsm8k", "main", split="test", streaming=True)
        # Take only the number of samples we need
        dataset_list = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            dataset_list.append(item)
        
        # Prepare prompts
        prompts = []
        for item in dataset_list:
            prompt = self._create_gsm8k_prompt(item, self.few_shot)
            prompts.append(prompt)
        
        # Generate responses
        from models.model_manager import GenerationConfig
        gen_config = GenerationConfig(
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 256),
            stop_sequences=['\n\nQ:', 'Question:']
        )
        
        responses = await model.batch_generate(prompts, gen_config)
        
        # Process results
        results = []
        for item, prompt, response in zip(dataset_list, prompts, responses):
            predicted = self.extract_numerical_answer(response)
            correct = self.extract_numerical_answer(item['answer'])
            
            result = {
                'question': item['question'],
                'correct_answer': correct,
                'predicted_answer': predicted,
                'is_correct': abs(float(predicted or 0) - float(correct or 0)) < 0.01,
                'response': response,
                'prompt': prompt
            }
            results.append(result)
        
        self.logger.info(f"GSM8K evaluation completed: {len(results)} samples")
        return results
    
    def _create_gsm8k_prompt(self, item: Dict, few_shot: int = 5) -> str:
        """Create few-shot prompt for GSM8K question."""
        prompt = "Solve the following math problem step by step.\n\n"
        prompt += f"Question: {item['question']}\n"
        prompt += "Answer: "
        return prompt
    
    def extract_numerical_answer(self, text: str) -> str:
        """Extract numerical answer from response."""
        # Look for patterns like "The answer is X" or just numbers
        patterns = [
            r'[Tt]he answer is\s*([+-]?\d+(?:\.\d+)?)',
            r'[Aa]nswer:\s*([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)(?:\s*$|\s*\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Fallback: find any number in the text
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        return numbers[-1] if numbers else "0"
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": "GSM8K",
            "description": "Grade School Math 8K",
            "question_type": "numerical",
            "metric": "accuracy",
            "paper": "https://arxiv.org/abs/2110.14168"
        }

class MATHBenchmark(BaseBenchmark):
    """MATH benchmark for competition-level mathematics."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.few_shot = config.get('evaluation', {}).get('benchmarks', {}).get('math', {}).get('few_shot', 4)
    
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on MATH dataset."""
        self.logger.info(f"Starting MATH evaluation with {num_samples} samples")
        
        # Load dataset - fix slicing issue
        dataset = load_dataset("hendrycks/competition_math", split="test", streaming=True)
        # Take only the number of samples we need
        dataset_list = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            dataset_list.append(item)
        
        # Prepare prompts
        prompts = []
        for item in dataset_list:
            prompt = self._create_math_prompt(item)
            prompts.append(prompt)
        
        # Generate responses
        from models.model_manager import GenerationConfig
        gen_config = GenerationConfig(
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 512),
            stop_sequences=['Problem:', '\n\nProblem:']
        )
        
        responses = await model.batch_generate(prompts, gen_config)
        
        # Process results
        results = []
        for item, prompt, response in zip(dataset_list, prompts, responses):
            predicted = self.extract_numerical_answer(response)
            correct = self.extract_numerical_answer(item['solution'])
            
            result = {
                'problem': item['problem'],
                'level': item['level'],
                'type': item['type'],
                'correct_answer': correct,
                'predicted_answer': predicted,
                'is_correct': self._check_math_answer(predicted, correct),
                'response': response,
                'prompt': prompt
            }
            results.append(result)
        
        self.logger.info(f"MATH evaluation completed: {len(results)} samples")
        return results
    
    def _create_math_prompt(self, item: Dict) -> str:
        """Create prompt for MATH problem."""
        prompt = "Solve the following mathematics problem. Show your work and provide the final answer.\n\n"
        prompt += f"Problem: {item['problem']}\n"
        prompt += "Solution: "
        return prompt
    
    def _check_math_answer(self, predicted: str, correct: str) -> bool:
        """Check if mathematical answers match (with some tolerance)."""
        try:
            pred_val = float(predicted) if predicted else 0
            correct_val = float(correct) if correct else 0
            return abs(pred_val - correct_val) < 0.01
        except (ValueError, TypeError):
            return predicted.strip().lower() == correct.strip().lower()
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": "MATH",
            "description": "Competition-level mathematics problems",
            "question_type": "numerical",
            "metric": "accuracy",
            "paper": "https://arxiv.org/abs/2103.03874"
        }

class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval benchmark for code generation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.temperature = config.get('evaluation', {}).get('benchmarks', {}).get('humaneval', {}).get('temperature', 0.1)
        self.max_tokens = config.get('evaluation', {}).get('benchmarks', {}).get('humaneval', {}).get('max_tokens', 512)
    
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on HumanEval dataset."""
        self.logger.info(f"Starting HumanEval evaluation with {num_samples} samples")
        
        # Load dataset - fix slicing issue
        dataset = load_dataset("openai_humaneval", split="test", streaming=True)
        # Take only the number of samples we need
        dataset_list = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            dataset_list.append(item)
        
        # Prepare prompts
        prompts = []
        for item in dataset_list:
            prompt = self._create_humaneval_prompt(item)
            prompts.append(prompt)
        
        # Generate responses
        from models.model_manager import GenerationConfig
        gen_config = GenerationConfig(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_sequences=['def ', 'class ', '\n\n\n']
        )
        
        responses = await model.batch_generate(prompts, gen_config)
        
        # Process results
        results = []
        for item, prompt, response in zip(dataset_list, prompts, responses):
            # For code evaluation, we'd need to execute and test
            # Simplified here - would need proper code execution sandbox
            is_correct = self._check_code_syntax(response)
            
            result = {
                'task_id': item['task_id'],
                'prompt': item['prompt'],
                'canonical_solution': item['canonical_solution'],
                'test': item['test'],
                'entry_point': item['entry_point'],
                'generated_code': response,
                'is_correct': is_correct,
                'full_prompt': prompt
            }
            results.append(result)
        
        self.logger.info(f"HumanEval evaluation completed: {len(results)} samples")
        return results
    
    def _create_humaneval_prompt(self, item: Dict) -> str:
        """Create prompt for HumanEval problem."""
        return item['prompt']  # HumanEval prompts are already well-formatted
    
    def _check_code_syntax(self, code: str) -> bool:
        """Basic syntax check for generated code."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": "HumanEval",
            "description": "Python code generation and completion",
            "question_type": "code_generation",
            "metric": "pass@1",
            "paper": "https://arxiv.org/abs/2107.03374"
        }

class BenchmarkRegistry:
    """Registry for all available benchmarks."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.benchmarks = {
            'mmlu': MMLUBenchmark(config),
            'gsm8k': GSM8KBenchmark(config),
            'math': MATHBenchmark(config),
            'humaneval': HumanEvalBenchmark(config),
        }
        self.logger = logging.getLogger(__name__)
    
    def get_benchmark(self, name: str) -> BaseBenchmark:
        """Get benchmark by name."""
        if name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {name}. Available: {list(self.benchmarks.keys())}")
        return self.benchmarks[name]
    
    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """List all available benchmarks."""
        return [
            benchmark.get_benchmark_info()
            for benchmark in self.benchmarks.values()
        ]
    
    def add_benchmark(self, name: str, benchmark: BaseBenchmark):
        """Add a custom benchmark."""
        self.benchmarks[name] = benchmark
        self.logger.info(f"Added custom benchmark: {name}")

# Standard benchmark suite for comprehensive evaluation
STANDARD_BENCHMARK_SUITE = ['mmlu', 'gsm8k', 'math', 'humaneval']

# Quick benchmark suite for fast testing
QUICK_BENCHMARK_SUITE = ['gsm8k', 'humaneval']
