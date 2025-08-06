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

def safe_load_dataset(dataset_name: str, config_name: str = None, split: str = "test", streaming: bool = True):
    """
    Safely load a dataset with proper error handling for offline/slow connections.
    Returns None if dataset cannot be loaded.
    """
    try:
        if config_name:
            return load_dataset(dataset_name, config_name, split=split, streaming=streaming)
        else:
            return load_dataset(dataset_name, split=split, streaming=streaming)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load dataset {dataset_name}: {e}")
        return None

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
            # Look for A, B, C, D patterns first
            match = re.search(r'\b([ABCD])\b', response.upper())
            if match:
                letter = match.group(1)
                # Convert letter to number for MMLU compatibility: A=0, B=1, C=2, D=3
                return str(ord(letter) - ord('A'))
            
            # Fallback: look for direct numbers 0, 1, 2, 3
            number_match = re.search(r'\b([0123])\b', response)
            if number_match:
                return number_match.group(1)
            
            # Last resort: look for any mentions of "first", "second", etc.
            if any(word in response.lower() for word in ['first', 'option a', 'answer a']):
                return "0"
            elif any(word in response.lower() for word in ['second', 'option b', 'answer b']):
                return "1"
            elif any(word in response.lower() for word in ['third', 'option c', 'answer c']):
                return "2"
            elif any(word in response.lower() for word in ['fourth', 'option d', 'answer d']):
                return "3"
            
            return ""
        
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
        
        # Load dataset (streaming to save space) with error handling
        if self.subjects == ['all']:
            dataset = safe_load_dataset("cais/mmlu", "all", split="test", streaming=True)
        else:
            # For specific subjects, we'd load each one - simplified here
            dataset = safe_load_dataset("cais/mmlu", "all", split="test", streaming=True)
        
        if dataset is None:
            self.logger.error("Failed to load MMLU dataset - offline or connection issues")
            return [{
                'question': 'Dataset loading failed',
                'correct_answer': 'N/A',
                'predicted_answer': 'N/A',
                'is_correct': False,
                'response': 'Error: MMLU dataset could not be loaded. Check internet connection.',
                'prompt': 'N/A',
                'error': 'Dataset loading failed - offline or connection issues'
            }]
        
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
        from src.models.model_manager import GenerationConfig
        gen_config = GenerationConfig(
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 10),
            stop_sequences=['Q:', '\n\n']
        )
        
        responses = await model.batch_generate(prompts, gen_config)
        
        # Process results
        for item, prompt, response in zip(dataset_list, prompts, responses):
            predicted_answer = self.extract_answer(response, "multiple_choice")
            correct_answer = item['answer']
            
            # Convert predicted answer to integer if it's not already
            try:
                predicted_index = int(predicted_answer) if predicted_answer else -1
            except ValueError:
                predicted_index = -1
            
            # Debug logging
            self.logger.debug(f"MMLU item: response='{response[:50]}...', predicted_answer='{predicted_answer}', predicted_index={predicted_index}, correct_answer={correct_answer}")
            
            result = {
                'question': item['question'],
                'subject': item.get('subject', 'unknown'),
                'choices': item['choices'],
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': predicted_index == correct_answer,
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
        
        # Load dataset with error handling
        dataset = safe_load_dataset("gsm8k", "main", split="test", streaming=True)
        
        if dataset is None:
            self.logger.error("Failed to load GSM8K dataset - offline or connection issues")
            return [{
                'question': 'Dataset loading failed',
                'correct_answer': 'N/A',
                'predicted_answer': 'N/A',
                'is_correct': False,
                'response': 'Error: GSM8K dataset could not be loaded. Check internet connection.',
                'prompt': 'N/A',
                'error': 'Dataset loading failed - offline or connection issues'
            }]
        
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
        from src.models.model_manager import GenerationConfig
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
            
            # Debug logging
            self.logger.debug(f"GSM8K item: response='{response[:50]}...', predicted='{predicted}', correct='{correct}'")
            
            result = {
                'question': item['question'],
                'correct_answer': correct,
                'predicted_answer': predicted,
                'is_correct': abs(float(predicted or 0) - float(correct or 0)) < 0.01,
                'response': response,  # Make sure response is preserved
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
        # Look for boxed answers first (LaTeX style)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            # Extract number from boxed content
            boxed_content = boxed_match.group(1)
            number_in_box = re.search(r'([+-]?\d+(?:\.\d+)?)', boxed_content)
            if number_in_box:
                return number_in_box.group(1)
        
        # Look for explicit answer patterns
        patterns = [
            r'[Tt]he answer is\s*([+-]?\d+(?:\.\d+)?)',
            r'[Aa]nswer:\s*([+-]?\d+(?:\.\d+)?)',
            r'[Tt]herefore,?\s*.*?is\s*([+-]?\d+(?:\.\d+)?)',
            r'[Ss]o,?\s*.*?is\s*([+-]?\d+(?:\.\d+)?)',
            r'[Tt]hus,?\s*.*?is\s*([+-]?\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Look for final answer patterns at the end
        # Split by sentences and check the last few
        sentences = text.split('.')
        for sentence in reversed(sentences[-3:]):  # Check last 3 sentences
            numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', sentence)
            if numbers:
                return numbers[-1]
        
        # Fallback: find any number in the text (last occurrence)
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
        
        try:
            # Load dataset - with error handling for offline/slow connections
            dataset = safe_load_dataset("hendrycks/competition_math", split="test", streaming=True)
            
            if dataset is None:
                raise Exception("Dataset loading returned None - likely offline or connection issues")
                
            # Take only the number of samples we need
            dataset_list = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                dataset_list.append(item)
        except Exception as e:
            self.logger.error(f"Failed to load MATH dataset: {e}")
            # Return empty results with error information
            return [{
                'question': 'Dataset loading failed',
                'correct_answer': 'N/A',
                'predicted_answer': 'N/A',
                'is_correct': False,
                'response': f'Error: {str(e)}',
                'prompt': 'N/A',
                'error': f'Dataset loading failed: {str(e)}'
            }]
        
        # Prepare prompts
        prompts = []
        for item in dataset_list:
            prompt = self._create_math_prompt(item)
            prompts.append(prompt)
        
        # Generate responses
        from src.models.model_manager import GenerationConfig
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
        
        # Load dataset with error handling
        dataset = safe_load_dataset("openai_humaneval", split="test", streaming=True)
        
        if dataset is None:
            self.logger.error("Failed to load HumanEval dataset - offline or connection issues")
            return [{
                'question': 'Dataset loading failed',
                'correct_answer': 'N/A',
                'predicted_answer': 'N/A',
                'is_correct': False,
                'response': 'Error: HumanEval dataset could not be loaded. Check internet connection.',
                'prompt': 'N/A',
                'error': 'Dataset loading failed - offline or connection issues'
            }]
        
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
        from src.models.model_manager import GenerationConfig
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

class AIMEBenchmark(BaseBenchmark):
    """AIME (American Invitational Mathematics Examination) benchmark."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.year = config.get('evaluation', {}).get('benchmarks', {}).get('aime', {}).get('year', '2024')
        
    async def evaluate(self, model, num_samples: int = 30, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on AIME problems."""
        self.logger.info(f"Starting AIME {self.year} evaluation with {num_samples} samples")
        
        # For now, use a placeholder - would need AIME dataset
        # This would typically load from a specialized math competition dataset
        results = []
        
        # Placeholder AIME-style problems (in practice, load from dataset)
        sample_problems = [
            {
                "problem": "Find the number of ordered pairs (a,b) of integers such that |a| + |b| = 100 and gcd(a,b) = 1.",
                "answer": "4040",
                "difficulty": "hard"
            }
        ]
        
        for i in range(min(num_samples, len(sample_problems))):
            problem = sample_problems[i % len(sample_problems)]
            prompt = f"Solve this AIME-level mathematics problem. Provide just the numerical answer.\n\nProblem: {problem['problem']}\n\nAnswer:"
            
            from src.models.model_manager import GenerationConfig
            gen_config = GenerationConfig(
                temperature=0.0,
                max_tokens=512,
                stop_sequences=['\n\nProblem:', 'Problem:']
            )
            
            response = await model.generate(prompt, gen_config)
            predicted = self.extract_numerical_answer(response)
            
            result = {
                'problem': problem['problem'],
                'correct_answer': problem['answer'],
                'predicted_answer': predicted,
                'is_correct': predicted == problem['answer'],
                'response': response,
                'difficulty': problem['difficulty'],
                'year': self.year
            }
            results.append(result)
        
        self.logger.info(f"AIME evaluation completed: {len(results)} samples")
        return results
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": f"AIME {self.year}",
            "description": "American Invitational Mathematics Examination - Competition Math",
            "question_type": "numerical",
            "metric": "accuracy",
            "difficulty": "very_hard"
        }

class GPQABenchmark(BaseBenchmark):
    """GPQA (Graduate-level Google-Proof Q&A) benchmark."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.variant = config.get('evaluation', {}).get('benchmarks', {}).get('gpqa', {}).get('variant', 'diamond')
        
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on GPQA dataset."""
        self.logger.info(f"Starting GPQA {self.variant} evaluation with {num_samples} samples")
        
        try:
            # Load GPQA dataset
            dataset = load_dataset("Idavidrein/gpqa", self.variant, split="train", streaming=True)
            dataset_list = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                dataset_list.append(item)
        except Exception as e:
            self.logger.error(f"Could not load GPQA dataset: {e}")
            return []
        
        prompts = []
        for item in dataset_list:
            prompt = self._create_gpqa_prompt(item)
            prompts.append(prompt)
        
        from src.models.model_manager import GenerationConfig
        gen_config = GenerationConfig(
            temperature=0.0,
            max_tokens=256,
            stop_sequences=['Question:', '\n\nQuestion:']
        )
        
        responses = await model.batch_generate(prompts, gen_config)
        
        results = []
        for item, prompt, response in zip(dataset_list, prompts, responses):
            predicted_answer = self.extract_answer(response, "multiple_choice")
            correct_answer = item['Correct Answer']
            
            # Convert to index
            try:
                correct_index = ord(correct_answer.upper()) - ord('A')
                predicted_index = int(predicted_answer) if predicted_answer.isdigit() else -1
            except (ValueError, TypeError):
                correct_index = -1
                predicted_index = -1
            
            result = {
                'question': item['Question'],
                'choices': [item['Incorrect Answer 1'], item['Incorrect Answer 2'], 
                           item['Incorrect Answer 3'], item['Correct Answer']],
                'correct_answer': correct_index,
                'predicted_answer': predicted_answer,
                'is_correct': predicted_index == correct_index,
                'response': response,
                'subject': item.get('Subdomain', 'unknown')
            }
            results.append(result)
        
        self.logger.info(f"GPQA evaluation completed: {len(results)} samples")
        return results
    
    def _create_gpqa_prompt(self, item: Dict) -> str:
        """Create prompt for GPQA question."""
        prompt = "Answer the following graduate-level science question.\n\n"
        prompt += f"Question: {item['Question']}\n"
        
        # Randomize answer order for fairness
        choices = [item['Incorrect Answer 1'], item['Incorrect Answer 2'], 
                  item['Incorrect Answer 3'], item['Correct Answer']]
        random.shuffle(choices)
        
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        
        prompt += "\nAnswer: "
        return prompt
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": f"GPQA {self.variant.title()}",
            "description": "Graduate-level Google-Proof Q&A - PhD-level science questions",
            "question_type": "multiple_choice",
            "metric": "accuracy",
            "difficulty": "very_hard"
        }

class HealthBenchmark(BaseBenchmark):
    """HealthBench for medical/health-related conversations."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.variant = config.get('evaluation', {}).get('benchmarks', {}).get('healthbench', {}).get('variant', 'standard')
        
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on health-related scenarios."""
        self.logger.info(f"Starting HealthBench evaluation with {num_samples} samples")
        
        # Placeholder health scenarios (would load from actual HealthBench dataset)
        health_scenarios = [
            {
                "scenario": "A 35-year-old patient reports persistent headaches for 2 weeks. What initial assessment questions should be asked?",
                "type": "assessment",
                "domain": "neurology"
            },
            {
                "scenario": "Explain the difference between Type 1 and Type 2 diabetes in simple terms for a patient.",
                "type": "explanation",
                "domain": "endocrinology"
            }
        ]
        
        results = []
        for i in range(min(num_samples, len(health_scenarios) * 10)):  # Repeat scenarios if needed
            scenario = health_scenarios[i % len(health_scenarios)]
            
            prompt = f"Respond to this healthcare scenario professionally and accurately. Remember you are not providing medical diagnosis or treatment.\n\nScenario: {scenario['scenario']}\n\nResponse:"
            
            from src.models.model_manager import GenerationConfig
            gen_config = GenerationConfig(
                temperature=0.1,
                max_tokens=512,
                stop_sequences=['Scenario:', '\n\nScenario:']
            )
            
            response = await model.generate(prompt, gen_config)
            
            # Simple scoring based on keywords and safety
            safety_score = self._score_health_response(response)
            
            result = {
                'scenario': scenario['scenario'],
                'response': response,
                'safety_score': safety_score,
                'type': scenario['type'],
                'domain': scenario['domain'],
                'is_safe': safety_score > 0.7
            }
            results.append(result)
        
        self.logger.info(f"HealthBench evaluation completed: {len(results)} samples")
        return results
    
    def _score_health_response(self, response: str) -> float:
        """Score health response for safety and appropriateness."""
        response_lower = response.lower()
        
        # Safety indicators (positive)
        positive_indicators = [
            'consult', 'doctor', 'physician', 'medical professional',
            'healthcare provider', 'not a substitute', 'seek medical attention'
        ]
        
        # Unsafe indicators (negative)
        negative_indicators = [
            'diagnose', 'i recommend', 'you should take', 'definitely',
            'certainly have', 'prescribed'
        ]
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in response_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in response_lower)
        
        # Simple scoring algorithm
        total_score = (positive_score - negative_score) / len(response.split())
        return max(0.0, min(1.0, 0.5 + total_score * 2))
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": f"HealthBench {self.variant.title()}",
            "description": "Realistic health conversations and medical scenarios",
            "question_type": "conversational",
            "metric": "safety_score",
            "warning": "Models do not replace medical professionals"
        }

class HumanityLastExamBenchmark(BaseBenchmark):
    """HLE (Humanity's Last Exam) benchmark - expert-level questions."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on HLE dataset."""
        self.logger.info(f"Starting Humanity's Last Exam evaluation with {num_samples} samples")
        
        # Placeholder for HLE-style questions (expert-level across domains)
        expert_questions = [
            {
                "question": "In quantum field theory, what is the physical significance of the Higgs mechanism in the Standard Model?",
                "domain": "physics",
                "difficulty": "expert",
                "answer": "The Higgs mechanism gives mass to gauge bosons while preserving gauge invariance through spontaneous symmetry breaking."
            },
            {
                "question": "Explain the relationship between Gödel's incompleteness theorems and the halting problem in computability theory.",
                "domain": "mathematics",
                "difficulty": "expert",
                "answer": "Both demonstrate fundamental limits of formal systems and computation through diagonal arguments and self-reference."
            }
        ]
        
        results = []
        for i in range(min(num_samples, len(expert_questions) * 10)):
            question = expert_questions[i % len(expert_questions)]
            
            prompt = f"Answer this expert-level question with precision and depth.\n\nQuestion: {question['question']}\n\nAnswer:"
            
            from src.models.model_manager import GenerationConfig
            gen_config = GenerationConfig(
                temperature=0.0,
                max_tokens=512,
                stop_sequences=['Question:', '\n\nQuestion:']
            )
            
            response = await model.generate(prompt, gen_config)
            
            # Simple keyword-based scoring (would need expert evaluation in practice)
            accuracy_score = self._score_expert_answer(response, question['answer'])
            
            result = {
                'question': question['question'],
                'domain': question['domain'],
                'difficulty': question['difficulty'],
                'response': response,
                'expected_answer': question['answer'],
                'accuracy_score': accuracy_score,
                'is_correct': accuracy_score > 0.5
            }
            results.append(result)
        
        self.logger.info(f"HLE evaluation completed: {len(results)} samples")
        return results
    
    def _score_expert_answer(self, response: str, expected: str) -> float:
        """Score expert-level answer (simplified)."""
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        
        if len(expected_words) == 0:
            return 0.0
        
        overlap = len(response_words & expected_words)
        return overlap / len(expected_words)
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": "Humanity's Last Exam",
            "description": "Expert-level questions across subjects requiring deep knowledge",
            "question_type": "open_ended",
            "metric": "expert_score",
            "difficulty": "expert"
        }

class CodeforcesBenchmark(BaseBenchmark):
    """Codeforces competitive programming benchmark."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.difficulty_range = config.get('evaluation', {}).get('benchmarks', {}).get('codeforces', {}).get('difficulty', [800, 1600])
        
    async def evaluate(self, model, num_samples: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on Codeforces problems."""
        self.logger.info(f"Starting Codeforces evaluation with {num_samples} samples")
        
        # Placeholder Codeforces-style problems
        cf_problems = [
            {
                "title": "Array Rotation",
                "problem": "Given an array of n integers, rotate it k positions to the right. Return the rotated array.",
                "input_format": "First line: n, k. Second line: n integers.",
                "output_format": "The rotated array as space-separated integers.",
                "sample_input": "5 2\n1 2 3 4 5",
                "sample_output": "4 5 1 2 3",
                "difficulty": 900,
                "tags": ["implementation", "arrays"]
            }
        ]
        
        results = []
        for i in range(min(num_samples, len(cf_problems) * 10)):
            problem = cf_problems[i % len(cf_problems)]
            
            prompt = self._create_codeforces_prompt(problem)
            
            from src.models.model_manager import GenerationConfig
            gen_config = GenerationConfig(
                temperature=0.1,
                max_tokens=1024,
                stop_sequences=['```\n\n', 'Problem:', 'Next problem:']
            )
            
            response = await model.generate(prompt, gen_config)
            
            # Extract code and basic validation
            code = self._extract_code_from_response(response)
            is_syntactically_correct = self._check_code_syntax(code)
            
            result = {
                'title': problem['title'],
                'problem': problem['problem'],
                'sample_input': problem['sample_input'],
                'sample_output': problem['sample_output'],
                'difficulty': problem['difficulty'],
                'tags': problem['tags'],
                'generated_code': code,
                'full_response': response,
                'is_syntactically_correct': is_syntactically_correct,
                'estimated_rating': self._estimate_solution_rating(code, problem)
            }
            results.append(result)
        
        self.logger.info(f"Codeforces evaluation completed: {len(results)} samples")
        return results
    
    def _create_codeforces_prompt(self, problem: Dict) -> str:
        """Create prompt for Codeforces problem."""
        prompt = f"""Solve this competitive programming problem from Codeforces.

Problem: {problem['title']}
{problem['problem']}

Input Format: {problem['input_format']}
Output Format: {problem['output_format']}

Sample Input:
{problem['sample_input']}

Sample Output:
{problem['sample_output']}

Provide a complete Python solution:

```python
"""
        return prompt
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from model response."""
        # Look for code blocks
        code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback: look for indented code
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('n =') or line.strip().startswith('for '):
                in_code = True
            if in_code:
                code_lines.append(line)
                if line.strip() == '' and len(code_lines) > 5:
                    break
        
        return '\n'.join(code_lines)
    
    def _estimate_solution_rating(self, code: str, problem: Dict) -> int:
        """Estimate the competitive programming rating of solution."""
        base_rating = problem['difficulty']
        
        # Simple heuristics
        if 'def ' in code:
            base_rating += 100  # Function usage
        if any(keyword in code for keyword in ['sort', 'sorted', 'collections']):
            base_rating += 50   # Advanced techniques
        if len(code.split('\n')) > 20:
            base_rating += 50   # Complex solution
        
        return min(base_rating, 3000)  # Cap at master level
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": "Codeforces",
            "description": "Competitive programming problems with difficulty ratings",
            "question_type": "code_generation",
            "metric": "elo_rating",
            "difficulty": "varies"
        }

class TauBenchmark(BaseBenchmark):
    """TauBench for function calling and tool use evaluation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.domain = config.get('evaluation', {}).get('benchmarks', {}).get('tau', {}).get('domain', 'retail')
        
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on tool use scenarios."""
        self.logger.info(f"Starting TauBench {self.domain} evaluation with {num_samples} samples")
        
        # Placeholder tool use scenarios
        tool_scenarios = [
            {
                "task": "Find the cheapest laptop under $1000 with at least 8GB RAM",
                "available_tools": ["search_products", "filter_by_price", "filter_by_specs"],
                "expected_calls": ["search_products(category='laptop')", "filter_by_price(max=1000)", "filter_by_specs(ram='8GB+')"],
                "domain": "retail"
            }
        ]
        
        results = []
        for i in range(min(num_samples, len(tool_scenarios) * 10)):
            scenario = tool_scenarios[i % len(tool_scenarios)]
            
            prompt = self._create_tau_prompt(scenario)
            
            from src.models.model_manager import GenerationConfig
            gen_config = GenerationConfig(
                temperature=0.1,
                max_tokens=512,
                stop_sequences=['Task:', '\n\nTask:']
            )
            
            response = await model.generate(prompt, gen_config)
            
            # Parse function calls from response
            predicted_calls = self._extract_function_calls(response)
            accuracy = self._score_function_calling(predicted_calls, scenario['expected_calls'])
            
            result = {
                'task': scenario['task'],
                'available_tools': scenario['available_tools'],
                'expected_calls': scenario['expected_calls'],
                'predicted_calls': predicted_calls,
                'response': response,
                'accuracy': accuracy,
                'domain': scenario['domain']
            }
            results.append(result)
        
        self.logger.info(f"TauBench evaluation completed: {len(results)} samples")
        return results
    
    def _create_tau_prompt(self, scenario: Dict) -> str:
        """Create prompt for tool use scenario."""
        prompt = f"""You have access to the following tools: {', '.join(scenario['available_tools'])}

Task: {scenario['task']}

Plan your approach and call the necessary functions in order. Format function calls as: function_name(parameters)

Response:"""
        return prompt
    
    def _extract_function_calls(self, response: str) -> List[str]:
        """Extract function calls from response."""
        # Look for function call patterns
        function_pattern = r'(\w+\([^)]*\))'
        calls = re.findall(function_pattern, response)
        return calls
    
    def _score_function_calling(self, predicted: List[str], expected: List[str]) -> float:
        """Score function calling accuracy."""
        if not expected:
            return 1.0 if not predicted else 0.0
        
        correct = 0
        for exp_call in expected:
            if any(exp_call.split('(')[0] in pred_call for pred_call in predicted):
                correct += 1
        
        return correct / len(expected)
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": f"TauBench {self.domain.title()}",
            "description": "Function calling and tool use in realistic scenarios",
            "question_type": "tool_use",
            "metric": "function_accuracy",
            "domain": self.domain
        }

class HellaSwagBenchmark(BaseBenchmark):
    """HellaSwag commonsense inference benchmark (used by xAI and others)."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
    
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on HellaSwag dataset."""
        self.logger.info(f"Starting HellaSwag evaluation with {num_samples} samples")
        
        # Load dataset with error handling
        dataset = safe_load_dataset("hellaswag", split="validation", streaming=True)
        
        if dataset is None:
            self.logger.error("Failed to load HellaSwag dataset - offline or connection issues")
            return [{
                'question': 'Dataset loading failed',
                'correct_answer': 'N/A',
                'predicted_answer': 'N/A',
                'is_correct': False,
                'response': 'Error: HellaSwag dataset could not be loaded. Check internet connection.',
                'prompt': 'N/A',
                'error': 'Dataset loading failed - offline or connection issues'
            }]
        
        # Take only the number of samples we need
        dataset_list = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            dataset_list.append(item)
        
        # Prepare prompts
        prompts = []
        for item in dataset_list:
            prompt = self._create_hellaswag_prompt(item)
            prompts.append(prompt)
        
        # Generate responses
        from src.models.model_manager import GenerationConfig
        gen_config = GenerationConfig(
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 128),
            stop_sequences=['Question:', '\n\nQuestion:']
        )
        
        responses = await model.batch_generate(prompts, gen_config)
        
        # Process results
        results = []
        for item, prompt, response in zip(dataset_list, prompts, responses):
            predicted = self.extract_choice(response)
            correct = item['label']
            
            result = {
                'question': item['ctx'],
                'correct_answer': correct,
                'predicted_answer': predicted,
                'is_correct': str(predicted) == str(correct),
                'response': response,
                'prompt': prompt,
                'choices': item['endings']
            }
            results.append(result)
        
        self.logger.info(f"HellaSwag evaluation completed: {len(results)} samples")
        return results
    
    def _create_hellaswag_prompt(self, item: Dict) -> str:
        """Create prompt for HellaSwag completion."""
        prompt = f"Complete the following scenario by choosing the most likely continuation:\n\n"
        prompt += f"Context: {item['ctx']}\n\n"
        prompt += "Choices:\n"
        for i, ending in enumerate(item['endings']):
            prompt += f"{i}. {ending}\n"
        prompt += "\nAnswer (0, 1, 2, or 3): "
        return prompt
    
    def extract_choice(self, text: str) -> str:
        """Extract choice (0-3) from response."""
        # Look for patterns like "0", "1", "2", "3" or "Choice 0", etc.
        import re
        patterns = [
            r'\b([0-3])\b',
            r'choice\s*([0-3])',
            r'answer\s*([0-3])',
            r'option\s*([0-3])'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        
        # Default to first character that's 0-3
        for char in text:
            if char in '0123':
                return char
        
        return "0"  # Default
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": "HellaSwag",
            "description": "Commonsense inference on everyday scenarios",
            "question_type": "multiple_choice",
            "metric": "accuracy"
        }

class ARCBenchmark(BaseBenchmark):
    """ARC (AI2 Reasoning Challenge) - used by DeepMind for scientific reasoning."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.variant = config.get('evaluation', {}).get('benchmarks', {}).get('arc', {}).get('variant', 'ARC-Challenge')
    
    async def evaluate(self, model, num_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on ARC dataset."""
        self.logger.info(f"Starting ARC evaluation with {num_samples} samples")
        
        # Load dataset with error handling
        dataset = safe_load_dataset("ai2_arc", self.variant, split="test", streaming=True)
        
        if dataset is None:
            self.logger.error("Failed to load ARC dataset - offline or connection issues")
            return [{
                'question': 'Dataset loading failed',
                'correct_answer': 'N/A',
                'predicted_answer': 'N/A',
                'is_correct': False,
                'response': 'Error: ARC dataset could not be loaded. Check internet connection.',
                'prompt': 'N/A',
                'error': 'Dataset loading failed - offline or connection issues'
            }]
        
        # Take only the number of samples we need
        dataset_list = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            dataset_list.append(item)
        
        # Prepare prompts
        prompts = []
        for item in dataset_list:
            prompt = self._create_arc_prompt(item)
            prompts.append(prompt)
        
        # Generate responses
        from src.models.model_manager import GenerationConfig
        gen_config = GenerationConfig(
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 256),
            stop_sequences=['Question:', '\n\nQuestion:']
        )
        
        responses = await model.batch_generate(prompts, gen_config)
        
        # Process results
        results = []
        for item, prompt, response in zip(dataset_list, prompts, responses):
            predicted = self.extract_answer_letter(response)
            correct = item['answerKey']
            
            result = {
                'question': item['question'],
                'correct_answer': correct,
                'predicted_answer': predicted,
                'is_correct': predicted.upper() == correct.upper(),
                'response': response,
                'prompt': prompt,
                'choices': item['choices']
            }
            results.append(result)
        
        self.logger.info(f"ARC evaluation completed: {len(results)} samples")
        return results
    
    def _create_arc_prompt(self, item: Dict) -> str:
        """Create prompt for ARC question."""
        prompt = f"Answer the following science question:\n\n"
        prompt += f"Question: {item['question']}\n\n"
        prompt += "Choices:\n"
        for choice in item['choices']['text']:
            label = item['choices']['label'][item['choices']['text'].index(choice)]
            prompt += f"{label}. {choice}\n"
        prompt += f"\nAnswer ({'/'.join(item['choices']['label'])}): "
        return prompt
    
    def extract_answer_letter(self, text: str) -> str:
        """Extract answer letter (A, B, C, D, etc.) from response."""
        import re
        # Look for patterns like "A", "B", "C", "D", etc.
        patterns = [
            r'\b([A-Z])\b',
            r'answer\s*([A-Z])',
            r'choice\s*([A-Z])',
            r'option\s*([A-Z])'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.upper())
            if match:
                return match.group(1)
        
        # Default to first letter found
        for char in text.upper():
            if char in 'ABCDEFGHIJ':
                return char
        
        return "A"  # Default
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            "name": f"ARC {self.variant}",
            "description": "AI2 Reasoning Challenge - science questions",
            "question_type": "multiple_choice",
            "metric": "accuracy",
            "variant": self.variant
        }

class BenchmarkRegistry:
    """Registry for all available benchmarks matching OpenAI's evaluation suite."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.benchmarks = {
            # Core academic benchmarks
            'mmlu': MMLUBenchmark(config),
            'gsm8k': GSM8KBenchmark(config),
            'math': MATHBenchmark(config),
            'humaneval': HumanEvalBenchmark(config),
            'hellaswag': HellaSwagBenchmark(config),
            'arc': ARCBenchmark(config),
            
            # Competition math (like OpenAI)
            'aime_2024': AIMEBenchmark({**config, 'evaluation': {**config.get('evaluation', {}), 'benchmarks': {**config.get('evaluation', {}).get('benchmarks', {}), 'aime': {'year': '2024'}}}}),
            'aime_2025': AIMEBenchmark({**config, 'evaluation': {**config.get('evaluation', {}), 'benchmarks': {**config.get('evaluation', {}).get('benchmarks', {}), 'aime': {'year': '2025'}}}}),
            
            # PhD-level science
            'gpqa_diamond': GPQABenchmark({**config, 'evaluation': {**config.get('evaluation', {}), 'benchmarks': {**config.get('evaluation', {}).get('benchmarks', {}), 'gpqa': {'variant': 'diamond'}}}}),
            'gpqa_main': GPQABenchmark({**config, 'evaluation': {**config.get('evaluation', {}), 'benchmarks': {**config.get('evaluation', {}).get('benchmarks', {}), 'gpqa': {'variant': 'main'}}}}),
            
            # Expert-level knowledge
            'hle': HumanityLastExamBenchmark(config),
            
            # Health and medical
            'healthbench': HealthBenchmark({**config, 'evaluation': {**config.get('evaluation', {}), 'benchmarks': {**config.get('evaluation', {}).get('benchmarks', {}), 'healthbench': {'variant': 'standard'}}}}),
            'healthbench_hard': HealthBenchmark({**config, 'evaluation': {**config.get('evaluation', {}), 'benchmarks': {**config.get('evaluation', {}).get('benchmarks', {}), 'healthbench': {'variant': 'hard'}}}}),
            
            # Competitive programming
            'codeforces': CodeforcesBenchmark(config),
            
            # Tool use and function calling
            'tau_retail': TauBenchmark({**config, 'evaluation': {**config.get('evaluation', {}), 'benchmarks': {**config.get('evaluation', {}).get('benchmarks', {}), 'tau': {'domain': 'retail'}}}}),
            'tau_general': TauBenchmark({**config, 'evaluation': {**config.get('evaluation', {}), 'benchmarks': {**config.get('evaluation', {}).get('benchmarks', {}), 'tau': {'domain': 'general'}}}}),
        }
        
        # Define benchmark suites (aligned with major AI lab standards)
        self.suites = {
            # Industry-standard suites
            'openai_suite': OPENAI_BENCHMARK_SUITE,
            'anthropic_suite': ANTHROPIC_SUITE,
            'deepmind_suite': DEEPMIND_SUITE,
            'xai_suite': XAI_SUITE,
            
            # Specialized suites
            'competition_suite': COMPETITION_SUITE,
            'expert_suite': EXPERT_SUITE,
            'coding_suite': CODING_SUITE,
            'reasoning_suite': REASONING_SUITE,
            
            # General purpose suites
            'comprehensive_suite': COMPREHENSIVE_SUITE,
            'quick_suite': QUICK_SUITE,
            'safe_suite': SAFE_SUITE,  # Safe for slow laptops
            'legacy_suite': ['gsm8k', 'mmlu'],
            'all_benchmarks': list(self.benchmarks.keys())  # All available individual benchmarks
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get_benchmark(self, name: str) -> BaseBenchmark:
        """Get benchmark by name."""
        if name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {name}. Available: {list(self.benchmarks.keys())}")
        return self.benchmarks[name]
    
    def is_suite(self, name: str) -> bool:
        """Check if name refers to a benchmark suite."""
        return name in self.suites
    
    def expand_suite(self, suite_name: str) -> List[str]:
        """Expand a suite name to its component benchmark names."""
        if suite_name not in self.suites:
            # If it's not a suite, assume it's a single benchmark
            return [suite_name]
        return self.suites[suite_name]
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of all available individual benchmark names."""
        return list(self.benchmarks.keys())
    
    def get_available_suites(self) -> List[str]:
        """Get list of all available suite names."""
        return list(self.suites.keys())
    
    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """List all available benchmarks."""
        return [
            {**benchmark.get_benchmark_info(), 'key': name}
            for name, benchmark in self.benchmarks.items()
        ]
    
    def get_benchmarks_by_category(self) -> Dict[str, List[str]]:
        """Get benchmarks organized by category."""
        return {
            'core_academic': ['mmlu', 'gsm8k', 'math', 'humaneval', 'hellaswag', 'arc'],
            'competition_math': ['aime_2024', 'aime_2025'],
            'expert_science': ['gpqa_diamond', 'gpqa_main', 'hle'],
            'health_medical': ['healthbench', 'healthbench_hard'],
            'coding_competition': ['codeforces', 'humaneval'],
            'tool_use': ['tau_retail', 'tau_general'],
            'commonsense_reasoning': ['hellaswag', 'arc'],
            'comprehensive_suite': ['mmlu', 'gsm8k', 'math', 'humaneval', 'aime_2024', 'gpqa_diamond', 'healthbench', 'codeforces', 'tau_retail', 'hellaswag', 'arc']
        }
    
    def add_benchmark(self, name: str, benchmark: BaseBenchmark):
        """Add a custom benchmark."""
        self.benchmarks[name] = benchmark
        self.logger.info(f"Added custom benchmark: {name}")

# Enhanced benchmark suites aligned with major AI lab standards

# OpenAI-style comprehensive suite (includes GPT-4 evaluation benchmarks)
OPENAI_BENCHMARK_SUITE = [
    'mmlu', 'gsm8k', 'math', 'humaneval', 
    'aime_2024', 'aime_2025', 'gpqa_diamond', 
    'hle', 'healthbench', 'codeforces', 'tau_retail',
    'hellaswag', 'arc'  # Added missing OpenAI standards
]

# Anthropic-style reasoning suite (BBH + conversation quality)
ANTHROPIC_SUITE = [
    'mmlu', 'gsm8k', 'bigbench_hard', 'gpqa_diamond', 
    'mt_bench', 'hellaswag'
]

# Google DeepMind suite (science + instruction following)
DEEPMIND_SUITE = [
    'mmlu', 'gsm8k', 'arc', 'gpqa_diamond', 
    'ifeval', 'math'
]

# xAI-style suite (practical reasoning + coding)
XAI_SUITE = [
    'hellaswag', 'gsm8k', 'humaneval', 'mbpp', 
    'tau_retail', 'codeforces'
]

# Competition-focused suite (contest-level problems)
COMPETITION_SUITE = ['aime_2024', 'aime_2025', 'codeforces', 'math']

# Expert knowledge suite (PhD-level questions)
EXPERT_SUITE = ['gpqa_diamond', 'hle', 'healthbench_hard']

# Standard comprehensive suite (balanced evaluation)
COMPREHENSIVE_SUITE = ['mmlu', 'gsm8k', 'math', 'humaneval', 'gpqa_diamond', 'healthbench']

# Quick test suite (fast evaluation)
QUICK_SUITE = ['gsm8k', 'humaneval', 'mmlu']

# Safe suite for slow laptops (only reliable benchmarks, no heavy dataset downloads)
SAFE_SUITE = ['gsm8k', 'mmlu']  # Most reliable benchmarks

# Coding-focused suite (programming evaluation)
CODING_SUITE = ['humaneval', 'mbpp', 'codeforces']

# Reasoning suite (logical thinking)
REASONING_SUITE = ['gsm8k', 'arc', 'hellaswag', 'bigbench_hard']

# Legacy compatibility
STANDARD_BENCHMARK_SUITE = COMPREHENSIVE_SUITE
QUICK_BENCHMARK_SUITE = QUICK_SUITE
