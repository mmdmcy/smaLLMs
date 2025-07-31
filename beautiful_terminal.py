#!/usr/bin/env python3
"""
smaLLMs Beautiful Terminal Interface
===================================
Clean, professional terminal output inspired by modern benchmarking tools.
"""

import time
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import asyncio

@dataclass
class EvaluationStatus:
    """Track evaluation status for display."""
    model: str
    benchmark: str
    tests_completed: int
    total_tests: int
    accuracy: Optional[float] = None
    errors: int = 0
    running: bool = False
    avg_cost: float = 0.0
    avg_duration: float = 0.0
    slowest_time: float = 0.0

class BeautifulTerminal:
    """Beautiful terminal interface for evaluations."""
    
    def __init__(self):
        self.statuses: List[EvaluationStatus] = []
        self.overall_stats = {
            'completed': 0,
            'total': 0,
            'correct': 0,
            'errors': 0,
            'running': 0,
            'total_cost': 0.0,
            'avg_duration': 0.0
        }
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def clear_screen(self):
        """Clear the terminal screen."""
        print('\033[2J\033[H', end='')
    
    def print_header(self):
        """Print the beautiful header."""
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        
        print(f"\033[96m→  \033[0msmaLLMs \033[90mevaluation platform\033[0m \033[90m✕ \033[94mPython Benchmarking Suite\033[0m")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\033[90mRunning smaLLMs Evaluation @ \033[91m{current_time}\033[90m...\033[0m")
        print()
    
    def print_table_header(self):
        """Print the table header."""
        headers = [
            "Model", "Tests", "% Right", "Errors", "Running Tests", 
            "Avg Cost", "Avg Duration", "Slowest"
        ]
        
        # Print header with proper spacing
        print(f"\033[1m{'Model':<20} {'Tests':<8} {'% Right':<8} {'Errors':<8} {'Running Tests':<15} {'Avg Cost':<10} {'Avg Duration':<12} {'Slowest':<8}\033[0m")
        
    def format_model_name(self, model: str) -> str:
        """Format model name to be shorter and cleaner."""
        if '/' in model:
            return model.split('/')[-1].replace('-', '-').replace('Instruct', '').replace('instruct', '').strip('-')
        return model
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in a clean way."""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m{secs:.1f}s"
    
    def get_status_color(self, accuracy: Optional[float], running: bool) -> str:
        """Get color code based on status."""
        if running:
            return '\033[93m'  # Yellow for running
        elif accuracy is None:
            return '\033[90m'  # Gray for not started
        elif accuracy == 0:
            return '\033[91m'  # Red for failed
        elif accuracy < 0.3:
            return '\033[91m'  # Red for poor performance
        elif accuracy < 0.7:
            return '\033[93m'  # Yellow for moderate performance
        else:
            return '\033[92m'  # Green for good performance
    
    def print_model_row(self, status: EvaluationStatus):
        """Print a single model row."""
        model_name = self.format_model_name(status.model)
        
        # Calculate percentage
        if status.total_tests > 0:
            pct_right = (status.accuracy or 0) * 100
            tests_display = f"{status.tests_completed}/{status.total_tests}"
        else:
            pct_right = 0
            tests_display = "-"
        
        # Status and colors
        color = self.get_status_color(status.accuracy, status.running)
        reset = '\033[0m'
        
        # Running tests indicator
        running_indicator = str(status.tests_completed) if status.running else "-"
        
        # Format costs and times
        cost_display = f"${status.avg_cost:.4f}" if status.avg_cost > 0 else "$0.0000"
        duration_display = self.format_duration(status.avg_duration)
        slowest_display = self.format_duration(status.slowest_time)
        
        print(f"{color}{model_name:<20}{reset} {tests_display:<8} {pct_right:>6.0f}% {reset}"
              f"{status.errors:<8} {running_indicator:<15} {cost_display:<10} {duration_display:<12} {slowest_display:<8}")
    
    def print_progress_bar(self):
        """Print overall progress bar."""
        if self.overall_stats['total'] > 0:
            progress = self.overall_stats['completed'] / self.overall_stats['total']
            bar_width = 40
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            pct = progress * 100
            
            print(f"\n\033[92m{bar}\033[0m \033[91m{pct:.0f}%\033[0m ({self.overall_stats['completed']}/{self.overall_stats['total']} completed)")
        
    def print_overall_stats(self):
        """Print overall statistics."""
        elapsed = time.time() - self.start_time
        
        # Calculate overall accuracy
        if self.overall_stats['completed'] > 0:
            overall_accuracy = (self.overall_stats['correct'] / self.overall_stats['completed']) * 100
        else:
            overall_accuracy = 0
        
        print(f"\nOverall: {self.overall_stats['completed']}/{self.overall_stats['total']} done • "
              f"{overall_accuracy:.0f}% correct • {self.overall_stats['errors']} errors • "
              f"{self.overall_stats['running']} running • {self.format_duration(self.overall_stats['avg_duration'])} avg duration • "
              f"${self.overall_stats['total_cost']:.4f} total cost")
    
    def update_status(self, model: str, benchmark: str, tests_completed: int, 
                     total_tests: int, accuracy: Optional[float] = None, 
                     errors: int = 0, running: bool = False, cost: float = 0.0, 
                     duration: float = 0.0):
        """Update status for a model."""
        with self.lock:
            # Find or create status
            status_id = f"{model}_{benchmark}"
            status = None
            for s in self.statuses:
                if f"{s.model}_{s.benchmark}" == status_id:
                    status = s
                    break
            
            if status is None:
                status = EvaluationStatus(
                    model=model,
                    benchmark=benchmark,
                    tests_completed=tests_completed,
                    total_tests=total_tests
                )
                self.statuses.append(status)
            
            # Update status
            status.tests_completed = tests_completed
            status.total_tests = total_tests
            status.accuracy = accuracy
            status.errors = errors
            status.running = running
            if cost > 0:
                status.avg_cost = cost
            if duration > 0:
                status.avg_duration = duration
                status.slowest_time = max(status.slowest_time, duration)
    
    def update_overall_stats(self):
        """Update overall statistics."""
        with self.lock:
            self.overall_stats['completed'] = sum(1 for s in self.statuses if s.accuracy is not None)
            self.overall_stats['total'] = len(self.statuses)
            self.overall_stats['correct'] = sum(int((s.accuracy or 0) * s.tests_completed) for s in self.statuses if s.accuracy is not None)
            self.overall_stats['errors'] = sum(s.errors for s in self.statuses)
            self.overall_stats['running'] = sum(1 for s in self.statuses if s.running)
            self.overall_stats['total_cost'] = sum(s.avg_cost for s in self.statuses)
            
            completed_statuses = [s for s in self.statuses if s.accuracy is not None]
            if completed_statuses:
                self.overall_stats['avg_duration'] = sum(s.avg_duration for s in completed_statuses) / len(completed_statuses)
    
    def render(self):
        """Render the complete interface."""
        self.update_overall_stats()
        
        self.clear_screen()
        self.print_header()
        self.print_table_header()
        
        # Sort statuses: running first, then by accuracy descending
        sorted_statuses = sorted(self.statuses, 
                               key=lambda s: (not s.running, -(s.accuracy or 0)))
        
        for status in sorted_statuses:
            self.print_model_row(status)
        
        self.print_progress_bar()
        self.print_overall_stats()
        
        sys.stdout.flush()

# Global terminal instance
terminal = BeautifulTerminal()

def start_evaluation_display(models: List[str], benchmarks: List[str]):
    """Initialize the evaluation display."""
    # Initialize all model-benchmark combinations
    for model in models:
        for benchmark in benchmarks:
            terminal.update_status(model, benchmark, 0, 100)  # Assume 100 tests per benchmark
    
    terminal.render()

def update_evaluation_progress(model: str, benchmark: str, completed: int, 
                             total: int, accuracy: Optional[float] = None,
                             cost: float = 0.0, duration: float = 0.0, 
                             running: bool = False):
    """Update progress for a specific evaluation."""
    terminal.update_status(model, benchmark, completed, total, accuracy, 
                          cost=cost, duration=duration, running=running)
    terminal.render()

def evaluation_completed(model: str, benchmark: str, accuracy: float, 
                        cost: float, duration: float):
    """Mark evaluation as completed."""
    terminal.update_status(model, benchmark, 100, 100, accuracy, 
                          cost=cost, duration=duration, running=False)
    terminal.render()

def evaluation_failed(model: str, benchmark: str, error_count: int = 1):
    """Mark evaluation as failed."""
    terminal.update_status(model, benchmark, 0, 100, accuracy=0.0, 
                          errors=error_count, running=False)
    terminal.render()

def start_evaluation(model: str, benchmark: str):
    """Mark evaluation as started."""
    terminal.update_status(model, benchmark, 0, 100, running=True)
    terminal.render()

if __name__ == "__main__":
    # Demo of the beautiful terminal
    models = ["google/gemma-2-2b-it", "Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]
    benchmarks = ["gsm8k", "mmlu"]
    
    start_evaluation_display(models, benchmarks)
    
    # Simulate some evaluations
    import random
    for i in range(10):
        time.sleep(1)
        model = random.choice(models)
        benchmark = random.choice(benchmarks)
        
        if i < 3:
            start_evaluation(model, benchmark)
        elif i < 7:
            update_evaluation_progress(model, benchmark, random.randint(10, 90), 100, running=True)
        else:
            evaluation_completed(model, benchmark, random.uniform(0.1, 0.9), 
                               random.uniform(0.01, 0.5), random.uniform(10, 60))
