#!/usr/bin/env python3
"""
smaLLMs - Small Language Model Evaluation Platform
================================================
One command to rule them all - streamlined evaluation experience
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import asyncio
from typing import Dict, List, Any

# Ensure we can import our modules
sys.path.append(str(Path(__file__).parent))

from intelligent_evaluator import IntelligentEvaluationOrchestrator, IntelligentEvaluationConfig
from simple_exporter import SimpleResultsExporter
from src.models.model_manager import ModelManager
import yaml

class BeautifulSmaLLMsTerminal:
    """Beautiful terminal interface matching your reference image."""
    
    def __init__(self):
        self.models_tested = 0
        self.total_tests = 0
        self.correct_answers = 0
        self.total_cost = 0.0
        self.start_time = time.time()
        self.active_models = []
        
        # Colors for terminal
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.MAGENTA = '\033[95m'
        self.CYAN = '\033[96m'
        self.WHITE = '\033[97m'
        self.BOLD = '\033[1m'
        self.RESET = '\033[0m'
        self.GRAY = '\033[90m'
        
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def format_duration(self, seconds: float) -> str:
        """Format duration nicely."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def format_cost(self, cost: float) -> str:
        """Format cost nicely."""
        return f"${cost:.4f}"
    
    def print_header(self):
        """Print the beautiful header."""
        print(f"\n{self.CYAN}→{self.RESET}  {self.BOLD}smaLLMs{self.RESET} {self.GRAY}evaluation platform{self.RESET} {self.RED}×{self.RESET} {self.WHITE}Small Language Model Benchmarking{self.RESET}")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{self.GRAY}Running smaLLMs Evaluation Suite @ {current_time}{self.RESET}\n")
    
    def print_model_table_header(self):
        """Print the model evaluation table header."""
        print(f"{self.BOLD}{'Model':<20} {'Tests':<8} {'% Right':<8} {'Errors':<8} {'Running Tests':<13} {'Avg Cost':<10} {'Avg Duration':<12} {'Slowest':<8}{self.RESET}")
        print(f"{self.GRAY}{'-' * 95}{self.RESET}")
    
    def update_model_status(self, model_name: str, tests_done: int, total_tests: int, 
                          correct: int, errors: int, running: int, avg_cost: float, 
                          avg_duration: float, slowest: float):
        """Update status for a specific model."""
        clean_name = model_name.split('/')[-1][:18]  # Clean and truncate
        
        # Calculate percentage
        pct_right = (correct / tests_done * 100) if tests_done > 0 else 0
        
        # Color code percentage
        if pct_right >= 80:
            pct_color = self.GREEN
        elif pct_right >= 50:
            pct_color = self.YELLOW
        elif pct_right > 0:
            pct_color = self.CYAN  # Changed from RED to CYAN for low but non-zero performance
        else:
            pct_color = self.GRAY  # Gray for zero performance (not an error, just no success)
        
        # Color code running tests
        running_color = self.CYAN if running > 0 else self.GRAY
        
        print(f"{clean_name:<20} {tests_done}/{total_tests:<4} {pct_color}{pct_right:>6.0f}%{self.RESET} "
              f"{errors if errors > 0 else '-':<8} {running_color}{running if running > 0 else '-':<13}{self.RESET} "
              f"{self.format_cost(avg_cost):<10} {self.format_duration(avg_duration):<12} "
              f"{self.format_duration(slowest)}")
    
    def print_progress_bar(self, completed: int, total: int, width: int = 50):
        """Print a beautiful progress bar."""
        if total == 0:
            return
            
        progress = completed / total
        filled = int(width * progress)
        bar = f"{self.GREEN}{'█' * filled}{self.GRAY}{'░' * (width - filled)}{self.RESET}"
        percentage = progress * 100
        
        print(f"\n{bar} {percentage:.0f}% ({completed}/{total} completed)")
    
    def print_summary(self, completed: int, total: int, correct: int, errors: int, 
                     running: int, avg_duration: float, total_cost: float):
        """Print the bottom summary line."""
        overall_pct = (correct / completed * 100) if completed > 0 else 0
        
        # Only use red for actual errors
        error_color = self.RED if errors > 0 else self.GRAY
        pct_color = self.GREEN if overall_pct >= 50 else self.CYAN if overall_pct > 0 else self.GRAY
        
        print(f"\n{self.BOLD}Overall:{self.RESET} {completed}/{total} done • "
              f"{pct_color}{overall_pct:.0f}% correct{self.RESET} • "
              f"{error_color}{errors if errors > 0 else '-'} errors{self.RESET} • "
              f"{self.CYAN}{running} running{self.RESET} • "
              f"{self.format_duration(avg_duration)} avg duration • "
              f"{self.format_cost(total_cost)} total cost")

class SmaLLMsLauncher:
    """Central launcher for the smaLLMs platform."""
    
    def __init__(self):
        self.terminal = BeautifulSmaLLMsTerminal()
        self.orchestrator = None
        self.exporter = SimpleResultsExporter()
        self.current_eval_metadata = {}  # Track current evaluation metadata
        self.model_manager = None
        self.config = self.load_config()
        
        # Ensure directories exist
        Path("results/cache").mkdir(parents=True, exist_ok=True)
        Path("smaLLMs_results").mkdir(parents=True, exist_ok=True)
        Path("website_exports").mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Return default config
            config = {
                'evaluation_mode': {
                    'default': 'local',
                    'prefer_local': True,
                    'auto_discover_models': True,
                    'include_vision_models': True
                },
                'ollama': {'base_url': 'http://localhost:11434'},
                'lm_studio': {'base_url': 'http://localhost:1234'},
                'storage': {'local_cache_mb': 100, 'low_resource_mode': True}  # Conservative defaults
            }
        
        # Load external model configuration
        models_config_path = Path("config/models.yaml")
        if models_config_path.exists():
            with open(models_config_path, 'r') as f:
                models_config = yaml.safe_load(f)
                config['external_models'] = models_config
        
        return config
    
    async def init_model_manager(self):
        """Initialize model manager and discover local models."""
        if self.model_manager is None:
            self.model_manager = ModelManager(self.config)
            if self.config.get('evaluation_mode', {}).get('auto_discover_models', True):
                try:
                    await self.model_manager.discover_local_models()
                except Exception as e:
                    print(f"{self.terminal.YELLOW}  Could not discover some local models: {e}{self.terminal.RESET}")
        return self.model_manager
    
    def print_welcome(self):
        """Print welcome message and mode selection with OpenAI-level capabilities."""
        self.terminal.clear_screen()
        print(f"""
{self.terminal.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                          {self.terminal.BOLD}smaLLMs Evaluation Platform{self.terminal.RESET}{self.terminal.CYAN}                          ║
║                     {self.terminal.GRAY}OpenAI-Level LLM Benchmarking Suite{self.terminal.RESET}{self.terminal.CYAN}                        ║
║                     {self.terminal.YELLOW}Now with ALL benchmarks used by top AI labs!{self.terminal.RESET}{self.terminal.CYAN}                     ║
╚══════════════════════════════════════════════════════════════════════════════╝{self.terminal.RESET}

{self.terminal.BOLD}Choose Your Evaluation Mode:{self.terminal.RESET}

{self.terminal.BOLD}LOCAL MODEL EVALUATION{self.terminal.RESET}
  {self.terminal.GREEN}local{self.terminal.RESET}    - {self.terminal.BOLD}Enter Local Mode{self.terminal.RESET}      - Evaluate your Ollama & LM Studio models
  
{self.terminal.BOLD}CLOUD MODEL EVALUATION{self.terminal.RESET}  
  {self.terminal.GREEN}cloud{self.terminal.RESET}    - {self.terminal.BOLD}Enter Cloud Mode{self.terminal.RESET}      - Evaluate HuggingFace models via API

{self.terminal.BOLD}MARATHON MODE{self.terminal.RESET}
  {self.terminal.RED}marathon{self.terminal.RESET}  - {self.terminal.BOLD}Overnight Mode{self.terminal.RESET}        - ALL models × ALL benchmarks (while you sleep!)

{self.terminal.BOLD}NEW: OpenAI-Level Benchmarks Available:{self.terminal.RESET}
  Competition Math: AIME 2024/2025, MATH, GSM8K
  Competitive Programming: Codeforces, HumanEval
  Expert Knowledge: GPQA Diamond, HLE, MMLU  
  Specialized: HealthBench, TauBench (tool use)

{self.terminal.BOLD}Quick Commands:{self.terminal.RESET}
  {self.terminal.CYAN}discover{self.terminal.RESET}  - Find all local models on your system
  {self.terminal.CYAN}status{self.terminal.RESET}    - Show current results summary  
  {self.terminal.CYAN}export{self.terminal.RESET}    - Export results for website/analysis
  {self.terminal.CYAN}space{self.terminal.RESET}     - Check current disk space usage
  {self.terminal.CYAN}help{self.terminal.RESET}      - Show this menu again

{self.terminal.BOLD}Configuration:{self.terminal.RESET}
  {self.terminal.GRAY}config/config.yaml{self.terminal.RESET}       - Main configuration settings
  {self.terminal.GRAY}config/models.yaml{self.terminal.RESET}       - {self.terminal.YELLOW}Add your own HuggingFace models here{self.terminal.RESET}

{self.terminal.YELLOW}NEW: Marathon Mode - Perfect for comprehensive overnight evaluation!{self.terminal.RESET}
{self.terminal.YELLOW}Start with 'local' for interactive mode or 'marathon' for complete automation!{self.terminal.RESET}
""")

    def check_disk_space(self):
        """Check and display current disk space usage."""
        import shutil
        print(f"\n{self.terminal.BOLD}Disk Space Information{self.terminal.RESET}")
        
        try:
            total, used, free = shutil.disk_usage(Path.cwd())
            
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)
            used_percent = (used / total) * 100
            
            print(f"Drive: {Path.cwd().drive if hasattr(Path.cwd(), 'drive') else Path.cwd().anchor}")
            print(f"Total: {total_gb:.1f}GB")
            print(f"Used:  {used_gb:.1f}GB ({used_percent:.1f}%)")
            print(f"Free:  {free_gb:.1f}GB")
                
        except Exception as e:
            print(f"{self.terminal.RED}Could not check disk space: {e}{self.terminal.RESET}")

    async def run_troubleshoot(self):
        """Comprehensive troubleshooting and diagnostics."""
        print(f"\n{self.terminal.BOLD}smaLLMs Troubleshooting & Diagnostics{self.terminal.RESET}")
        print(f"Running comprehensive system checks...\n")
        
        issues_found = 0
        suggestions = []
        
        # 1. Check Ollama connectivity
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        print(f"{self.terminal.GREEN}Ollama{self.terminal.RESET}: Connected - {len(models)} models available")
                        
                        # Check for slow/large models
                        large_models = [m for m in models if any(size in m['name'].lower() for size in ['7b', '13b', '70b', 'large'])]
                        if large_models:
                            print(f"   Large models detected: {len(large_models)} (may cause timeouts)")
                    else:
                        print(f"{self.terminal.RED}Ollama{self.terminal.RESET}: HTTP {response.status}")
                        issues_found += 1
                        suggestions.append("Restart Ollama service")
        except asyncio.TimeoutError:
            print(f"{self.terminal.RED}Ollama{self.terminal.RESET}: Connection timeout")
            issues_found += 1
            suggestions.append("Check if Ollama is running: 'ollama serve'")
        except Exception as e:
            print(f"{self.terminal.RED}Ollama{self.terminal.RESET}: {str(e)[:50]}...")
            issues_found += 1
            suggestions.append("Install/restart Ollama")
        
        # 2. Check LM Studio connectivity  
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:1234/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('data', [])
                        print(f"{self.terminal.GREEN}LM Studio{self.terminal.RESET}: Connected - {len(models)} models loaded")
                    else:
                        print(f"{self.terminal.YELLOW}LM Studio{self.terminal.RESET}: Available but no models loaded")
        except:
            print(f"{self.terminal.YELLOW}LM Studio{self.terminal.RESET}: Not running (optional)")
        
        # 3. Check system resources
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print(f"\n{self.terminal.BOLD}System Resources:{self.terminal.RESET}")
            print(f"CPU Usage: {cpu_percent:.1f}%")
            print(f"RAM: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)")
            
            if memory.percent > 90:
                print(f"{self.terminal.RED}Very high RAM usage!{self.terminal.RESET}")
                issues_found += 1
                suggestions.append("Close other applications or restart computer")
            elif memory.percent > 80:
                print(f"{self.terminal.YELLOW}High RAM usage{self.terminal.RESET}")
                suggestions.append("Consider closing other applications")
            
            if memory.total / (1024**3) < 8:
                print(f"{self.terminal.YELLOW}Low RAM system ({memory.total / (1024**3):.1f}GB){self.terminal.RESET}")
                suggestions.append("Use smaller sample counts (10-25) for evaluations")
                
        except ImportError:
            print(f"{self.terminal.YELLOW}Cannot check system resources (psutil not installed){self.terminal.RESET}")
        
        # 4. Check disk space
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        print(f"\n{self.terminal.BOLD}Disk Space:{self.terminal.RESET}")
        print(f"Free: {free_gb:.1f}GB")
        
        if free_gb < 2:
            print(f"{self.terminal.RED}Very low disk space!{self.terminal.RESET}")
            issues_found += 1
            suggestions.append("Free up disk space or run cleanup")
        elif free_gb < 5:
            print(f"{self.terminal.YELLOW}Low disk space{self.terminal.RESET}")
            suggestions.append("Monitor disk usage during evaluations")
        
        # 5. Check configuration
        print(f"\n{self.terminal.BOLD}Configuration:{self.terminal.RESET}")
        timeout = self.config.get('ollama', {}).get('timeout', 300)
        retries = self.config.get('ollama', {}).get('max_retries', 4)
        print(f"Timeout: {timeout}s, Max retries: {retries}")
        
        if timeout < 180:
            print(f"{self.terminal.YELLOW}Short timeout may cause issues with large models{self.terminal.RESET}")
            suggestions.append("Increase timeout in config/config.yaml (recommended: 300s)")
        
        # Summary and recommendations
        print(f"\n{self.terminal.BOLD}Diagnostic Summary:{self.terminal.RESET}")
        if issues_found == 0:
            print(f"{self.terminal.GREEN}No major issues detected!{self.terminal.RESET}")
            print(f"Your system appears ready for evaluations.")
        else:
            print(f"{self.terminal.RED}{issues_found} issues found{self.terminal.RESET}")
        
        if suggestions:
            print(f"\n{self.terminal.BOLD}Recommendations:{self.terminal.RESET}")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        # Timeout-specific guidance
        print(f"\n{self.terminal.CYAN}For timeout issues specifically:{self.terminal.RESET}")
        print(f"   • Start with small evaluations (5 models, 10 samples)")
        print(f"   • Use 'lightning' or 'standard' presets instead of 'all_benchmarks'") 
        print(f"   • Try individual benchmarks (gsm8k, mmlu) before suites")
        print(f"   • Check if models are loaded: 'ollama list'")
        print(f"   • Larger models (3B+) need more time - this is normal")
        
        print(f"\n{self.terminal.GREEN}Quick Fixes:{self.terminal.RESET}")
        print(f"  1. Restart Ollama: 'ollama serve'")
        print(f"  2. Check models: 'ollama list'") 
        print(f"  3. Use smaller evaluations first")
        print(f"  4. Try 'comprehensive_suite' instead of 'all_benchmarks'")

    # ================================
    # LOCAL MODEL EVALUATION MODE
    # ================================
    
    def print_local_mode_menu(self):
        """Print the local model evaluation menu with OpenAI-level benchmarks."""
        self.terminal.clear_screen()
        print(f"""
{self.terminal.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                     {self.terminal.BOLD}LOCAL MODEL EVALUATION MODE{self.terminal.RESET}{self.terminal.CYAN}                            ║
║                     {self.terminal.GRAY}Evaluate your models with OpenAI-level benchmarks{self.terminal.RESET}{self.terminal.CYAN}                        ║
╚══════════════════════════════════════════════════════════════════════════════╝{self.terminal.RESET}

{self.terminal.BOLD}Local Evaluation Options:{self.terminal.RESET}
  {self.terminal.GREEN}all{self.terminal.RESET}     - {self.terminal.BOLD}All Local Models{self.terminal.RESET}      - Evaluate ALL Ollama + LM Studio models
  {self.terminal.GREEN}ollama{self.terminal.RESET}  - {self.terminal.BOLD}Ollama Only{self.terminal.RESET}           - Just your Ollama models  
  {self.terminal.GREEN}lms{self.terminal.RESET}     - {self.terminal.BOLD}LM Studio Only{self.terminal.RESET}        - Just your LM Studio models
  {self.terminal.GREEN}vision{self.terminal.RESET}  - {self.terminal.BOLD}Vision Models{self.terminal.RESET}         - All vision-capable local models

{self.terminal.BOLD}Benchmark Suites Available:{self.terminal.RESET}
  Competition Math: AIME 2024/2025, MATH, GSM8K
  Competitive Programming: Codeforces, HumanEval
  Expert Knowledge: GPQA Diamond, HLE, MMLU
  Specialized: HealthBench, TauBench (tool use)
  OpenAI Suite: All benchmarks used by OpenAI o3/o4

{self.terminal.BOLD}Local Commands:{self.terminal.RESET}
  {self.terminal.CYAN}discover{self.terminal.RESET} - Find all local models on your system
  {self.terminal.CYAN}benchmarks{self.terminal.RESET} - List all available benchmarks  
  {self.terminal.CYAN}status{self.terminal.RESET}   - Show current evaluation results
  {self.terminal.CYAN}space{self.terminal.RESET}    - Check disk space (important for local evaluation!)
  {self.terminal.CYAN}troubleshoot{self.terminal.RESET} - Diagnose and fix timeout/connection issues  
  {self.terminal.CYAN}back{self.terminal.RESET}     - Return to main menu
  {self.terminal.CYAN}exit{self.terminal.RESET}     - Quit application

{self.terminal.YELLOW} NEW: Enhanced timeout handling and checkpoint/resume system!{self.terminal.RESET}
{self.terminal.YELLOW} If you had timeout issues, run 'troubleshoot' for fixes{self.terminal.RESET}
""")
    

    async def run_local_mode(self):
        """Run the local model evaluation mode."""
        self.print_local_mode_menu()
        
        while True:
            try:
                cmd = input(f"\n{self.terminal.BOLD}local{self.terminal.RESET} {self.terminal.GRAY}${self.terminal.RESET} ").strip().lower()
                
                if cmd in ['exit', 'quit', 'q']:
                    return
                elif cmd == 'back':
                    return
                elif cmd == 'help':
                    self.print_local_mode_menu()
                elif cmd == 'discover':
                    await self.discover_local_models_interactive()
                elif cmd in ['all', 'everything']:
                    success = await self.run_local_evaluation("all", include_vision=True)
                    if success:
                        print(f"\n{self.terminal.CYAN} Run 'status' to see results or 'export' to create website files{self.terminal.RESET}")
                elif cmd == 'ollama':
                    success = await self.run_local_evaluation("ollama", include_vision=True)
                    if success:
                        print(f"\n{self.terminal.CYAN} Run 'status' to see results or 'export' to create website files{self.terminal.RESET}")
                elif cmd in ['lms', 'lm_studio', 'lmstudio']:
                    success = await self.run_local_evaluation("lm_studio", include_vision=True)
                    if success:
                        print(f"\n{self.terminal.CYAN} Run 'status' to see results or 'export' to create website files{self.terminal.RESET}")
                elif cmd == 'vision':
                    success = await self.run_local_evaluation("vision", include_vision=True)
                    if success:
                        print(f"\n{self.terminal.CYAN} Run 'status' to see results or 'export' to create website files{self.terminal.RESET}")
                elif cmd in ['marathon', 'marathon_mode', 'overnight']:
                    success = await self.run_marathon_mode()
                    if success:
                        print(f"\n{self.terminal.GREEN} Marathon Mode completed! Check 'status' for comprehensive results{self.terminal.RESET}")
                elif cmd == 'status':
                    self.show_status()
                elif cmd == 'space':
                    self.check_disk_space()
                elif cmd in ['benchmarks', 'bench']:
                    self.show_available_benchmarks()
                elif cmd in ['troubleshoot', 'trouble', 'fix', 'debug']:
                    await self.run_troubleshoot()
                else:
                    print(f"{self.terminal.YELLOW} Unknown command: {cmd}{self.terminal.RESET}")
                    print(f"Type 'discover' to find models, 'troubleshoot' to fix issues, or 'help' for options")
                    
            except KeyboardInterrupt:
                print(f"\n{self.terminal.YELLOW}Use 'exit' to return to main menu{self.terminal.RESET}")
            except Exception as e:
                print(f"{self.terminal.RED} Error: {e}{self.terminal.RESET}")

    async def run_marathon_mode(self):
        """Marathon Mode: Run ALL local models on ALL benchmarks - perfect for overnight runs."""
        print(f"\n{self.terminal.RED} MARATHON MODE - Complete Evaluation Suite{self.terminal.RESET}")
        print(f"This mode will automatically discover ALL your local models and run them on ALL benchmarks.")
        print(f"Perfect for overnight runs - just start it and let your laptop work while you sleep!\n")
        
        try:
            # Auto-discover all local models
            model_manager = await self.init_model_manager()
            all_models = await model_manager.get_all_local_models()
            
            if not all_models:
                print(f"{self.terminal.YELLOW} No local models found. Make sure Ollama/LM Studio are running.{self.terminal.RESET}")
                return False
            
            model_names = [m['name'] for m in all_models]
            
            # Calculate total evaluation scope
            total_benchmarks = 14  # All individual benchmarks
            samples_per_evaluation = 50  # Good balance for overnight
            total_evaluations = len(model_names) * total_benchmarks * samples_per_evaluation
            estimated_hours = total_evaluations * 0.02 / 60  # Conservative estimate
            
            print(f"{self.terminal.BOLD} Marathon Mode Configuration:{self.terminal.RESET}")
            print(f"   Models discovered: {len(model_names)}")
            print(f"    Benchmarks: {total_benchmarks} (all OpenAI-level benchmarks)")
            print(f"    Samples per evaluation: {samples_per_evaluation}")
            print(f"    Total evaluations: {total_evaluations:,}")
            print(f"     Estimated duration: {estimated_hours:.1f} hours")
            
            print(f"\n{self.terminal.CYAN} Models to be evaluated:{self.terminal.RESET}")
            for i, model in enumerate(model_names, 1):
                provider = next((m['provider'] for m in all_models if m['name'] == model), 'unknown')
                print(f"   {i:2d}. {model} ({provider})")
            
            print(f"\n{self.terminal.CYAN} Benchmarks to be run:{self.terminal.RESET}")
            benchmarks = ['mmlu', 'gsm8k', 'math', 'humaneval', 'aime_2024', 'aime_2025', 
                         'gpqa_diamond', 'hle', 'healthbench', 'healthbench_hard', 
                         'codeforces', 'tau_retail', 'tau_general', 'arc']
            for i, bench in enumerate(benchmarks, 1):
                print(f"   {i:2d}. {bench}")
            
            print(f"\n{self.terminal.GREEN} Marathon Mode Features:{self.terminal.RESET}")
            print(f"    Automatic checkpoint/resume (interrupt-safe)")
            print(f"    Resource management (pauses between models)")
            print(f"    Continuous progress logging")
            print(f"    Results saved after each benchmark")
            print(f"    Comprehensive final report")
            
            confirm = input(f"\n{self.terminal.YELLOW}Start Marathon Mode? This will run for ~{estimated_hours:.1f} hours [y/N]:{self.terminal.RESET} ").strip().lower()
            if confirm not in ['y', 'yes']:
                print(f"{self.terminal.GREEN}Marathon Mode cancelled.{self.terminal.RESET}")
                return False
            
            print(f"\n{self.terminal.GREEN} Starting Marathon Mode...{self.terminal.RESET}")
            print(f" Perfect for overnight runs! You can safely interrupt and resume anytime.")
            print(f" Results are continuously saved to evaluation_sessions/")
            
            # Create marathon configuration
            config_dict = {
                'models': model_names,
                'samples': samples_per_evaluation,
                'benchmark': 'all_benchmarks',
                'name': f"Marathon Mode ({len(model_names)} models × {total_benchmarks} benchmarks)",
                'preset': 'marathon_mode',
                'evaluation_mode': 'local',
                'marathon_mode': True
            }
            
            # Run the marathon evaluation
            result = await self.run_evaluation_with_display_async(config_dict)
            
            if result:
                print(f"\n{self.terminal.GREEN} MARATHON MODE COMPLETED!{self.terminal.RESET}")
                print(f" Comprehensive evaluation of {len(model_names)} models completed")
                print(f" Run 'export' to create website files with all results")
            
            return result
            
        except Exception as e:
            print(f"{self.terminal.RED} Marathon Mode failed: {e}{self.terminal.RESET}")
            return False

    # ================================  
    # CLOUD MODEL EVALUATION MODE
    # ================================
    
    def print_cloud_mode_menu(self):
        """Print the cloud model evaluation menu with OpenAI-level benchmarks."""
        self.terminal.clear_screen()
        print(f"""
{self.terminal.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                     {self.terminal.BOLD}  CLOUD MODEL EVALUATION MODE{self.terminal.RESET}{self.terminal.CYAN}                           ║
║                     {self.terminal.GRAY}Evaluate HuggingFace models with OpenAI-level benchmarks{self.terminal.RESET}{self.terminal.CYAN}                  ║
╚══════════════════════════════════════════════════════════════════════════════╝{self.terminal.RESET}

{self.terminal.BOLD} Cloud Evaluation Presets:{self.terminal.RESET}
  {self.terminal.GREEN}1{self.terminal.RESET} - {self.terminal.BOLD}Lightning Demo{self.terminal.RESET}     - 3 models, 5 samples (~1 min, ~$0.02)
  {self.terminal.GREEN}2{self.terminal.RESET} - {self.terminal.BOLD}Quick Benchmark{self.terminal.RESET}    - 5 models, 25 samples (~3 min, ~$0.05)  
  {self.terminal.GREEN}3{self.terminal.RESET} - {self.terminal.BOLD}Standard Eval{self.terminal.RESET}      - 8 models, 50 samples (~10 min, ~$0.15)
  {self.terminal.GREEN}4{self.terminal.RESET} - {self.terminal.BOLD}OpenAI-Level{self.terminal.RESET}       - 12 models, 100+ samples (~60 min, ~$1.50)
  {self.terminal.GREEN}5{self.terminal.RESET} - {self.terminal.BOLD}Competition Suite{self.terminal.RESET}  - AIME + Codeforces + MATH (~45 min, ~$1.00)
  {self.terminal.GREEN}6{self.terminal.RESET} - {self.terminal.BOLD}Expert Suite{self.terminal.RESET}       - GPQA + HLE + HealthBench (~30 min, ~$0.80)
  {self.terminal.GREEN}7{self.terminal.RESET} - {self.terminal.BOLD}Custom Selection{self.terminal.RESET}   - Choose your own models & benchmarks

{self.terminal.BOLD} Available Benchmarks:{self.terminal.RESET}
   {self.terminal.CYAN}Math & Competition:{self.terminal.RESET} GSM8K, MATH, AIME 2024/2025
   {self.terminal.CYAN}Knowledge & Science:{self.terminal.RESET} MMLU, GPQA Diamond, HLE
   {self.terminal.CYAN}Code & Programming:{self.terminal.RESET} HumanEval, Codeforces  
   {self.terminal.CYAN}Specialized:{self.terminal.RESET} HealthBench, TauBench (tool use)

{self.terminal.BOLD} Cloud Commands:{self.terminal.RESET}
  {self.terminal.CYAN}models{self.terminal.RESET}   - Show available HuggingFace models
  {self.terminal.CYAN}benchmarks{self.terminal.RESET} - List all available benchmarks
  {self.terminal.CYAN}status{self.terminal.RESET}   - Show current evaluation results
  {self.terminal.CYAN}config{self.terminal.RESET}   - Check HuggingFace API configuration
  {self.terminal.CYAN}back{self.terminal.RESET}     - Return to main menu
  {self.terminal.CYAN}exit{self.terminal.RESET}     - Quit application

{self.terminal.BOLD} Cost Information:{self.terminal.RESET}
  • Lightning Demo: ~$0.02 (perfect for testing)
  • OpenAI-Level: ~$1.50 (matches OpenAI's evaluation rigor)
  • Competition Suite: ~$1.00 (AIME, Codeforces, MATH like top labs)

{self.terminal.YELLOW} Now supporting ALL benchmarks used by OpenAI, Anthropic, Google DeepMind!{self.terminal.RESET}
{self.terminal.YELLOW} Make sure your HuggingFace token is set in config/config.yaml{self.terminal.RESET}
""")
    

    def run_cloud_mode(self):
        """Run the cloud model evaluation mode."""
        self.print_cloud_mode_menu()
        
        while True:
            try:
                cmd = input(f"\n{self.terminal.BOLD}cloud{self.terminal.RESET} {self.terminal.GRAY}${self.terminal.RESET} ").strip()
                
                if cmd.lower() in ['exit', 'quit', 'q']:
                    return
                elif cmd.lower() == 'back':
                    return  
                elif cmd.lower() == 'help':
                    self.print_cloud_mode_menu()
                elif cmd == '1':
                    self.run_cloud_preset('lightning')
                elif cmd == '2':
                    self.run_cloud_preset('quick')
                elif cmd == '3':
                    self.run_cloud_preset('standard')
                elif cmd == '4':
                    # OpenAI-Level evaluation
                    self.run_cloud_preset('openai_level')
                elif cmd == '5':
                    # Competition Suite
                    self.run_cloud_preset('competition')
                elif cmd == '6':
                    # Expert Suite
                    self.run_cloud_preset('expert')
                elif cmd == '7':
                    # Custom Selection
                    config_dict = asyncio.run(self.interactive_cloud_model_selection())
                    if config_dict:
                        config_dict['preset'] = 'custom'
                        success = self.run_evaluation_with_display(config_dict)
                        if success:
                            print(f"\n{self.terminal.CYAN} Run 'export' to create website files{self.terminal.RESET}")
                elif cmd.lower() == 'benchmarks':
                    self.show_available_benchmarks()
                elif cmd.lower() == 'models':
                    self.show_available_cloud_models()
                elif cmd.lower() == 'config':
                    self.check_cloud_config()
                elif cmd.lower() == 'status':
                    self.show_status()
                else:
                    print(f"{self.terminal.YELLOW} Unknown command: {cmd}{self.terminal.RESET}")
                    print(f"Type 'help' for options or try presets 1-5")
                    
            except KeyboardInterrupt:
                print(f"\n{self.terminal.YELLOW}Use 'exit' to return to main menu{self.terminal.RESET}")
            except Exception as e:
                print(f"{self.terminal.RED} Error: {e}{self.terminal.RESET}")

    def run_cloud_preset(self, preset_name: str):
        """Run a cloud evaluation preset with enhanced benchmark support."""
        try:
            config = self.get_preset_config(preset_name)
            if not config:
                print(f"{self.terminal.RED} Preset '{preset_name}' not found{self.terminal.RESET}")
                return
                
            config_dict = {
                'models': config['models'],
                'samples': config['samples'],
                'benchmark': config.get('benchmark', 'gsm8k'),
                'name': config['name'],
                'preset': preset_name
            }
            
            print(f"\n{self.terminal.BOLD} Starting {config['name']}{self.terminal.RESET}")
            print(f" {config['description']}")
            print(f" Models: {len(config['models'])}")
            print(f" Samples: {config['samples']}")
            print(f" Benchmarks: {config.get('benchmark', 'gsm8k')}")
            
            # Show estimated time and cost
            est_time_mins = len(config['models']) * config['samples'] * 0.05  # rough estimate
            est_cost = len(config['models']) * config['samples'] * 0.001  # rough estimate
            print(f" Estimated time: ~{est_time_mins:.0f} minutes")
            print(f" Estimated cost: ~${est_cost:.2f}")
            
            confirm = input(f"\n{self.terminal.CYAN}Continue with this evaluation? [Y/n]:{self.terminal.RESET} ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                success = self.run_evaluation_with_display(config_dict)
                if success:
                    print(f"\n{self.terminal.GREEN} {config['name']} completed successfully!{self.terminal.RESET}")
                    print(f"{self.terminal.CYAN} Run 'export' to create website files{self.terminal.RESET}")
            else:
                print(f"{self.terminal.YELLOW}Evaluation cancelled{self.terminal.RESET}")
                
        except Exception as e:
            print(f"{self.terminal.RED} Preset evaluation failed: {e}{self.terminal.RESET}")

    def show_available_benchmarks(self):
        """Show all available benchmarks with descriptions."""
        print(f"\n{self.terminal.BOLD} Available Benchmarks (OpenAI-Level Suite){self.terminal.RESET}")
        
        benchmarks = {
            "Core Academic": {
                "gsm8k": "Grade School Math 8K - Math word problems",
                "mmlu": "Massive Multitask Language Understanding - Multiple choice knowledge",
                "math": "MATH - Competition-level mathematics problems", 
                "humaneval": "HumanEval - Python code generation and completion"
            },
            "Competition & Expert": {
                "aime_2024": "AIME 2024 - American Invitational Mathematics Examination",
                "aime_2025": "AIME 2025 - Latest math competition problems",
                "gpqa_diamond": "GPQA Diamond - PhD-level science questions",
                "hle": "Humanity's Last Exam - Expert-level cross-domain questions",
                "codeforces": "Codeforces - Competitive programming problems"
            },
            "Specialized": {
                "healthbench": "HealthBench - Medical conversation safety and accuracy",
                "healthbench_hard": "HealthBench Hard - Challenging medical scenarios",
                "tau_retail": "TauBench Retail - Function calling and tool use",
                "tau_general": "TauBench General - General tool use scenarios"
            },
            "Evaluation Suites": {
                "openai_suite": "All benchmarks used by OpenAI (11 benchmarks)",
                "competition_suite": "Math and programming competitions (AIME + Codeforces + MATH)",
                "expert_suite": "Expert-level knowledge (GPQA + HLE + HealthBench)",
                "comprehensive_suite": "Best overall coverage (6 core benchmarks)",
                "legacy_suite": "Original smaLLMs (GSM8K + MMLU)"
            }
        }
        
        for category, bench_dict in benchmarks.items():
            print(f"\n{self.terminal.BOLD}{category}:{self.terminal.RESET}")
            for bench_key, description in bench_dict.items():
                difficulty_icon = "" if "expert" in description.lower() or "competition" in description.lower() else ""
                print(f"  {difficulty_icon} {self.terminal.GREEN}{bench_key:<20}{self.terminal.RESET} - {description}")
        
        print(f"\n{self.terminal.YELLOW} OpenAI Suite matches the evaluation rigor of o3/o4 models{self.terminal.RESET}")
        print(f"{self.terminal.YELLOW} Competition Suite focuses on mathematical and programming challenges{self.terminal.RESET}")
        print(f"{self.terminal.YELLOW} Expert Suite tests advanced domain knowledge{self.terminal.RESET}")

    def show_available_cloud_models(self):
        """Show available cloud models from external config."""
        print(f"\n{self.terminal.BOLD}  Available HuggingFace Models{self.terminal.RESET}")
        
        external_models = self.config.get('external_models', {})
        huggingface_models = external_models.get('huggingface_models', {})
        
        if not huggingface_models:
            print(f"{self.terminal.YELLOW}No models configured. Edit config/models.yaml to add models.{self.terminal.RESET}")
            return
            
        for category, models in huggingface_models.items():
            if isinstance(models, list) and models:
                print(f"\n{self.terminal.BOLD}{category.upper()} Models:{self.terminal.RESET}")
                for model in models[:5]:  # Show first 5 of each category
                    print(f"  • {model}")
                if len(models) > 5:
                    print(f"  ... and {len(models) - 5} more")

    def check_cloud_config(self):
        """Check HuggingFace API configuration."""
        print(f"\n{self.terminal.BOLD} Cloud Configuration Status{self.terminal.RESET}")
        
        hf_config = self.config.get('huggingface', {})
        token = hf_config.get('token', '')
        
        if token and token != 'YOUR_HF_TOKEN_HERE':
            print(f"{self.terminal.GREEN} HuggingFace token configured{self.terminal.RESET}")
        else:
            print(f"{self.terminal.RED} HuggingFace token not configured{self.terminal.RESET}")
            print(f"   Edit config/config.yaml and add your token from https://huggingface.co/settings/tokens")
            
        pro_features = hf_config.get('use_pro_features', False)
        print(f" Pro features: {'Enabled' if pro_features else 'Disabled'}")
        
        # Check model configuration
        external_models = self.config.get('external_models', {})
        if external_models:
            model_count = sum(len(models) for models in external_models.get('huggingface_models', {}).values() if isinstance(models, list))
            print(f" Configured models: {model_count}")
        else:
            print(f"{self.terminal.YELLOW}  No external models configured in config/models.yaml{self.terminal.RESET}")

    async def interactive_cloud_model_selection(self) -> Dict[str, Any]:
        """Interactive cloud model selection."""
        print(f"\n{self.terminal.BOLD} Custom Cloud Evaluation Setup{self.terminal.RESET}")
        
        # Get available cloud models
        cloud_models = self.get_cloud_models_list()
        
        if not cloud_models:
            print(f"{self.terminal.RED} No cloud models available. Check config/models.yaml{self.terminal.RESET}")
            return {}
            
        print(f"\n{self.terminal.BOLD} Model Selection Options:{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}a{self.terminal.RESET} - All available models ({len(cloud_models)} models)")
        print(f"  {self.terminal.GREEN}c{self.terminal.RESET} - Choose specific models manually")
        
        while True:
            choice = input(f"\n{self.terminal.CYAN}Select models [a/c]:{self.terminal.RESET} ").strip().lower()
            
            if choice == 'a':
                models = cloud_models
                print(f" Selected all {len(models)} available cloud models")
                break
            elif choice == 'c':
                models = await self._manual_model_selection(cloud_models)
                break
            else:
                print(f"{self.terminal.YELLOW}Please enter 'a' or 'c'{self.terminal.RESET}")
        
        if not models:
            return {}
        
        # Sample count and benchmark selection (reuse existing methods)
        samples = self._get_cloud_sample_count_interactive()
        benchmark = self._get_benchmark_interactive()
        
        return {
            'models': models,
            'samples': samples, 
            'benchmark': benchmark,
            'name': f"Custom Cloud ({len(models)} models, {samples} samples)"
        }

    def get_cloud_models_list(self) -> List[str]:
        """Get list of all available cloud models from external config."""
        cloud_models = []
        external_models = self.config.get('external_models', {})
        huggingface_models = external_models.get('huggingface_models', {})
        
        for category, models in huggingface_models.items():
            if isinstance(models, list):
                cloud_models.extend(models)
                
        # Fallback to basic models if no external config
        if not cloud_models:
            cloud_models = [
                "google/gemma-2-2b-it",
                "Qwen/Qwen2.5-1.5B-Instruct", 
                "meta-llama/Llama-3.2-1B-Instruct",
            ]
            
        return cloud_models

    def _get_cloud_sample_count_interactive(self) -> int:
        """Get sample count for cloud evaluation with cost estimates."""
        print(f"\n{self.terminal.BOLD} Sample Count (cloud evaluation with cost estimates):{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}1{self.terminal.RESET} - Lightning: 5 samples (~$0.005 per model) {self.terminal.GRAY}Fast test{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}2{self.terminal.RESET} - Quick Test: 10 samples (~$0.01 per model) {self.terminal.GRAY}Basic validation{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}3{self.terminal.RESET} - Light Eval: 25 samples (~$0.03 per model) {self.terminal.GRAY}Good estimate{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}4{self.terminal.RESET} - Standard: 50 samples (~$0.05 per model) {self.terminal.GRAY}Reliable results{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}5{self.terminal.RESET} - Thorough: 100 samples (~$0.10 per model) {self.terminal.GRAY}High confidence{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}6{self.terminal.RESET} - OpenAI-level: 500+ samples (~$0.50+ per model) {self.terminal.GRAY}Publication-ready{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}7{self.terminal.RESET} - Custom amount")
        
        print(f"\n{self.terminal.YELLOW} For OpenAI-level benchmarks (AIME, GPQA), recommend 100+ samples{self.terminal.RESET}")
        
        while True:
            choice = input(f"\n{self.terminal.CYAN}Select sample count [1-7]:{self.terminal.RESET} ").strip()
            if choice == '1':
                return 5
            elif choice == '2':
                return 10
            elif choice == '3':
                return 25
            elif choice == '4':
                return 50
            elif choice == '5':
                return 100
            elif choice == '6':
                return 500
            elif choice == '7':
                try:
                    return int(input(f"{self.terminal.CYAN}Enter custom sample count:{self.terminal.RESET} "))
                except ValueError:
                    print(f"{self.terminal.YELLOW}Please enter a valid number{self.terminal.RESET}")
            else:
                print(f"{self.terminal.YELLOW}Please enter 1-7{self.terminal.RESET}")
    
    def get_preset_config(self, preset: str) -> Dict[str, Any]:
        """Get preset evaluation configurations - Using external model configuration."""
        # First try to load from external models.yaml
        external_models = self.config.get('external_models', {})
        external_presets = external_models.get('presets', {})
        
        if preset in external_presets:
            return external_presets[preset]
        
        # Fallback to basic verified models if external config not available
        basic_models = [
            "google/gemma-2-2b-it",           # CONFIRMED WORKING
            "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
            "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
        ]
        
        presets = {
            "lightning": {
                "name": "Lightning Demo",
                "models": basic_models,
                "samples": 5,
                "description": "Ultra-fast 1-minute demo with 3 confirmed working models",
                "benchmark": "gsm8k"
            },
            "safe": {
                "name": "Safe Mode (Slow Laptops)",
                "models": basic_models,
                "samples": 10,
                "description": "Slow laptop friendly - only reliable benchmarks, minimal samples",
                "benchmark": "safe_suite"
            },
            "quick": {
                "name": "Quick Benchmark", 
                "models": basic_models,
                "samples": 25,
                "description": "3-minute evaluation with 3 reliable instruct models",
                "benchmark": "gsm8k"
            },
            "standard": {
                "name": "Standard Evaluation",
                "models": basic_models,
                "samples": 50,
                "description": "10-minute comprehensive test with 3 working models",
                "benchmark": "comprehensive_suite"
            },
            "openai_level": {
                "name": "OpenAI-Level Evaluation",
                "models": basic_models,
                "samples": 100,
                "description": "60-minute evaluation matching OpenAI's rigor",
                "benchmark": "openai_suite"
            },
            "competition": {
                "name": "Competition Suite",
                "models": basic_models,
                "samples": 50,
                "description": "45-minute math and programming competition evaluation",
                "benchmark": "competition_suite"
            },
            "expert": {
                "name": "Expert Suite",
                "models": basic_models,
                "samples": 75,
                "description": "30-minute expert-level knowledge evaluation",
                "benchmark": "expert_suite"
            }
        }
        return presets.get(preset, {})
    
    async def discover_local_models_interactive(self):
        """Interactive local model discovery and display."""
        print(f"\n{self.terminal.BOLD} Discovering Local Models...{self.terminal.RESET}")
        
        try:
            model_manager = await self.init_model_manager()
            discovered = await model_manager.discover_local_models()
            
            total_models = sum(len(models) for models in discovered.values())
            
            if total_models == 0:
                print(f"{self.terminal.YELLOW} No local models found{self.terminal.RESET}")
                print("Make sure Ollama and/or LM Studio are running with models loaded")
                return
            
            print(f"\n{self.terminal.GREEN} Found {total_models} local models:{self.terminal.RESET}\n")
            
            # Display by provider
            for provider, models in discovered.items():
                if models:
                    print(f"{self.terminal.BOLD}{provider.upper()}:{self.terminal.RESET}")
                    for model in models:
                        vision_icon = "" if model.get('supports_vision', False) else ""
                        size_info = f"({model.get('size_gb', 0):.1f}GB)" if model.get('size_gb', 0) > 0 else ""
                        print(f"  {vision_icon} {model['name']} {size_info}")
                    print()
            
            # Show summary by type
            text_models = [m for models in discovered.values() for m in models if not m.get('supports_vision', False)]
            vision_models = [m for models in discovered.values() for m in models if m.get('supports_vision', False)]
            
            print(f"{self.terminal.CYAN}Summary:{self.terminal.RESET}")
            print(f"   Text models: {len(text_models)}")
            print(f"   Vision models: {len(vision_models)}")
            print(f"   Total: {total_models}")
            
        except Exception as e:
            print(f"{self.terminal.RED} Discovery failed: {e}{self.terminal.RESET}")
    
    async def run_local_evaluation(self, provider_filter: str = "all", include_vision: bool = True):
        """Run evaluation on local models."""
        print(f"\n{self.terminal.BOLD} Starting Local Model Evaluation{self.terminal.RESET}")
        
        try:
            model_manager = await self.init_model_manager()
            all_models = await model_manager.get_all_local_models()
            
            if not all_models:
                print(f"{self.terminal.YELLOW} No local models found. Run 'discover' first.{self.terminal.RESET}")
                return False
            
            # Filter models by provider
            if provider_filter == "ollama":
                models = [m for m in all_models if m['provider'] == 'ollama']
            elif provider_filter == "lm_studio":
                models = [m for m in all_models if m['provider'] == 'lm_studio']
            elif provider_filter == "vision":
                models = [m for m in all_models if m.get('supports_vision', False)]
            else:
                models = all_models
            
            if not include_vision:
                models = [m for m in models if not m.get('supports_vision', False)]
            
            if not models:
                print(f"{self.terminal.YELLOW} No models match the filter criteria{self.terminal.RESET}")
                return False
            
            print(f"Found {len(models)} models to evaluate")
            
            # Ask for evaluation settings
            samples = self._get_sample_count_interactive()
            benchmark = self._get_benchmark_interactive()
            
            # Handle marathon mode - auto-configure for maximum coverage
            if benchmark == 'marathon_mode':
                print(f"\n{self.terminal.RED} MARATHON MODE ACTIVATED!{self.terminal.RESET}")
                print(f"This will run ALL {len(models)} local models on ALL benchmarks with 50 samples each.")
                
                # Calculate total evaluations
                total_benchmarks = 14  # All individual benchmarks
                total_evaluations = len(models) * total_benchmarks * 50
                estimated_hours = total_evaluations * 0.02 / 60  # 1.2 seconds per evaluation
                
                print(f"\n{self.terminal.YELLOW} Marathon Mode Statistics:{self.terminal.RESET}")
                print(f"   • Models: {len(models)}")
                print(f"   • Benchmarks: {total_benchmarks} (all individual benchmarks)")
                print(f"   • Samples per evaluation: 50")
                print(f"   • Total evaluations: {total_evaluations:,}")
                print(f"   • Estimated duration: {estimated_hours:.1f} hours")
                
                print(f"\n{self.terminal.CYAN} Marathon Mode Features:{self.terminal.RESET}")
                print(f"    Auto-discovery of all local models")
                print(f"    Automatic checkpoint/resume if interrupted")
                print(f"    Resource management (pauses between models)")
                print(f"    Detailed progress logging")
                print(f"    Final comprehensive report with all results")
                
                print(f"\n{self.terminal.BOLD}Perfect for overnight runs - just start it and let it work!{self.terminal.RESET}")
                
                confirm = input(f"\n{self.terminal.YELLOW}Start Marathon Mode evaluation? [y/N]:{self.terminal.RESET} ").strip().lower()
                if confirm not in ['y', 'yes']:
                    print(f"{self.terminal.GREEN}Marathon Mode cancelled.{self.terminal.RESET}")
                    return False
                
                # Auto-set parameters for marathon mode
                samples = 50  # Good balance for overnight run
                benchmark = 'all_benchmarks'  # All individual benchmarks
                
                print(f"\n{self.terminal.GREEN} Starting Marathon Mode evaluation...{self.terminal.RESET}")
                print(f" This will take approximately {estimated_hours:.1f} hours. Perfect for overnight!")
                print(f" Progress will be saved continuously - you can interrupt and resume anytime.")
            
            # Convert to model names for evaluation
            model_names = [m['name'] for m in models]
            
            config_dict = {
                'models': model_names,
                'samples': samples,
                'benchmark': benchmark,
                'name': f"Local Evaluation ({provider_filter.title()}, {len(models)} models)",
                'preset': f'local_{provider_filter}',
                'evaluation_mode': 'local',
                'marathon_mode': benchmark == 'all_benchmarks' and samples == 50
            }
            
            result = await self.run_evaluation_with_display_async(config_dict)
            return result
            
        except Exception as e:
            print(f"{self.terminal.RED} Local evaluation failed: {e}{self.terminal.RESET}")
            return False
    
    def _get_sample_count_interactive(self) -> int:
        """Get sample count interactively with timing estimates."""
        print(f"\n{self.terminal.BOLD} Sample Count (local models with timing estimates):{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}1{self.terminal.RESET} - Lightning: 3 samples (~15 sec per model) {self.terminal.GRAY}Ultra-fast test{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}2{self.terminal.RESET} - Quick Test: 5 samples (~30 sec per model) {self.terminal.GRAY}Basic check{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}3{self.terminal.RESET} - Light Eval: 10 samples (~1 min per model) {self.terminal.GRAY}Good estimate{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}4{self.terminal.RESET} - Standard: 25 samples (~3 min per model) {self.terminal.GRAY}Reliable results{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}5{self.terminal.RESET} - Thorough: 50 samples (~6 min per model) {self.terminal.GRAY}High confidence{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}6{self.terminal.RESET} - OpenAI-level: 100+ samples (~12+ min per model) {self.terminal.GRAY}Publication-ready{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}7{self.terminal.RESET} - Custom amount")
        
        print(f"\n{self.terminal.YELLOW} For competition benchmarks (AIME, Codeforces), recommend 25+ samples{self.terminal.RESET}")
        
        while True:
            choice = input(f"\n{self.terminal.CYAN}Select sample count [1-7]:{self.terminal.RESET} ").strip()
            if choice == '1':
                return 3
            elif choice == '2':
                return 5
            elif choice == '3':
                return 10
            elif choice == '4':
                return 25
            elif choice == '5':
                return 50
            elif choice == '6':
                return 100
            elif choice == '7':
                try:
                    count = int(input(f"{self.terminal.CYAN}Enter custom sample count:{self.terminal.RESET} "))
                    if count <= 0:
                        print(f"{self.terminal.YELLOW}Please enter a positive number{self.terminal.RESET}")
                        continue
                    return count
                except ValueError:
                    print(f"{self.terminal.YELLOW}Please enter a valid number{self.terminal.RESET}")
            else:
                print(f"{self.terminal.YELLOW}Please enter 1-7{self.terminal.RESET}")
    
    def _get_benchmark_interactive(self) -> str:
        """Get benchmark selection interactively with OpenAI-level options."""
        print(f"\n{self.terminal.BOLD} Benchmark Selection (OpenAI-Level Evaluation):{self.terminal.RESET}")
        print(f"\n{self.terminal.BOLD} Standard Benchmarks:{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}1{self.terminal.RESET} - GSM8K: Math word problems")
        print(f"  {self.terminal.GREEN}2{self.terminal.RESET} - MMLU: Multiple choice knowledge")
        print(f"  {self.terminal.GREEN}3{self.terminal.RESET} - HumanEval: Code generation")
        print(f"  {self.terminal.GREEN}4{self.terminal.RESET} - MATH: Competition mathematics")
        
        print(f"\n{self.terminal.BOLD} Competition & Expert Level:{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}5{self.terminal.RESET} - AIME 2024: Math competition (like OpenAI)")
        print(f"  {self.terminal.GREEN}6{self.terminal.RESET} - AIME 2025: Latest math competition")
        print(f"  {self.terminal.GREEN}7{self.terminal.RESET} - GPQA Diamond: PhD-level science")
        print(f"  {self.terminal.GREEN}8{self.terminal.RESET} - Codeforces: Competitive programming")
        
        print(f"\n{self.terminal.BOLD} Expert Knowledge:{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}9{self.terminal.RESET} - HLE: Humanity's Last Exam (expert-level)")
        print(f"  {self.terminal.GREEN}10{self.terminal.RESET} - HealthBench: Medical conversations")
        print(f"  {self.terminal.GREEN}11{self.terminal.RESET} - TauBench: Function calling & tool use")
        
        print(f"\n{self.terminal.BOLD} Evaluation Suites:{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}12{self.terminal.RESET} - OpenAI Suite: All benchmarks used by OpenAI")
        print(f"  {self.terminal.GREEN}13{self.terminal.RESET} - Competition Suite: AIME + Codeforces + MATH")
        print(f"  {self.terminal.GREEN}14{self.terminal.RESET} - Expert Suite: GPQA + HLE + HealthBench")
        print(f"  {self.terminal.GREEN}15{self.terminal.RESET} - Comprehensive: Best overall coverage")
        print(f"  {self.terminal.GREEN}16{self.terminal.RESET} - Legacy (GSM8K + MMLU): Original benchmarks")
        print(f"  {self.terminal.GREEN}17{self.terminal.RESET} - ALL BENCHMARKS: Every single benchmark available")
        print(f"  {self.terminal.GREEN}18{self.terminal.RESET} -  MARATHON MODE: All models + All benchmarks (overnight run)")
        
        while True:
            choice = input(f"\n{self.terminal.CYAN}Select benchmark [1-18]:{self.terminal.RESET} ").strip()
            
            benchmark_map = {
                '1': 'gsm8k',
                '2': 'mmlu', 
                '3': 'humaneval',
                '4': 'math',
                '5': 'aime_2024',
                '6': 'aime_2025', 
                '7': 'gpqa_diamond',
                '8': 'codeforces',
                '9': 'hle',
                '10': 'healthbench',
                '11': 'tau_retail',
                '12': 'openai_suite',
                '13': 'competition_suite',
                '14': 'expert_suite', 
                '15': 'comprehensive_suite',
                '16': 'legacy_suite',
                '17': 'all_benchmarks',
                '18': 'marathon_mode'
            }
            
            if choice in benchmark_map:
                selection = benchmark_map[choice]
                
                # Show what's included in suites
                if selection == 'openai_suite':
                    print(f"  {self.terminal.GRAY}→ MMLU, GSM8K, MATH, HumanEval, AIME 2024/2025, GPQA Diamond, HLE, HealthBench, Codeforces, TauBench{self.terminal.RESET}")
                elif selection == 'competition_suite':
                    print(f"  {self.terminal.GRAY}→ AIME 2024, AIME 2025, Codeforces, MATH{self.terminal.RESET}")
                elif selection == 'expert_suite':
                    print(f"  {self.terminal.GRAY}→ GPQA Diamond, Humanity's Last Exam, HealthBench Hard{self.terminal.RESET}")
                elif selection == 'comprehensive_suite':
                    print(f"  {self.terminal.GRAY}→ MMLU, GSM8K, MATH, HumanEval, GPQA Diamond, HealthBench{self.terminal.RESET}")
                elif selection == 'legacy_suite':
                    print(f"  {self.terminal.GRAY}→ GSM8K + MMLU (original smaLLMs benchmarks){self.terminal.RESET}")
                elif selection == 'all_benchmarks':
                    print(f"  {self.terminal.GRAY}→ ALL 14 INDIVIDUAL BENCHMARKS: MMLU, GSM8K, MATH, HumanEval, AIME 2024, AIME 2025, GPQA Diamond, GPQA Main, HLE, HealthBench, HealthBench Hard, Codeforces, TauBench Retail, TauBench General{self.terminal.RESET}")
                elif selection == 'marathon_mode':
                    print(f"  {self.terminal.YELLOW}MARATHON MODE: All discovered local models × All benchmarks × 50 samples{self.terminal.RESET}")
                    print(f"  {self.terminal.GRAY}→ Perfect for overnight runs - just start it and leave your laptop on!{self.terminal.RESET}")
                
                return selection
            else:
                print(f"{self.terminal.YELLOW}Please enter 1-18{self.terminal.RESET}")
    
    async def get_all_suggested_models(self) -> List[str]:
        """Get all suggested models for custom selection - Use ACTUAL discovered local models + external config."""
        suggested = []
        
        # First, get your actual local models
        try:
            model_manager = await self.init_model_manager()
            local_models = await model_manager.get_all_local_models()
            local_model_names = [m['name'] for m in local_models if m.get('available', True)]
            suggested.extend(local_model_names)
        except Exception as e:
            print(f"{self.terminal.YELLOW}Could not get local models: {e}{self.terminal.RESET}")
        
        # Add models from external configuration
        external_models = self.config.get('external_models', {})
        huggingface_models = external_models.get('huggingface_models', {})
        
        # Add all categories of HuggingFace models from external config
        for category, models in huggingface_models.items():
            if isinstance(models, list):
                suggested.extend(models)
        
        # Fallback to basic verified models if no external config
        if not huggingface_models:
            fallback_models = [
                "google/gemma-2-2b-it",           # CONFIRMED WORKING
                "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
                "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
            ]
            suggested.extend(fallback_models)
        
        return suggested
    
    async def interactive_model_selection(self) -> Dict[str, Any]:
        """Interactive model selection with batch options."""
        print(f"\n{self.terminal.BOLD} Custom Evaluation Setup{self.terminal.RESET}")
        
        # Model selection
        suggestions = await self.get_all_suggested_models()
        print(f"\n{self.terminal.BOLD} Model Selection Options:{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}a{self.terminal.RESET} - All available models ({len(suggestions)} models)")
        print(f"  {self.terminal.GREEN}l{self.terminal.RESET} - Local models only (your installed models)")
        print(f"  {self.terminal.GREEN}c{self.terminal.RESET} - Choose specific models manually")
        
        while True:
            choice = input(f"\n{self.terminal.CYAN}Select models [a/l/c]:{self.terminal.RESET} ").strip().lower()
            
            if choice == 'a':
                models = suggestions
                print(f" Selected all {len(models)} available models")
                break
            elif choice == 'l':
                # Get only local models
                try:
                    model_manager = await self.init_model_manager()
                    local_models = await model_manager.get_all_local_models()
                    models = [m['name'] for m in local_models if m.get('available', True)]
                    print(f" Selected {len(models)} local models")
                except Exception as e:
                    print(f"{self.terminal.RED}Error getting local models: {e}{self.terminal.RESET}")
                    continue
                break
            elif choice == 'c':
                models = await self._manual_model_selection(suggestions)
                break
            else:
                print(f"{self.terminal.YELLOW}Please enter 'a', 'l', or 'c'{self.terminal.RESET}")
        
        if not models:
            return {}
        
        # Sample count selection
        print(f"\n{self.terminal.BOLD} Sample Count Options:{self.terminal.RESET}")
        sample_options = {
            '1': ('Quick Test', 10, '~1-2 minutes'),
            '2': ('Light Eval', 25, '~3-5 minutes'), 
            '3': ('Standard', 50, '~8-15 minutes'),
            '4': ('Thorough', 100, '~20-45 minutes'),
            '5': ('Custom', 0, 'You choose')
        }
        
        for key, (name, count, time) in sample_options.items():
            print(f"  {self.terminal.GREEN}{key}{self.terminal.RESET} - {name}: {count} samples {self.terminal.GRAY}({time}){self.terminal.RESET}")
        
        while True:
            choice = input(f"\n{self.terminal.CYAN}Select sample count [1-5]:{self.terminal.RESET} ").strip()
            
            if choice in sample_options:
                if choice == '5':
                    try:
                        samples = int(input(f"{self.terminal.CYAN}Enter custom sample count:{self.terminal.RESET} "))
                        if samples <= 0:
                            print(f"{self.terminal.YELLOW}Please enter a positive number{self.terminal.RESET}")
                            continue
                    except ValueError:
                        print(f"{self.terminal.YELLOW}Please enter a valid number{self.terminal.RESET}")
                        continue
                else:
                    samples = sample_options[choice][1]
                break
            else:
                print(f"{self.terminal.YELLOW}Please enter 1-5{self.terminal.RESET}")
        
        # Benchmark selection
        benchmark_options = {
            '1': ('GSM8K', 'gsm8k', 'Math word problems'),
            '2': ('MMLU', 'mmlu', 'Multiple choice knowledge'),
            '3': ('HumanEval', 'humaneval', 'Code generation'),
            '4': ('MATH', 'math', 'Competition mathematics'),
            '5': ('AIME 2024', 'aime_2024', 'Math competition (OpenAI-level)'),
            '6': ('GPQA Diamond', 'gpqa_diamond', 'PhD-level science'),
            '7': ('OpenAI Suite', 'openai_suite', 'All OpenAI benchmarks'),
            '8': ('Comprehensive', 'comprehensive_suite', 'Best overall coverage'),
            '9': ('Legacy', 'legacy_suite', 'GSM8K + MMLU'),
            '10': ('ALL BENCHMARKS', 'all_benchmarks', 'Every single benchmark available')
        }
        
        print(f"\n{self.terminal.BOLD} Benchmark Selection:{self.terminal.RESET}")
        for key, (name, code, desc) in benchmark_options.items():
            print(f"  {self.terminal.GREEN}{key}{self.terminal.RESET} - {name}: {desc}")
        
        while True:
            choice = input(f"\n{self.terminal.CYAN}Select benchmark [1-10]:{self.terminal.RESET} ").strip()
            if choice in benchmark_options:
                benchmark = benchmark_options[choice][1]
                break
            else:
                print(f"{self.terminal.YELLOW}Please enter 1-10{self.terminal.RESET}")
        
        return {
            'models': models,
            'samples': samples, 
            'benchmark': benchmark,
            'name': f"Custom ({len(models)} models, {samples} samples)"
        }
    
    async def _manual_model_selection(self, suggestions: List[str]) -> List[str]:
        """Manual model selection with multi-select."""
        print(f"\n{self.terminal.BOLD}Available Models:{self.terminal.RESET}")
        for i, model in enumerate(suggestions, 1):
            print(f"  {i:2d}. {model}")
        
        print(f"\n{self.terminal.YELLOW}Selection options:{self.terminal.RESET}")
        print("  • Single: 3")
        print("  • Multiple: 1,3,5")  
        print("  • Range: 1-5")
        print("  • Mix: 1,3-5,8")
        print("  • Custom model: type full name")
        
        models = []
        while True:
            try:
                inp = input(f"\n{self.terminal.GREEN}Select models (empty to finish):{self.terminal.RESET} ").strip()
                if not inp:
                    break
                
                # Parse selection
                if ',' in inp or '-' in inp:
                    # Multiple selection
                    for part in inp.split(','):
                        part = part.strip()
                        if '-' in part:
                            # Range
                            start, end = map(int, part.split('-'))
                            for i in range(start, end + 1):
                                if 1 <= i <= len(suggestions):
                                    model = suggestions[i-1]
                                    if model not in models:
                                        models.append(model)
                        else:
                            # Single number
                            i = int(part)
                            if 1 <= i <= len(suggestions):
                                model = suggestions[i-1]
                                if model not in models:
                                    models.append(model)
                elif inp.isdigit():
                    # Single selection
                    i = int(inp)
                    if 1 <= i <= len(suggestions):
                        model = suggestions[i-1]
                        if model not in models:
                            models.append(model)
                else:
                    # Custom model name
                    if inp not in models:
                        models.append(inp)
                
                print(f"   Selected {len(models)} models")
                    
            except (ValueError, IndexError):
                print(f"{self.terminal.YELLOW}Invalid selection. Use numbers, ranges (1-5), or model names{self.terminal.RESET}")
            except KeyboardInterrupt:
                print(f"\n{self.terminal.YELLOW}Selection cancelled{self.terminal.RESET}")
                return []
        
        return models
    
    async def run_evaluation_with_display_async(self, config_dict: Dict[str, Any]) -> bool:
        """Async version of run evaluation with beautiful terminal display."""
        models = config_dict.get('models', [])
        samples = config_dict.get('samples', 50)
        benchmark = config_dict.get('benchmark', 'gsm8k')
        name = config_dict.get('name', 'Evaluation')
        
        if not models:
            print(f"{self.terminal.RED} No models selected{self.terminal.RESET}")
            return False

        # Initialize storage and start evaluation session
        from src.utils.storage import ResultStorage
        storage = ResultStorage(self.config)
        session_name = f"local_eval_{len(models)}models_{samples}samples"
        session_id = storage.start_evaluation_session(session_name)
        
        print(f"\n{self.terminal.BOLD} Starting {name}{self.terminal.RESET}")
        print(f"Models: {len(models)} | Benchmark: {benchmark} | Samples: {samples}")
        print(f"Session: {session_id}")
        print(f"Results: evaluation_sessions/{datetime.now().strftime('%Y-%m-%d')}/{session_id}/\n")
        
        #  CRITICAL: Resource estimation and warning system
        from src.benchmarks.benchmark_registry import BenchmarkRegistry
        registry = BenchmarkRegistry(self.config)
        
        if registry.is_suite(benchmark):
            expanded_benchmarks = registry.expand_suite(benchmark)
            total_evaluations = len(models) * len(expanded_benchmarks) * samples
            estimated_hours = total_evaluations * 0.02 / 60  # Rough estimate: 1.2 seconds per evaluation
            
            print(f"{self.terminal.YELLOW}  RESOURCE WARNING:{self.terminal.RESET}")
            print(f"   • Total evaluations: {total_evaluations:,} ({len(models)} models × {len(expanded_benchmarks)} benchmarks × {samples} samples)")
            print(f"   • Estimated time: {estimated_hours:.1f} hours")
            print(f"   • Benchmarks: {', '.join(expanded_benchmarks)}")
            
            if total_evaluations > 5000:
                print(f"\n{self.terminal.RED} MASSIVE EVALUATION DETECTED!{self.terminal.RESET}")
                print(f"This evaluation will take {estimated_hours:.1f} hours and may overwhelm your system.")
                print(f"\n{self.terminal.CYAN} Recommended alternatives:{self.terminal.RESET}")
                print(f"   • Use 'comprehensive_suite' (6 benchmarks) instead of 'all_benchmarks' (14 benchmarks)")
                print(f"   • Use 'safe_suite' (2 benchmarks) for slow laptops")
                print(f"   • Reduce sample count to 25-50 for initial testing")
                print(f"   • Run fewer models at once (5-10 instead of {len(models)})")
                print(f"\n{self.terminal.YELLOW}  For slow laptops:{self.terminal.RESET}")
                print(f"   • Use the 'safe' preset for most reliable experience")
                print(f"   • Some benchmarks require internet for dataset downloads")
                
                confirm = input(f"\n{self.terminal.YELLOW}Continue with this massive evaluation anyway? [y/N]:{self.terminal.RESET} ").strip().lower()
                if confirm not in ['y', 'yes']:
                    print(f"{self.terminal.GREEN} Smart choice! Try a smaller evaluation first.{self.terminal.RESET}")
                    return False
        
        # Store evaluation metadata for export
        self.current_eval_metadata = {
            'session_id': session_id,
            'evaluation_type': name,
            'preset_used': config_dict.get('preset', 'custom'),
            'total_models': len(models),
            'samples_per_model': samples,
            'benchmarks': [benchmark] if benchmark != 'both' else ['gsm8k', 'mmlu'],
            'started_at': datetime.now().isoformat(),
            'models_evaluated': models
        }
        
        # Start evaluation display
        self.terminal.print_header()
        self.terminal.print_model_table_header()
        
        try:
            # Run models one by one (laptop-friendly) with checkpoint/resume support
            from src.evaluator import EvaluationOrchestrator, EvaluationConfig
            
            total_models = len(models)
            completed_models = 0
            failed_models = []
            
            # Check if we can resume from a previous session
            existing_results = storage.get_cached_results()
            completed_evaluations = set()
            
            for result in existing_results:
                if result.get('session_id') == session_id:
                    key = f"{result.get('model_name')}_{result.get('benchmark_name')}"
                    completed_evaluations.add(key)
            
            if completed_evaluations:
                print(f" Found {len(completed_evaluations)} completed evaluations from previous session - will skip these")
            
            for i, model_name in enumerate(models, 1):
                print(f"\n{self.terminal.CYAN} Evaluating model {i}/{total_models}: {model_name}{self.terminal.RESET}")
                
                # Handle benchmark suites by expanding them to individual benchmarks
                from src.benchmarks.benchmark_registry import BenchmarkRegistry
                registry = BenchmarkRegistry(self.config)
                
                if registry.is_suite(benchmark):
                    benchmarks_to_run = registry.expand_suite(benchmark)
                    print(f"  {self.terminal.GRAY}Expanding {benchmark} suite to: {', '.join(benchmarks_to_run)}{self.terminal.RESET}")
                else:
                    benchmarks_to_run = [benchmark]
                
                model_success = True
                consecutive_failures = 0
                
                for bench in benchmarks_to_run:
                    # Check if this evaluation was already completed
                    eval_key = f"{model_name}_{bench}"
                    if eval_key in completed_evaluations:
                        print(f"  {self.terminal.GREEN} {bench}: Already completed (skipped){self.terminal.RESET}")
                        continue
                    
                    print(f"  {self.terminal.GRAY}Running {bench} benchmark...{self.terminal.RESET}")
                    
                    # Create individual evaluation config
                    eval_config = EvaluationConfig(
                        model_name=model_name,
                        benchmark_name=bench,
                        num_samples=samples
                    )
                    
                    try:
                        # Run evaluation for this model and benchmark
                        orchestrator = EvaluationOrchestrator()
                        model_result = await orchestrator.evaluate_single(eval_config)
                        
                        # Show result
                        if model_result.error:
                            print(f"    {self.terminal.RED} {bench}: {model_result.error}{self.terminal.RESET}")
                            consecutive_failures += 1
                            model_success = False
                            
                            # If multiple timeouts in a row, suggest reducing scope
                            if 'timeout' in model_result.error.lower():
                                print(f"    {self.terminal.YELLOW} Tip: This model may be too slow for {samples} samples. Consider reducing sample count or using smaller benchmarks.{self.terminal.RESET}")
                                
                                # Auto-skip if too many consecutive failures
                                if consecutive_failures >= 3:
                                    print(f"    {self.terminal.YELLOW}  Skipping remaining benchmarks for {model_name} due to repeated failures{self.terminal.RESET}")
                                    failed_models.append(model_name)
                                    break
                        else:
                            accuracy = model_result.accuracy * 100
                            print(f"    {self.terminal.GREEN} {bench}: {accuracy:.1f}% accuracy{self.terminal.RESET}")
                            consecutive_failures = 0  # Reset failure counter on success
                        
                        # Save checkpoint after each benchmark
                        completed_evaluations.add(eval_key)
                        
                    except Exception as bench_error:
                        print(f"    {self.terminal.RED} {bench}: {bench_error}{self.terminal.RESET}")
                        consecutive_failures += 1
                        model_success = False
                        
                        # If this looks like a resource issue, offer guidance
                        if 'timeout' in str(bench_error).lower() or 'connection' in str(bench_error).lower():
                            print(f"    {self.terminal.YELLOW} Resource tip: Try 'ollama list' to check if model is loaded, or reduce concurrent evaluations{self.terminal.RESET}")
                            
                            # Auto-skip after repeated connection failures
                            if consecutive_failures >= 3:
                                print(f"    {self.terminal.YELLOW}  Skipping remaining benchmarks for {model_name} due to connection issues{self.terminal.RESET}")
                                failed_models.append(model_name)
                                break
                        continue
                
                if model_success or consecutive_failures < 3:
                    completed_models += 1
                    print(f"{self.terminal.GREEN} Completed {model_name}{self.terminal.RESET}")
                else:
                    print(f"{self.terminal.YELLOW}  Partially completed {model_name} (some benchmarks failed){self.terminal.RESET}")
                
                # Brief pause between models (laptop-friendly)
                if i < len(models):
                    print(f"  {self.terminal.GRAY}Pausing 5 seconds between models for resource management...{self.terminal.RESET}")
                    await asyncio.sleep(5)
            
            # Update metadata with completion info
            self.current_eval_metadata.update({
                'completed_at': datetime.now().isoformat(),
                'evaluations_completed': completed_models,
                'failed_models': failed_models,
                'total_successful_evaluations': len(completed_evaluations)
            })
            
            # End the evaluation session
            storage.end_evaluation_session()
            
            # Smart summary with actionable guidance
            print(f"\n{self.terminal.GREEN} Evaluation completed!{self.terminal.RESET}")
            print(f" Results: {completed_models}/{total_models} models successful")
            print(f" Total evaluations: {len(completed_evaluations)}")
            
            if failed_models:
                print(f"\n{self.terminal.YELLOW}  Models with issues: {len(failed_models)}{self.terminal.RESET}")
                for model in failed_models:
                    print(f"   • {model}")
                print(f"\n{self.terminal.CYAN} Improvement suggestions:{self.terminal.RESET}")
                print(f"   • Try smaller benchmarks first (gsm8k, mmlu)")
                print(f"   • Reduce sample count to 25-50")
                print(f"   • Check 'ollama list' to ensure models are loaded")
                print(f"   • Restart Ollama service if connections are failing")
            
            print(f"\n Results saved to evaluation_sessions/{datetime.now().strftime('%Y-%m-%d')}/{session_id}/")
            print(f" Run 'export' to create website files")
            
            return completed_models > 0  # Return True if at least some models completed
            
        except Exception as e:
            # End session even on failure
            try:
                storage.end_evaluation_session()
            except:
                pass
            print(f"\n{self.terminal.RED} Evaluation failed: {e}{self.terminal.RESET}")
            import traceback
            print(f"{self.terminal.GRAY}{traceback.format_exc()}{self.terminal.RESET}")
            return False
    
    def run_evaluation_with_display(self, config_dict: Dict[str, Any]) -> bool:
        """Run evaluation with beautiful terminal display."""
        models = config_dict.get('models', [])
        samples = config_dict.get('samples', 50)
        benchmark = config_dict.get('benchmark', 'gsm8k')
        name = config_dict.get('name', 'Evaluation')
        
        if not models:
            print(f"{self.terminal.RED} No models selected{self.terminal.RESET}")
            return False
        
        print(f"\n{self.terminal.BOLD} Starting {name}{self.terminal.RESET}")
        print(f"Models: {len(models)} | Benchmark: {benchmark} | Samples: {samples}")
        print(f"Results: smaLLMs_results/ - organized by date\n")
        
        # Store evaluation metadata for export
        self.current_eval_metadata = {
            'evaluation_type': name,
            'preset_used': config_dict.get('preset', 'custom'),
            'total_models': len(models),
            'samples_per_model': samples,
            'benchmarks': [benchmark] if benchmark != 'both' else ['gsm8k', 'mmlu'],
            'started_at': datetime.now().isoformat(),
            'models_evaluated': models
        }
        
        # Create intelligent evaluation config
        intelligent_config = IntelligentEvaluationConfig(
            models=models,
            benchmarks=[benchmark] if benchmark != 'both' else ['gsm8k', 'mmlu'],
            samples_per_eval=samples,
            progressive_sampling=samples > 50,  # Use progressive for larger evals
            requests_per_minute=15,  # Extra conservative for local models
            concurrent_models=1,  # Always one at a time for laptops
            delay_between_evals=3.0,  # Extra pause between models for resource management
            output_dir="smaLLMs_results"
        )
        
        # Initialize orchestrator
        self.orchestrator = IntelligentEvaluationOrchestrator(intelligent_config)
        
        # Start evaluation in background and show progress
        self.terminal.print_header()
        self.terminal.print_model_table_header()
        
        try:
            # Run the actual evaluation - this method needs to be async
            print(f"{self.terminal.YELLOW}  Starting evaluation (this will take some time)...{self.terminal.RESET}")
            
            # For now, let's use a simple synchronous approach
            # We'll create a basic evaluation loop that shows progress
            from src.evaluator import EvaluationOrchestrator, EvaluationConfig
            
            # Create basic config for each model
            total_models = len(models)
            completed_models = 0
            
            for i, model_name in enumerate(models, 1):
                print(f"\n{self.terminal.CYAN} Evaluating model {i}/{total_models}: {model_name}{self.terminal.RESET}")
                
                # Handle benchmark suites by expanding them to individual benchmarks
                from src.benchmarks.benchmark_registry import BenchmarkRegistry
                registry = BenchmarkRegistry(self.config)
                
                if registry.is_suite(benchmark):
                    benchmarks_to_run = registry.expand_suite(benchmark)
                    print(f"  {self.terminal.GRAY}Expanding {benchmark} suite to: {', '.join(benchmarks_to_run)}{self.terminal.RESET}")
                else:
                    benchmarks_to_run = [benchmark]
                
                for bench in benchmarks_to_run:
                    print(f"  {self.terminal.GRAY}Running {bench} benchmark...{self.terminal.RESET}")
                    
                    # Create individual evaluation config
                    eval_config = EvaluationConfig(
                        model_name=model_name,
                        benchmark_name=bench,
                        num_samples=samples
                    )
                    
                    try:
                        # Run evaluation for this model and benchmark
                        orchestrator = EvaluationOrchestrator()
                        model_result = asyncio.run(orchestrator.evaluate_single(eval_config))
                        
                        # Show result
                        if model_result.error:
                            print(f"    {self.terminal.RED} {bench}: {model_result.error}{self.terminal.RESET}")
                        else:
                            accuracy = model_result.accuracy * 100
                            print(f"    {self.terminal.GREEN} {bench}: {accuracy:.1f}% accuracy{self.terminal.RESET}")
                        
                    except Exception as bench_error:
                        print(f"    {self.terminal.RED} {bench}: {bench_error}{self.terminal.RESET}")
                        continue
                
                completed_models += 1
                print(f"{self.terminal.GREEN} Completed {model_name}{self.terminal.RESET}")
                
                # Brief pause between models (laptop-friendly)
                import time
                time.sleep(3)
            
            print(f"\n{self.terminal.GREEN} Evaluation completed! {completed_models}/{total_models} models successful{self.terminal.RESET}")
            print(f" Results saved to smaLLMs_results/")
            print(f" Run 'export' to create website files")
            return True
            
        except Exception as e:
            print(f"\n{self.terminal.RED} Evaluation failed: {e}{self.terminal.RESET}")
            import traceback
            print(f"{self.terminal.GRAY}{traceback.format_exc()}{self.terminal.RESET}")
            return False
    
    def export_results(self):
        """Export results for website."""
        print(f"\n{self.terminal.BOLD} Exporting Results{self.terminal.RESET}")
        
        try:
            exported_files = self.exporter.export_for_website()
            
            if exported_files:
                print(f"\n{self.terminal.GREEN} Export completed!{self.terminal.RESET}")
                print(f"{self.terminal.BOLD}Files created:{self.terminal.RESET}")
                for file_type, file_path in exported_files.items():
                    print(f"   {file_type.upper()}: {Path(file_path).name}")
                
                print(f"\n{self.terminal.YELLOW} Next steps:{self.terminal.RESET}")
                print("  • Open HTML file in browser to view results")
                print("  • Copy JSON to your website project")  
                print("  • Use CSV for detailed analysis in Excel")
            else:
                print(f"{self.terminal.YELLOW}  No results found to export{self.terminal.RESET}")
                print("Run some evaluations first!")
                
        except Exception as e:
            print(f"{self.terminal.RED} Export failed: {e}{self.terminal.RESET}")
    
    def show_status(self):
        """Show current results summary."""
        print(f"\n{self.terminal.BOLD} Current Status{self.terminal.RESET}")
        
        # Check for recent results
        cache_dir = Path("results/cache")
        report_dir = Path("smaLLMs_results")
        export_dir = Path("website_exports")
        
        cache_files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
        report_files = list(report_dir.glob("*.json")) if report_dir.exists() else []
        export_files = list(export_dir.glob("*")) if export_dir.exists() else []
        
        print(f" {self.terminal.CYAN}results/cache/{self.terminal.RESET}: {len(cache_files)} cached evaluations")
        print(f" {self.terminal.CYAN}smaLLMs_results/{self.terminal.RESET}: {len(report_files)} reports")
        print(f" {self.terminal.CYAN}website_exports/{self.terminal.RESET}: {len(export_files)} export files")
        
        if report_files:
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            mod_time = datetime.fromtimestamp(latest_report.stat().st_mtime)
            print(f"\n{self.terminal.GREEN} Latest report:{self.terminal.RESET} {latest_report.name}")
            print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Quick summary from exporter
        try:
            data = self.exporter.load_latest_results()
            results = data.get('results', [])
            if results:
                df = self.exporter.create_clean_leaderboard(results)
                self.exporter.print_summary(df)
        except Exception as e:
            print(f"{self.terminal.GRAY}Could not load results summary: {e}{self.terminal.RESET}")
    
    def run_interactive(self):
        """Main interactive loop with mode selection."""
        self.print_welcome()
        
        while True:
            try:
                cmd = input(f"\n{self.terminal.BOLD}smaLLMs{self.terminal.RESET} {self.terminal.GRAY}${self.terminal.RESET} ").strip().lower()
                
                if cmd in ['exit', 'quit', 'q']:
                    print(f"{self.terminal.YELLOW} Goodbye!{self.terminal.RESET}")
                    break
                
                # Mode Selection
                elif cmd == 'local':
                    asyncio.run(self.run_local_mode())
                    self.print_welcome()  # Return to main menu
                    
                elif cmd == 'cloud':
                    self.run_cloud_mode()
                    self.print_welcome()  # Return to main menu
                
                # Quick Commands (available from main menu)
                elif cmd == 'discover':
                    asyncio.run(self.discover_local_models_interactive())
                    
                elif cmd == 'status':
                    self.show_status()
                    
                elif cmd == 'export':
                    self.export_results()
                    
                elif cmd == 'space':
                    self.check_disk_space()
                    
                elif cmd in ['help', 'h']:
                    self.print_welcome()
                    
                elif cmd == 'clear':
                    self.terminal.clear_screen()
                    self.print_welcome()
                
                elif cmd in ['marathon', 'marathon_mode', 'overnight']:
                    # Quick marathon mode from main menu
                    success = asyncio.run(self.run_marathon_mode())
                    if success:
                        print(f"\n{self.terminal.GREEN} Marathon Mode completed! Check 'status' for comprehensive results{self.terminal.RESET}")
                    self.print_welcome()  # Return to main menu
                    
                else:
                    print(f"{self.terminal.YELLOW} Unknown command: {cmd}{self.terminal.RESET}")
                    print(f"Type 'local' for local models, 'cloud' for HuggingFace models, or 'help' for options")
                    
            except KeyboardInterrupt:
                print(f"\n{self.terminal.YELLOW}Use 'exit' to quit{self.terminal.RESET}")
            except Exception as e:
                print(f"{self.terminal.RED} Error: {e}{self.terminal.RESET}")
                import traceback
                print(f"{self.terminal.GRAY}{traceback.format_exc()}{self.terminal.RESET}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="smaLLMs - Small Language Model Evaluation Platform")
    parser.add_argument('command', nargs='?', default='interactive', 
                       choices=['interactive', 'eval', 'quick', 'export', 'status'],
                       help='Command to run')
    parser.add_argument('--models', nargs='+', help='Models to evaluate')
    parser.add_argument('--benchmark', default='gsm8k', help='Benchmark to use')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples')
    
    args = parser.parse_args()
    launcher = SmaLLMsLauncher()
    
    if args.command == 'interactive':
        launcher.run_interactive()
    elif args.command == 'eval':
        models = args.models if args.models else launcher.interactive_model_selection()
        launcher.run_evaluation_with_display(models, args.benchmark, args.samples)
    elif args.command == 'quick':
        models = launcher.get_quick_models()
        launcher.run_evaluation_with_display(models, args.benchmark, min(args.samples, 20))
    elif args.command == 'export':
        launcher.export_results()
    elif args.command == 'status':
        launcher.show_status()

# Export alias for backwards compatibility
SmaLLMsEvaluator = SmaLLMsLauncher

if __name__ == "__main__":
    main()
