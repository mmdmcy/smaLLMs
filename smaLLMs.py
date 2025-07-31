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
        
        # Ensure directories exist
        Path("results/cache").mkdir(parents=True, exist_ok=True)
        Path("smaLLMs_results").mkdir(parents=True, exist_ok=True)
        Path("website_exports").mkdir(parents=True, exist_ok=True)
    
    def print_welcome(self):
        """Print welcome message and options."""
        self.terminal.clear_screen()
        print(f"""
{self.terminal.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                          {self.terminal.BOLD}🚀 smaLLMs Evaluation Platform{self.terminal.RESET}{self.terminal.CYAN}                          ║
║                     {self.terminal.GRAY}Your Enterprise-Grade LLM Benchmarking Suite{self.terminal.RESET}{self.terminal.CYAN}                   ║
╚══════════════════════════════════════════════════════════════════════════════╝{self.terminal.RESET}

{self.terminal.BOLD}🚀 Quick Start Options:{self.terminal.RESET}
  {self.terminal.GREEN}1{self.terminal.RESET} - {self.terminal.BOLD}Lightning Demo{self.terminal.RESET}     - 3 models, 10 samples (~2 min)
  {self.terminal.GREEN}2{self.terminal.RESET} - {self.terminal.BOLD}Quick Benchmark{self.terminal.RESET}    - 5 models, 25 samples (~5 min)  
  {self.terminal.GREEN}3{self.terminal.RESET} - {self.terminal.BOLD}Standard Eval{self.terminal.RESET}      - 8 models, 50 samples (~15 min)
  {self.terminal.GREEN}4{self.terminal.RESET} - {self.terminal.BOLD}Comprehensive{self.terminal.RESET}      - 12 models, 100 samples (~45 min)
  {self.terminal.GREEN}5{self.terminal.RESET} - {self.terminal.BOLD}Custom{self.terminal.RESET}             - Choose your own models & settings

{self.terminal.BOLD}📊 Other Commands:{self.terminal.RESET}
  {self.terminal.CYAN}export{self.terminal.RESET}   - Export results for website/analysis
  {self.terminal.CYAN}status{self.terminal.RESET}   - Show current results summary
  {self.terminal.CYAN}help{self.terminal.RESET}     - Show this menu again

{self.terminal.BOLD}📁 File Structure:{self.terminal.RESET}
  📁 {self.terminal.GRAY}results/cache/{self.terminal.RESET}           - Individual evaluation cache
  📁 {self.terminal.GRAY}smaLLMs_results/{self.terminal.RESET}     - Organized evaluation results  
  📁 {self.terminal.GRAY}website_exports/{self.terminal.RESET}         - Clean data for your website

{self.terminal.YELLOW}💡 Just type a number (1-5) to start evaluating!{self.terminal.RESET}
""")
    
    def get_preset_config(self, preset: str) -> Dict[str, Any]:
        """Get preset evaluation configurations - Using only CONFIRMED WORKING models."""
        presets = {
            "lightning": {
                "name": "Lightning Demo",
                "models": [
                    "google/gemma-2-2b-it",           # CONFIRMED WORKING
                    "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
                    "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
                ],
                "samples": 10,
                "description": "Quick 2-minute demo with 3 confirmed working models"
            },
            "quick": {
                "name": "Quick Benchmark", 
                "models": [
                    "google/gemma-2-2b-it",           # CONFIRMED WORKING
                    "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
                    "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
                    "Qwen/Qwen2.5-3B-Instruct",      # CONFIRMED WORKING
                    "meta-llama/Llama-3.2-3B-Instruct", # CONFIRMED WORKING
                ],
                "samples": 25,
                "description": "8-minute evaluation with 5 reliable instruct models"
            },
            "standard": {
                "name": "Standard Evaluation",
                "models": [
                    "google/gemma-2-2b-it",           # CONFIRMED WORKING
                    "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
                    "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
                    "Qwen/Qwen2.5-3B-Instruct",      # CONFIRMED WORKING
                    "meta-llama/Llama-3.2-3B-Instruct", # CONFIRMED WORKING
                    "Qwen/Qwen2.5-7B-Instruct",      # Should work (7B size)
                    "HuggingFaceTB/SmolLM2-1.7B-Instruct", # Should work (instruct)
                    "google/gemma-2-9b-it",          # Confirmed working Gemma family
                ],
                "samples": 50,
                "description": "25-minute comprehensive test with 8 working models"
            },
            "comprehensive": {
                "name": "Comprehensive Benchmark",
                "models": [
                    "google/gemma-2-2b-it",           # CONFIRMED WORKING
                    "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
                    "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
                    "Qwen/Qwen2.5-3B-Instruct",      # CONFIRMED WORKING
                    "meta-llama/Llama-3.2-3B-Instruct", # CONFIRMED WORKING
                    "Qwen/Qwen2.5-7B-Instruct",      # Should work (7B size)
                    "HuggingFaceTB/SmolLM2-1.7B-Instruct", # Should work (instruct)
                    "google/gemma-2-9b-it",          # Confirmed working Gemma family
                    "mistralai/Mistral-7B-Instruct-v0.3", # Popular instruct model
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Small but reliable
                    "microsoft/DialoGPT-medium",      # Alternative to problematic models
                    "microsoft/DialoGPT-large",       # Larger alternative
                ],
                "samples": 100,
                "description": "60-minute thorough evaluation with 12 working models"
            }
        }
        return presets.get(preset, {})
    
    def get_all_suggested_models(self) -> List[str]:
        """Get all suggested models for custom selection - Only CONFIRMED WORKING models."""
        return [
            "google/gemma-2-2b-it",           # CONFIRMED WORKING
            "Qwen/Qwen2.5-1.5B-Instruct",    # CONFIRMED WORKING 
            "meta-llama/Llama-3.2-1B-Instruct", # CONFIRMED WORKING
            "Qwen/Qwen2.5-3B-Instruct",      # CONFIRMED WORKING
            "meta-llama/Llama-3.2-3B-Instruct", # CONFIRMED WORKING
            "Qwen/Qwen2.5-7B-Instruct",      # Should work (7B size)
            "HuggingFaceTB/SmolLM2-1.7B-Instruct", # Should work (instruct)
            "google/gemma-2-9b-it",          # Confirmed working Gemma family
            "mistralai/Mistral-7B-Instruct-v0.3", # Popular instruct model
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Small but reliable
            "microsoft/DialoGPT-medium",      # Alternative to problematic models
            "microsoft/DialoGPT-large",       # Larger alternative
        ]
    
    def interactive_model_selection(self) -> Dict[str, Any]:
        """Interactive model selection with batch options."""
        print(f"\n{self.terminal.BOLD}🎯 Custom Evaluation Setup{self.terminal.RESET}")
        
        # Model selection
        suggestions = self.get_all_suggested_models()
        print(f"\n{self.terminal.BOLD}📋 Model Selection Options:{self.terminal.RESET}")
        print(f"  {self.terminal.GREEN}a{self.terminal.RESET} - All suggested models ({len(suggestions)} models)")
        print(f"  {self.terminal.GREEN}t{self.terminal.RESET} - Top 5 models (curated best performers)")
        print(f"  {self.terminal.GREEN}s{self.terminal.RESET} - Small models only (< 2B parameters)")
        print(f"  {self.terminal.GREEN}c{self.terminal.RESET} - Choose specific models manually")
        
        while True:
            choice = input(f"\n{self.terminal.CYAN}Select models [a/t/s/c]:{self.terminal.RESET} ").strip().lower()
            
            if choice == 'a':
                models = suggestions
                print(f"✓ Selected all {len(models)} suggested models")
                break
            elif choice == 't':
                models = suggestions[:5]
                print(f"✓ Selected top 5 models")
                break
            elif choice == 's':
                # Small models (< 2B params)
                small_models = [
                    "microsoft/Phi-3-mini-4k-instruct",
                    "google/gemma-2-2b-it", 
                    "meta-llama/Llama-3.2-1B-Instruct",
                    "Qwen/Qwen2.5-1.5B-Instruct", 
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "bigscience/bloom-1b7"
                ]
                models = small_models
                print(f"✓ Selected {len(models)} small models")
                break
            elif choice == 'c':
                models = self._manual_model_selection(suggestions)
                break
            else:
                print(f"{self.terminal.YELLOW}Please enter 'a', 't', 's', or 'c'{self.terminal.RESET}")
        
        if not models:
            return {}
        
        # Sample count selection
        print(f"\n{self.terminal.BOLD}📊 Sample Count Options:{self.terminal.RESET}")
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
            '3': ('Both', 'both', 'GSM8K + MMLU')
        }
        
        print(f"\n{self.terminal.BOLD}📈 Benchmark Selection:{self.terminal.RESET}")
        for key, (name, code, desc) in benchmark_options.items():
            print(f"  {self.terminal.GREEN}{key}{self.terminal.RESET} - {name}: {desc}")
        
        while True:
            choice = input(f"\n{self.terminal.CYAN}Select benchmark [1-3]:{self.terminal.RESET} ").strip()
            if choice in benchmark_options:
                benchmark = benchmark_options[choice][1]
                break
            else:
                print(f"{self.terminal.YELLOW}Please enter 1-3{self.terminal.RESET}")
        
        return {
            'models': models,
            'samples': samples, 
            'benchmark': benchmark,
            'name': f"Custom ({len(models)} models, {samples} samples)"
        }
    
    def _manual_model_selection(self, suggestions: List[str]) -> List[str]:
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
                
                print(f"  ✓ Selected {len(models)} models")
                    
            except (ValueError, IndexError):
                print(f"{self.terminal.YELLOW}Invalid selection. Use numbers, ranges (1-5), or model names{self.terminal.RESET}")
            except KeyboardInterrupt:
                print(f"\n{self.terminal.YELLOW}Selection cancelled{self.terminal.RESET}")
                return []
        
        return models
    
    def run_evaluation_with_display(self, config_dict: Dict[str, Any]) -> bool:
        """Run evaluation with beautiful terminal display."""
        models = config_dict.get('models', [])
        samples = config_dict.get('samples', 50)
        benchmark = config_dict.get('benchmark', 'gsm8k')
        name = config_dict.get('name', 'Evaluation')
        
        if not models:
            print(f"{self.terminal.RED}❌ No models selected{self.terminal.RESET}")
            return False
        
        print(f"\n{self.terminal.BOLD}🚀 Starting {name}{self.terminal.RESET}")
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
            requests_per_minute=20,  # Conservative for API limits
            concurrent_models=1,  # One at a time to avoid limits
            delay_between_evals=2.0,
            output_dir="smaLLMs_results"
        )
        
        # Initialize orchestrator
        self.orchestrator = IntelligentEvaluationOrchestrator(intelligent_config)
        
        # Start evaluation in background and show progress
        self.terminal.print_header()
        self.terminal.print_model_table_header()
        
        try:
            # Run the actual evaluation using the correct method
            results = asyncio.run(self.orchestrator.run_intelligent_evaluation())
            
            # Update metadata with completion info
            self.current_eval_metadata.update({
                'completed_at': datetime.now().isoformat(),
                'total_cost': results.get('total_cost', 0),
                'total_duration': results.get('total_duration', 0),
                'evaluations_completed': results.get('evaluations_completed', 0)
            })
            
            print(f"\n{self.terminal.GREEN}✅ Evaluation completed successfully!{self.terminal.RESET}")
            print(f"📊 Results saved to smaLLMs_results/")
            print(f"💡 Run 'python simple_exporter.py' to create website export")
            return True
            
        except Exception as e:
            print(f"\n{self.terminal.RED}❌ Evaluation failed: {e}{self.terminal.RESET}")
            import traceback
            print(f"{self.terminal.GRAY}{traceback.format_exc()}{self.terminal.RESET}")
            return False
    
    def export_results(self):
        """Export results for website."""
        print(f"\n{self.terminal.BOLD}📤 Exporting Results{self.terminal.RESET}")
        
        try:
            exported_files = self.exporter.export_for_website()
            
            if exported_files:
                print(f"\n{self.terminal.GREEN}✅ Export completed!{self.terminal.RESET}")
                print(f"{self.terminal.BOLD}Files created:{self.terminal.RESET}")
                for file_type, file_path in exported_files.items():
                    print(f"  📄 {file_type.upper()}: {Path(file_path).name}")
                
                print(f"\n{self.terminal.YELLOW}💡 Next steps:{self.terminal.RESET}")
                print("  • Open HTML file in browser to view results")
                print("  • Copy JSON to your website project")  
                print("  • Use CSV for detailed analysis in Excel")
            else:
                print(f"{self.terminal.YELLOW}⚠️  No results found to export{self.terminal.RESET}")
                print("Run some evaluations first!")
                
        except Exception as e:
            print(f"{self.terminal.RED}❌ Export failed: {e}{self.terminal.RESET}")
    
    def show_status(self):
        """Show current results summary."""
        print(f"\n{self.terminal.BOLD}📊 Current Status{self.terminal.RESET}")
        
        # Check for recent results
        cache_dir = Path("results/cache")
        report_dir = Path("smaLLMs_results")
        export_dir = Path("website_exports")
        
        cache_files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
        report_files = list(report_dir.glob("*.json")) if report_dir.exists() else []
        export_files = list(export_dir.glob("*")) if export_dir.exists() else []
        
        print(f"📁 {self.terminal.CYAN}results/cache/{self.terminal.RESET}: {len(cache_files)} cached evaluations")
        print(f"📁 {self.terminal.CYAN}smaLLMs_results/{self.terminal.RESET}: {len(report_files)} reports")
        print(f"📁 {self.terminal.CYAN}website_exports/{self.terminal.RESET}: {len(export_files)} export files")
        
        if report_files:
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            mod_time = datetime.fromtimestamp(latest_report.stat().st_mtime)
            print(f"\n{self.terminal.GREEN}📋 Latest report:{self.terminal.RESET} {latest_report.name}")
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
        """Main interactive loop."""
        self.print_welcome()
        
        while True:
            try:
                cmd = input(f"\n{self.terminal.BOLD}smaLLMs{self.terminal.RESET} {self.terminal.GRAY}${self.terminal.RESET} ").strip()
                
                if cmd.lower() in ['exit', 'quit', 'q']:
                    print(f"{self.terminal.YELLOW}👋 Goodbye!{self.terminal.RESET}")
                    break
                
                elif cmd == '1':
                    # Lightning Demo
                    config = self.get_preset_config('lightning')
                    config_dict = {
                        'models': config['models'],
                        'samples': config['samples'],
                        'benchmark': 'gsm8k',
                        'name': config['name'],
                        'preset': 'lightning'
                    }
                    success = self.run_evaluation_with_display(config_dict)
                    if success:
                        print(f"\n{self.terminal.CYAN}💡 Run 'export' to create website files{self.terminal.RESET}")
                
                elif cmd == '2':
                    # Quick Benchmark
                    config = self.get_preset_config('quick')
                    config_dict = {
                        'models': config['models'],
                        'samples': config['samples'],
                        'benchmark': 'gsm8k',
                        'name': config['name'],
                        'preset': 'quick'
                    }
                    success = self.run_evaluation_with_display(config_dict)
                    if success:
                        print(f"\n{self.terminal.CYAN}💡 Run 'export' to create website files{self.terminal.RESET}")
                
                elif cmd == '3':
                    # Standard Eval
                    config = self.get_preset_config('standard')
                    config_dict = {
                        'models': config['models'],
                        'samples': config['samples'],
                        'benchmark': 'gsm8k',
                        'name': config['name'],
                        'preset': 'standard'
                    }
                    success = self.run_evaluation_with_display(config_dict)
                    if success:
                        print(f"\n{self.terminal.CYAN}💡 Run 'export' to create website files{self.terminal.RESET}")
                
                elif cmd == '4':
                    # Comprehensive
                    config = self.get_preset_config('comprehensive')
                    config_dict = {
                        'models': config['models'],
                        'samples': config['samples'],
                        'benchmark': 'gsm8k',
                        'name': config['name'],
                        'preset': 'comprehensive'
                    }
                    success = self.run_evaluation_with_display(config_dict)
                    if success:
                        print(f"\n{self.terminal.CYAN}💡 Run 'export' to create website files{self.terminal.RESET}")
                
                elif cmd == '5':
                    # Custom
                    config_dict = self.interactive_model_selection()
                    if config_dict:
                        config_dict['preset'] = 'custom'
                        success = self.run_evaluation_with_display(config_dict)
                        if success:
                            print(f"\n{self.terminal.CYAN}💡 Run 'export' to create website files{self.terminal.RESET}")
                    # Standard Eval
                    config = self.get_preset_config('standard')
                    config_dict = {
                        'models': config['models'],
                        'samples': config['samples'],
                        'benchmark': 'gsm8k',
                        'name': config['name']
                    }
                    success = self.run_evaluation_with_display(config_dict)
                    if success:
                        print(f"\n{self.terminal.CYAN}� Run 'export' to create website files{self.terminal.RESET}")
                
                elif cmd == '4':
                    # Comprehensive
                    config = self.get_preset_config('comprehensive')
                    config_dict = {
                        'models': config['models'],
                        'samples': config['samples'],
                        'benchmark': 'gsm8k',
                        'name': config['name']
                    }
                    success = self.run_evaluation_with_display(config_dict)
                    if success:
                        print(f"\n{self.terminal.CYAN}💡 Run 'export' to create website files{self.terminal.RESET}")
                
                elif cmd == '5':
                    # Custom
                    config_dict = self.interactive_model_selection()
                    if config_dict:
                        success = self.run_evaluation_with_display(config_dict)
                        if success:
                            print(f"\n{self.terminal.CYAN}💡 Run 'export' to create website files{self.terminal.RESET}")
                
                elif cmd.lower() == 'export':
                    self.export_results()
                
                elif cmd.lower() == 'status':
                    self.show_status()
                
                elif cmd.lower() in ['help', 'h']:
                    self.print_welcome()
                
                elif cmd.lower() == 'clear':
                    self.terminal.clear_screen()
                    self.print_welcome()
                
                else:
                    print(f"{self.terminal.YELLOW}❓ Unknown command: {cmd}{self.terminal.RESET}")
                    print(f"Type a number (1-5) or 'help' for options")
                    
            except KeyboardInterrupt:
                print(f"\n{self.terminal.YELLOW}Use 'exit' to quit{self.terminal.RESET}")
            except Exception as e:
                print(f"{self.terminal.RED}❌ Error: {e}{self.terminal.RESET}")
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

if __name__ == "__main__":
    main()
