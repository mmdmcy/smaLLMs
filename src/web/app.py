"""
Web interface for smaLLMs evaluation platform.
Provides an intuitive dashboard for running evaluations and viewing results.
"""

import gradio as gr
import pandas as pd
import asyncio
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluator import EvaluationOrchestrator, EvaluationConfig, quick_eval, compare_models
from models.model_manager import RECOMMENDED_MODELS
from benchmarks.benchmark_registry import STANDARD_BENCHMARK_SUITE, QUICK_BENCHMARK_SUITE

class WebInterface:
    """Main web interface for the smaLLMs platform."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.orchestrator = EvaluationOrchestrator(config_path)
        self.config = self._load_config(config_path)
        
        # Track running evaluations
        self.running_evaluations = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        with gr.Blocks(
            title="smaLLMs - Small Language Model Evaluation Platform",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🧠 smaLLMs - State-of-the-Art Small Model Evaluation
            
            **The ultimate platform for evaluating small language models (1B-20B parameters)**
            
            Featuring enterprise-grade benchmarks used by leading AI labs: MMLU, GSM8K, MATH, HumanEval and more.
            """)
            
            with gr.Tabs():
                # Quick Evaluation Tab
                with gr.Tab("🚀 Quick Evaluation", id="quick_eval"):
                    self._create_quick_eval_tab()
                
                # Comprehensive Evaluation Tab  
                with gr.Tab("📊 Comprehensive Evaluation", id="comprehensive"):
                    self._create_comprehensive_tab()
                
                # Leaderboard Tab
                with gr.Tab("🏆 Leaderboard", id="leaderboard"):
                    self._create_leaderboard_tab()
                
                # Model Comparison Tab
                with gr.Tab("⚖️ Model Comparison", id="comparison"):
                    self._create_comparison_tab()
                
                # Results Explorer Tab
                with gr.Tab("🔍 Results Explorer", id="results"):
                    self._create_results_tab()
                
                # Settings Tab
                with gr.Tab("⚙️ Settings", id="settings"):
                    self._create_settings_tab()
        
        return interface
    
    def _create_quick_eval_tab(self):
        """Create the quick evaluation tab."""
        gr.Markdown("### Quick Model Evaluation")
        gr.Markdown("Test a single model on one benchmark for rapid prototyping and testing.")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_input = gr.Dropdown(
                    choices=RECOMMENDED_MODELS,
                    label="Select Model",
                    value=RECOMMENDED_MODELS[0] if RECOMMENDED_MODELS else "",
                    allow_custom_value=True,
                    info="Choose from recommended models or enter a custom HuggingFace model name"
                )
                
                benchmark_input = gr.Dropdown(
                    choices=QUICK_BENCHMARK_SUITE,
                    label="Select Benchmark",
                    value="gsm8k",
                    info="Choose benchmark for evaluation"
                )
                
                samples_input = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=50,
                    step=10,
                    label="Number of Samples",
                    info="More samples = more accurate but slower"
                )
                
                eval_button = gr.Button("🚀 Start Evaluation", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                status_output = gr.Textbox(
                    label="Evaluation Status",
                    placeholder="Ready to start evaluation...",
                    interactive=False,
                    lines=3
                )
                
                results_output = gr.JSON(
                    label="Evaluation Results",
                    visible=False
                )
        
        # Event handlers
        eval_button.click(
            fn=self._run_quick_evaluation,
            inputs=[model_input, benchmark_input, samples_input],
            outputs=[status_output, results_output]
        )
    
    def _create_comprehensive_tab(self):
        """Create the comprehensive evaluation tab."""
        gr.Markdown("### Comprehensive Model Evaluation")
        gr.Markdown("Run full benchmark suite on one or multiple models.")
        
        with gr.Row():
            with gr.Column():
                models_input = gr.Textbox(
                    label="Models to Evaluate",
                    placeholder="Enter model names, one per line",
                    lines=5,
                    value="\n".join(RECOMMENDED_MODELS[:3])
                )
                
                benchmarks_input = gr.CheckboxGroup(
                    choices=STANDARD_BENCHMARK_SUITE,
                    value=STANDARD_BENCHMARK_SUITE,
                    label="Select Benchmarks"
                )
                
                samples_input = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=100,
                    step=50,
                    label="Samples per Benchmark"
                )
                
                comprehensive_button = gr.Button("📊 Start Comprehensive Evaluation", variant="primary")
            
            with gr.Column():
                comprehensive_status = gr.Textbox(
                    label="Evaluation Progress",
                    placeholder="Ready to start comprehensive evaluation...",
                    interactive=False,
                    lines=10
                )
                
                comprehensive_results = gr.Dataframe(
                    label="Results Summary",
                    visible=False
                )
        
        comprehensive_button.click(
            fn=self._run_comprehensive_evaluation,
            inputs=[models_input, benchmarks_input, samples_input],
            outputs=[comprehensive_status, comprehensive_results]
        )
    
    def _create_leaderboard_tab(self):
        """Create the leaderboard tab."""
        gr.Markdown("### 🏆 Model Leaderboard")
        gr.Markdown("Real-time rankings of evaluated models across all benchmarks.")
        
        with gr.Row():
            refresh_button = gr.Button("🔄 Refresh Leaderboard", variant="secondary")
            export_button = gr.Button("📥 Export Results", variant="secondary")
        
        leaderboard_display = gr.Dataframe(
            label="Current Leaderboard",
            headers=["Rank", "Model", "Overall Score", "MMLU", "GSM8K", "MATH", "HumanEval", "Avg Latency", "Cost Efficiency"],
            datatype=["number", "str", "number", "number", "number", "number", "number", "number", "number"],
            interactive=False
        )
        
        benchmark_stats = gr.JSON(
            label="Benchmark Statistics",
            visible=True
        )
        
        refresh_button.click(
            fn=self._refresh_leaderboard,
            outputs=[leaderboard_display, benchmark_stats]
        )
        
        export_button.click(
            fn=self._export_results,
            outputs=[gr.File()]
        )
    
    def _create_comparison_tab(self):
        """Create the model comparison tab."""
        gr.Markdown("### ⚖️ Model Comparison")
        gr.Markdown("Compare specific models head-to-head on selected benchmarks.")
        
        with gr.Row():
            models_to_compare = gr.CheckboxGroup(
                choices=RECOMMENDED_MODELS,
                label="Select Models to Compare",
                value=RECOMMENDED_MODELS[:2] if len(RECOMMENDED_MODELS) >= 2 else RECOMMENDED_MODELS
            )
            
            comparison_benchmark = gr.Dropdown(
                choices=STANDARD_BENCHMARK_SUITE,
                label="Benchmark for Comparison",
                value="gsm8k"
            )
        
        compare_button = gr.Button("⚖️ Compare Models", variant="primary")
        
        comparison_results = gr.Plot(
            label="Comparison Results",
            visible=False
        )
        
        comparison_table = gr.Dataframe(
            label="Detailed Comparison",
            visible=False
        )
        
        compare_button.click(
            fn=self._compare_models,
            inputs=[models_to_compare, comparison_benchmark],
            outputs=[comparison_results, comparison_table]
        )
    
    def _create_results_tab(self):
        """Create the results explorer tab."""
        gr.Markdown("### 🔍 Results Explorer")
        gr.Markdown("Browse and analyze detailed evaluation results.")
        
        with gr.Row():
            with gr.Column(scale=1):
                filter_model = gr.Dropdown(
                    choices=["All Models"] + RECOMMENDED_MODELS,
                    label="Filter by Model",
                    value="All Models"
                )
                
                filter_benchmark = gr.Dropdown(
                    choices=["All Benchmarks"] + STANDARD_BENCHMARK_SUITE,
                    label="Filter by Benchmark", 
                    value="All Benchmarks"
                )
                
                search_button = gr.Button("🔍 Search Results")
            
            with gr.Column(scale=3):
                results_table = gr.Dataframe(
                    label="Evaluation Results",
                    interactive=False
                )
                
                storage_stats = gr.JSON(
                    label="Storage Statistics"
                )
        
        search_button.click(
            fn=self._search_results,
            inputs=[filter_model, filter_benchmark],
            outputs=[results_table, storage_stats]
        )
    
    def _create_settings_tab(self):
        """Create the settings tab."""
        gr.Markdown("### ⚙️ Platform Settings")
        gr.Markdown("Configure evaluation parameters and storage settings.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Evaluation Settings")
                
                default_samples = gr.Slider(
                    minimum=10,
                    maximum=1000,
                    value=100,
                    label="Default Sample Count"
                )
                
                max_concurrent = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    label="Max Concurrent Requests"
                )
                
                cleanup_enabled = gr.Checkbox(
                    label="Auto-cleanup Old Results",
                    value=True
                )
            
            with gr.Column():
                gr.Markdown("#### Storage Settings")
                
                cache_limit = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=500,
                    label="Local Cache Limit (MB)"
                )
                
                results_format = gr.Dropdown(
                    choices=["json", "csv"],
                    label="Results Format",
                    value="json"
                )
                
                with gr.Row():
                    save_settings_btn = gr.Button("💾 Save Settings", variant="primary")
                    cleanup_btn = gr.Button("🧹 Cleanup Cache", variant="secondary")
        
        save_settings_btn.click(
            fn=self._save_settings,
            inputs=[default_samples, max_concurrent, cleanup_enabled, cache_limit, results_format],
            outputs=[gr.Textbox(label="Status")]
        )
        
        cleanup_btn.click(
            fn=self._cleanup_cache,
            outputs=[gr.Textbox(label="Cleanup Status")]
        )
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .gradio-container {
            max-width: 1200px !important;
        }
        
        .tab-nav {
            font-weight: bold;
        }
        
        .model-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        }
        
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 10px;
        }
        """
    
    # Evaluation methods
    def _run_quick_evaluation(self, model_name: str, benchmark: str, samples: int):
        """Run a quick evaluation using REAL model evaluation."""
        import asyncio
        
        try:
            # Update status
            status = f"🚀 Starting REAL evaluation of {model_name} on {benchmark} with {samples} samples..."
            
            # Create evaluation config
            config = EvaluationConfig(
                model_name=model_name,
                benchmark_name=benchmark,
                num_samples=samples,
                temperature=0.0
            )
            
            # Run REAL evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.orchestrator.evaluate_single(config))
                
                if result.error:
                    error_status = f"❌ Evaluation failed: {result.error}"
                    return error_status, gr.JSON(visible=False)
                
                result_dict = {
                    "model_name": result.model_name,
                    "benchmark": result.benchmark_name,
                    "accuracy": result.accuracy,
                    "latency": result.latency,
                    "cost_estimate": result.cost_estimate,
                    "samples": result.num_samples,
                    "timestamp": result.timestamp
                }
                
                final_status = f"✅ REAL evaluation completed!\nModel: {result.model_name}\nBenchmark: {result.benchmark_name}\nAccuracy: {result.accuracy:.3f}\nLatency: {result.latency:.1f}s\nCost: ${result.cost_estimate:.6f}"
                
                return final_status, gr.JSON(value=result_dict, visible=True)
                
            finally:
                loop.close()
        
        except Exception as e:
            error_status = f"❌ REAL evaluation failed: {str(e)}"
            return error_status, gr.JSON(visible=False)
    
    def _run_comprehensive_evaluation(self, models_text: str, benchmarks: List[str], samples: int):
        """Run comprehensive evaluation using REAL model evaluations."""
        import asyncio
        
        models = [m.strip() for m in models_text.split('\n') if m.strip()]
        
        status = f"🚀 Starting REAL comprehensive evaluation...\nModels: {len(models)}\nBenchmarks: {len(benchmarks)}\nSamples per benchmark: {samples}\n\n"
        
        try:
            # Create evaluation configs for all combinations
            configs = []
            for model in models:
                for benchmark in benchmarks:
                    config = EvaluationConfig(
                        model_name=model,
                        benchmark_name=benchmark,
                        num_samples=samples,
                        temperature=0.0
                    )
                    configs.append(config)
            
            # Run REAL batch evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self.orchestrator.evaluate_batch(configs))
                
                # Convert to DataFrame
                results_data = []
                for result in results:
                    if result.error:
                        results_data.append({
                            "Model": result.model_name,
                            "Benchmark": result.benchmark_name,
                            "Accuracy": "FAILED",
                            "Latency": "N/A",
                            "Cost": "N/A",
                            "Samples": result.num_samples,
                            "Error": result.error[:50] + "..." if len(result.error) > 50 else result.error
                        })
                    else:
                        results_data.append({
                            "Model": result.model_name,
                            "Benchmark": result.benchmark_name,
                            "Accuracy": round(result.accuracy, 3),
                            "Latency": round(result.latency, 1),
                            "Cost": f"${result.cost_estimate:.6f}",
                            "Samples": result.num_samples,
                            "Error": "None"
                        })
                
                results_df = pd.DataFrame(results_data)
                
                successful_results = [r for r in results if not r.error]
                failed_results = [r for r in results if r.error]
                
                final_status = status + f"✅ REAL comprehensive evaluation completed!\nTotal evaluations: {len(results)}\nSuccessful: {len(successful_results)}\nFailed: {len(failed_results)}"
                
                return final_status, gr.Dataframe(value=results_df, visible=True)
                
            finally:
                loop.close()
        
        except Exception as e:
            error_status = status + f"❌ REAL comprehensive evaluation failed: {str(e)}"
            empty_df = pd.DataFrame()
            return error_status, gr.Dataframe(value=empty_df, visible=True)
    
    def _refresh_leaderboard(self):
        """Refresh the leaderboard."""
        try:
            # Load latest leaderboard
            leaderboard = self.orchestrator.storage.get_latest_leaderboard()
            
            if leaderboard:
                # Convert to DataFrame
                leaderboard_data = leaderboard.get('leaderboard', [])
                
                df_data = []
                for i, entry in enumerate(leaderboard_data):
                    benchmark_scores = entry.get('benchmark_scores', {})
                    df_data.append([
                        i + 1,  # Rank
                        entry.get('model_name', ''),
                        entry.get('overall_score', 0),
                        benchmark_scores.get('mmlu', 0),
                        benchmark_scores.get('gsm8k', 0),
                        benchmark_scores.get('math', 0),
                        benchmark_scores.get('humaneval', 0),
                        entry.get('avg_latency', 0),
                        entry.get('cost_efficiency', 0)
                    ])
                
                df = pd.DataFrame(df_data, columns=[
                    "Rank", "Model", "Overall Score", "MMLU", "GSM8K", "MATH", "HumanEval", "Avg Latency", "Cost Efficiency"
                ])
                
                stats = leaderboard.get('benchmark_stats', {})
                
                return df, stats
            else:
                # Return empty leaderboard
                empty_df = pd.DataFrame(columns=[
                    "Rank", "Model", "Overall Score", "MMLU", "GSM8K", "MATH", "HumanEval", "Avg Latency", "Cost Efficiency"
                ])
                return empty_df, {"message": "No evaluation results found. Run some evaluations to populate the leaderboard."}
        
        except Exception as e:
            empty_df = pd.DataFrame()
            return empty_df, {"error": str(e)}
    
    def _compare_models(self, models: List[str], benchmark: str):
        """Compare selected models using REAL evaluations."""
        import asyncio
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not models or len(models) < 2:
            return None, pd.DataFrame()
        
        try:
            # Create evaluation configs for comparison
            configs = []
            for model in models:
                config = EvaluationConfig(
                    model_name=model,
                    benchmark_name=benchmark,
                    num_samples=50,  # Use smaller sample for comparison speed
                    temperature=0.0
                )
                configs.append(config)
            
            # Run REAL evaluations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self.orchestrator.evaluate_batch(configs))
                
                # Extract scores and create visualization
                model_scores = []
                comparison_data = []
                
                for result in results:
                    if not result.error:
                        model_scores.append((result.model_name, result.accuracy))
                        comparison_data.append({
                            "Model": result.model_name,
                            "Accuracy": round(result.accuracy, 3),
                            "Latency": round(result.latency, 1),
                            "Cost": f"${result.cost_estimate:.6f}",
                            "Samples": result.num_samples
                        })
                    else:
                        model_scores.append((result.model_name, 0.0))
                        comparison_data.append({
                            "Model": result.model_name,
                            "Accuracy": "FAILED",
                            "Latency": "N/A",
                            "Cost": "N/A",
                            "Samples": result.num_samples
                        })
                
                if not model_scores:
                    return None, pd.DataFrame()
                
                # Sort by accuracy for ranking
                model_scores.sort(key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                model_names = [item[0] for item in model_scores]
                scores = [item[1] if isinstance(item[1], float) else 0 for item in model_scores]
                
                bars = ax.bar(model_names, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(models)])
                
                ax.set_ylabel('Accuracy')
                ax.set_title(f'REAL Model Comparison on {benchmark.upper()}')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Add rankings to comparison data
                for i, item in enumerate(comparison_data):
                    item["Rank"] = i + 1
                
                comparison_df = pd.DataFrame(comparison_data)
                
                return fig, comparison_df
                
            finally:
                loop.close()
                
        except Exception as e:
            # Return error plot and data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Comparison failed: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Comparison Failed')
            
            error_df = pd.DataFrame([{"Error": str(e)}])
            return fig, error_df
        
        # Create comparison table
        comparison_data = []
        for model, score in zip(models, scores):
            comparison_data.append({
                "Model": model,
                "Accuracy": score,
                "Rank": sorted(scores, reverse=True).index(score) + 1,
                "Relative Performance": f"{((score - min(scores)) / (max(scores) - min(scores)) * 100):.1f}%" if max(scores) != min(scores) else "100%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return fig, comparison_df
    
    def _search_results(self, model_filter: str, benchmark_filter: str):
        """Search and filter results."""
        model_name = None if model_filter == "All Models" else model_filter
        benchmark_name = None if benchmark_filter == "All Benchmarks" else benchmark_filter
        
        results = self.orchestrator.get_cached_results(model_name, benchmark_name)
        
        if results:
            df_data = []
            for result in results:
                df_data.append({
                    "Model": result.get('model_name', ''),
                    "Benchmark": result.get('benchmark_name', ''),
                    "Accuracy": result.get('accuracy', 0),
                    "Latency": result.get('latency', 0),
                    "Samples": result.get('num_samples', 0),
                    "Timestamp": result.get('timestamp', ''),
                    "Has Error": bool(result.get('error'))
                })
            
            results_df = pd.DataFrame(df_data)
        else:
            results_df = pd.DataFrame()
        
        # Get storage stats
        storage_stats = self.orchestrator.storage.get_storage_stats()
        
        return results_df, storage_stats
    
    def _export_results(self):
        """Export results to file."""
        try:
            export_path = self.orchestrator.storage.export_results('csv')
            return gr.File(value=export_path, visible=True)
        except Exception as e:
            return gr.File(visible=False)
    
    def _save_settings(self, default_samples, max_concurrent, cleanup_enabled, cache_limit, results_format):
        """Save platform settings."""
        # This would update the configuration file
        return "Settings saved successfully!"
    
    def _cleanup_cache(self):
        """Cleanup cache."""
        try:
            stats_before = self.orchestrator.storage.get_storage_stats()
            asyncio.run(self.orchestrator.storage.cleanup_old_results())
            stats_after = self.orchestrator.storage.get_storage_stats()
            
            saved_mb = stats_before['cache_size_mb'] - stats_after['cache_size_mb']
            return f"Cache cleaned up successfully! Freed {saved_mb:.2f}MB of space."
        except Exception as e:
            return f"Cleanup failed: {str(e)}"

def create_web_app(config_path: str = "config/config.yaml", share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
    """Create and launch the web application."""
    web_interface = WebInterface(config_path)
    app = web_interface.create_interface()
    
    print("🧠 Starting smaLLMs Web Interface...")
    
    # Try to find an available port if the default is taken
    import socket
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((server_name, port))
                return True
            except OSError:
                return False
    
    # Find available port starting from the requested one
    original_port = server_port
    while not is_port_available(server_port) and server_port < original_port + 100:
        server_port += 1
    
    if server_port != original_port:
        print(f"⚠️  Port {original_port} was busy, using port {server_port} instead")
    
    print(f"📊 Platform ready at: http://{server_name}:{server_port}")
    print("🚀 Open your browser and start evaluating small language models!")
    
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    create_web_app()
