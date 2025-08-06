#!/usr/bin/env python3
"""
smaLLMs Simple Results Exporter
==============================
Clean, simple tool to export evaluation results for your website.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class SimpleResultsExporter:
    """Simple, reliable results exporter for website integration."""
    
    def __init__(self):
        self.results_dir = Path("smaLLMs_results")  # Unified results folder
        self.cache_dir = Path("results/cache")
        self.eval_metadata = {}  # Store evaluation metadata
        
    def find_latest_evaluation_run(self) -> Optional[Path]:
        """Find the most recent evaluation run folder."""
        if not self.results_dir.exists():
            return None
        
        # Look for date folders (YYYY-MM-DD format)
        date_folders = [d for d in self.results_dir.iterdir() 
                       if d.is_dir() and len(d.name) == 10 and d.name.count('-') == 2]
        
        if not date_folders:
            return None
        
        # Get the most recent date folder
        latest_date_folder = max(date_folders, key=lambda x: x.name)
        
        # Look for run folders within the date folder
        run_folders = [d for d in latest_date_folder.iterdir() 
                      if d.is_dir() and d.name.startswith('run_')]
        
        if not run_folders:
            return None
        
        # Get the most recent run folder
        latest_run_folder = max(run_folders, key=lambda x: x.name)
        
        return latest_run_folder
        
    def load_latest_results(self) -> Dict[str, Any]:
        """Load the most recent evaluation results from organized structure."""
        all_results = []
        metadata = {}
        
        # Try to find organized results first
        latest_run = self.find_latest_evaluation_run()
        if latest_run:
            # Load from organized structure
            reports_dir = latest_run / "reports"
            individual_dir = latest_run / "individual_results"
            
            # Load main report
            json_report = reports_dir / "full_evaluation_report.json"
            if json_report.exists():
                try:
                    with open(json_report, 'r') as f:
                        report = json.load(f)
                        if 'detailed_results' in report:
                            all_results.extend(report['detailed_results'])
                        
                        # Extract evaluation metadata from report
                        metadata = {
                            'evaluation_type': report.get('evaluation_type', 'Unknown'),
                            'total_cost': report.get('execution_summary', {}).get('total_cost', 0),
                            'total_duration': report.get('execution_summary', {}).get('total_time_minutes', 0),
                            'evaluations_completed': report.get('execution_summary', {}).get('total_evaluations', 0),
                            'models_tested': report.get('models_tested', []),
                            'report_generated': report.get('timestamp', ''),
                            'source_file': str(json_report),
                            'run_folder': str(latest_run.name),
                            'date_folder': str(latest_run.parent.name)
                        }
                        
                        print(f" Loaded {len(report.get('detailed_results', []))} results from organized run: {latest_run.name}")
                except Exception as e:
                    print(f"Error loading organized report {json_report}: {e}")
            
            # Load individual results as backup/supplement
            if individual_dir.exists():
                individual_files = list(individual_dir.glob("*.json"))
                for result_file in individual_files:
                    try:
                        with open(result_file, 'r') as f:
                            result = json.load(f)
                            # Only add if not already in main results
                            if not any(r.get('model') == result.get('model') and 
                                     r.get('benchmark') == result.get('benchmark') 
                                     for r in all_results):
                                all_results.append(result)
                    except Exception as e:
                        print(f"Error loading individual result {result_file}: {e}")
        
        # Fallback to old structure if no organized results found
        if not all_results:
            print(" No organized results found, checking legacy structure...")
            report_files = list(self.results_dir.glob("intelligent_evaluation_report_*.json"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_report, 'r') as f:
                        report = json.load(f)
                        if 'detailed_results' in report:
                            all_results.extend(report['detailed_results'])
                        
                        metadata = {
                            'evaluation_type': report.get('evaluation_type', 'Legacy'),
                            'total_cost': report.get('total_cost', 0),
                            'total_duration': report.get('total_duration', 0),
                            'evaluations_completed': report.get('evaluations_completed', 0),
                            'models_tested': report.get('models_tested', []),
                            'report_generated': report.get('timestamp', ''),
                            'source_file': latest_report.name,
                            'structure': 'legacy'
                        }
                        
                        print(f" Loaded {len(report.get('detailed_results', []))} results from legacy file: {latest_report.name}")
                except Exception as e:
                    print(f"Error loading legacy report {latest_report}: {e}")
        
        # Final fallback to cache
        if not all_results:
            print(" No reports found, loading from cache...")
            cache_files = list(self.cache_dir.glob("*.json"))
            for cache_file in sorted(cache_files, key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
                try:
                    with open(cache_file, 'r') as f:
                        result = json.load(f)
                        all_results.append(result)
                except Exception as e:
                    print(f"Error loading cache file {cache_file}: {e}")
        
        self.eval_metadata = metadata
        return {'results': all_results, 'metadata': metadata}
    
    def create_clean_leaderboard(self, results: List[Dict]) -> pd.DataFrame:
        """Create a comprehensive leaderboard with detailed statistics from results."""
        if not results:
            return pd.DataFrame()
        
        # Group by model
        model_data = {}
        
        for result in results:
            if not isinstance(result, dict):
                continue
                
            model = result.get('model', result.get('model_name', 'Unknown'))
            benchmark = result.get('benchmark', result.get('benchmark_name', 'Unknown'))
            accuracy = result.get('accuracy', 0)
            cost = result.get('cost', result.get('cost_estimate', 0))
            duration = result.get('duration', result.get('latency', 0))
            
            # Extract additional statistics
            load_time = result.get('load_time', result.get('model_load_time', 0))
            tokens_per_second = result.get('tokens_per_second', result.get('throughput', 0))
            memory_usage = result.get('memory_usage_mb', result.get('memory_usage', 0))
            model_provider = result.get('provider', result.get('model_provider', 'Unknown'))
            model_size_gb = result.get('model_size_gb', result.get('size_gb', 0))
            model_parameters = result.get('model_parameters', result.get('parameters', 'Unknown'))
            error_count = result.get('error_count', 0)
            total_requests = result.get('total_requests', result.get('num_samples', 1))
            avg_response_time = result.get('avg_response_time', duration)
            min_response_time = result.get('min_response_time', 0)
            max_response_time = result.get('max_response_time', 0)
            
            # Calculate success rate for this specific evaluation
            success_rate_this_eval = max(0, (total_requests - error_count) / max(1, total_requests))
            
            # Clean the data
            if pd.isna(accuracy) or accuracy is None:
                accuracy = 0.0
            if pd.isna(cost) or cost is None:
                cost = 0.0
            if pd.isna(duration) or duration is None:
                duration = 0.0
            
            if model not in model_data:
                model_data[model] = {
                    'model_name': model,
                    'model_provider': model_provider,
                    'model_size_gb': float(model_size_gb) if model_size_gb else 0.0,
                    'model_parameters': model_parameters,
                    'evaluations': [],
                    'benchmarks': {},
                    'benchmark_timings': {},
                    'total_cost': 0.0,
                    'total_time': 0.0,
                    'total_load_time': 0.0,
                    'total_tokens': 0.0,
                    'total_memory_usage': 0.0,
                    'total_errors': 0,
                    'total_requests': 0,
                    'eval_count': 0,
                    'response_times': [],
                    'per_benchmark_stats': {}
                }
            
            model_data[model]['evaluations'].append(result)
            model_data[model]['benchmarks'][benchmark] = float(accuracy)
            model_data[model]['benchmark_timings'][benchmark] = float(duration)
            model_data[model]['total_cost'] += float(cost)
            model_data[model]['total_time'] += float(duration)
            model_data[model]['total_load_time'] += float(load_time)
            model_data[model]['total_memory_usage'] += float(memory_usage)
            model_data[model]['total_errors'] += int(error_count)
            model_data[model]['total_requests'] += int(total_requests)
            model_data[model]['eval_count'] += 1
            
            # Track response times
            if avg_response_time > 0:
                model_data[model]['response_times'].append(float(avg_response_time))
            
            # Per-benchmark detailed stats
            if benchmark not in model_data[model]['per_benchmark_stats']:
                model_data[model]['per_benchmark_stats'][benchmark] = {
                    'accuracy': float(accuracy),
                    'duration': float(duration),
                    'cost': float(cost),
                    'success_rate': success_rate_this_eval,
                    'tokens_per_second': float(tokens_per_second),
                    'memory_usage': float(memory_usage)
                }
        
        # Convert to comprehensive leaderboard
        leaderboard_data = []
        
        for model, data in model_data.items():
            # Calculate comprehensive metrics
            benchmark_scores = [score for score in data['benchmarks'].values() if score > 0]
            avg_accuracy = np.mean(benchmark_scores) if benchmark_scores else 0.0
            overall_success_rate = max(0, (data['total_requests'] - data['total_errors']) / max(1, data['total_requests']))
            avg_cost = data['total_cost'] / data['eval_count'] if data['eval_count'] > 0 else 0.0
            avg_duration = data['total_time'] / data['eval_count'] if data['eval_count'] > 0 else 0.0
            avg_load_time = data['total_load_time'] / data['eval_count'] if data['eval_count'] > 0 else 0.0
            avg_memory = data['total_memory_usage'] / data['eval_count'] if data['eval_count'] > 0 else 0.0
            
            # Response time statistics
            response_times = data['response_times']
            median_response_time = np.median(response_times) if response_times else 0.0
            p95_response_time = np.percentile(response_times, 95) if response_times else 0.0
            response_time_std = np.std(response_times) if response_times else 0.0
            
            # Calculate efficiency metrics
            cost_efficiency = round(float(overall_success_rate / (avg_cost + 0.001)), 2)
            time_efficiency = round(float(avg_accuracy / (avg_duration + 0.1)), 4)  # Accuracy per second
            memory_efficiency = round(float(avg_accuracy / (avg_memory + 1)), 4) if avg_memory > 0 else round(avg_accuracy, 4)
            
            # Reliability score (combination of success rate and consistency)
            reliability_score = overall_success_rate * (1 - min(response_time_std / (median_response_time + 1), 0.5))
            
            # Clean model name for display
            clean_model_name = model.split('/')[-1] if '/' in model else model
            
            entry = {
                # Basic identification
                'Model': clean_model_name,
                'Full_Model_Name': model,
                'Provider': data['model_provider'],
                'Model_Size_GB': round(float(data['model_size_gb']), 2),
                'Parameters': data['model_parameters'],
                
                # Performance metrics
                'Overall_Accuracy': round(float(avg_accuracy), 3),
                'Success_Rate': round(float(overall_success_rate), 3),
                'Reliability_Score': round(float(reliability_score), 3),
                
                # Timing metrics
                'Avg_Duration_Seconds': round(float(avg_duration), 2),
                'Avg_Load_Time_Seconds': round(float(avg_load_time), 2),
                'Median_Response_Time': round(float(median_response_time), 2),
                'P95_Response_Time': round(float(p95_response_time), 2),
                'Response_Time_Std': round(float(response_time_std), 2),
                
                # Resource metrics
                'Avg_Memory_Usage_MB': round(float(avg_memory), 1),
                'Memory_Efficiency': round(float(memory_efficiency), 4),
                
                # Cost metrics
                'Total_Cost': round(float(data['total_cost']), 4),
                'Avg_Cost_Per_Eval': round(float(avg_cost), 6),
                'Cost_Efficiency': cost_efficiency,
                
                # Efficiency scores
                'Time_Efficiency': time_efficiency,  # Accuracy per second
                'Overall_Efficiency': round((cost_efficiency + time_efficiency + memory_efficiency) / 3, 3),
                
                # Evaluation statistics
                'Evaluations_Count': int(data['eval_count']),
                'Total_Requests': int(data['total_requests']),
                'Total_Errors': int(data['total_errors']),
                'Error_Rate': round(float(data['total_errors'] / max(1, data['total_requests'])), 3)
            }
            
            # Add benchmark-specific scores
            for benchmark, score in data['benchmarks'].items():
                entry[f'{benchmark.upper()}_Accuracy'] = round(float(score), 3)
                # Add timing for each benchmark
                if benchmark in data['benchmark_timings']:
                    entry[f'{benchmark.upper()}_Time_Sec'] = round(float(data['benchmark_timings'][benchmark]), 2)
                # Add detailed stats if available
                if benchmark in data['per_benchmark_stats']:
                    stats = data['per_benchmark_stats'][benchmark]
                    entry[f'{benchmark.upper()}_Success_Rate'] = round(float(stats['success_rate']), 3)
                    if stats['tokens_per_second'] > 0:
                        entry[f'{benchmark.upper()}_Tokens_Per_Sec'] = round(float(stats['tokens_per_second']), 1)
            
            leaderboard_data.append(entry)
        
        # Create DataFrame and sort by overall efficiency
        df = pd.DataFrame(leaderboard_data)
        if not df.empty:
            # Sort by multiple criteria: overall accuracy, efficiency, and reliability
            df = df.sort_values(
                ['Overall_Accuracy', 'Overall_Efficiency', 'Reliability_Score'], 
                ascending=[False, False, False]
            )
            df.reset_index(drop=True, inplace=True)
            df.index = df.index + 1  # Start from rank 1
        
        return df
    
    def export_for_website(self, output_dir: str = "website_exports") -> Dict[str, str]:
        """Export clean data for website integration with organized structure."""
        
        # Create organized export structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_folder = datetime.now().strftime("%Y-%m-%d")
        
        base_output = Path(output_dir)
        date_output = base_output / date_folder
        run_output = date_output / f"export_{timestamp}"
        
        # Create directories
        run_output.mkdir(parents=True, exist_ok=True)
        
        print(f" Exporting to organized structure: {run_output.absolute()}")
        
        # Load results
        data = self.load_latest_results()
        results = data.get('results', [])
        
        if not results:
            print(" No results found to export!")
            return {}
        
        # Create leaderboard
        df = self.create_clean_leaderboard(results)
        
        if df.empty:
            print(" No valid data to export!")
            return {}
        
        exported_files = {}
        
        # Add evaluation context to filename if available
        eval_type = self.eval_metadata.get('evaluation_type', 'Unknown').replace(' ', '_').lower()
        run_info = self.eval_metadata.get('run_folder', timestamp)
        base_name = f"smaLLMs_{eval_type}_{run_info}"
        
        # 1. CSV Export (for Excel, easy analysis)
        csv_file = run_output / f"{base_name}.csv"
        df.to_csv(csv_file, index=True)
        exported_files['csv'] = str(csv_file)
        print(f" CSV exported: {csv_file.name}")
        
        # 2. JSON Export (for web APIs)
        # Create beautiful structured JSON export
        json_data = {
            "export_metadata": {
                "generated_at": datetime.now().isoformat(),
                "format_version": "1.0",
                "description": "smaLLMs Evaluation Results Export",
                "total_models": int(len(df)),
                "total_evaluations": int(df['Evaluations'].sum()) if 'Evaluations' in df.columns else 0,
                "evaluation_date": timestamp,
                "evaluation_details": self.eval_metadata,
                "export_path": str(run_output)
            },
            "model_results": [],
            "summary_statistics": {}
        }
        
        # Process each model into structured format
        for _, row in df.iterrows():
            model_entry = {
                "model_info": {
                    "name": str(row.get('Model', 'Unknown')),
                    "full_name": str(row.get('Full_Model_Name', row.get('Model', 'Unknown'))),
                    "provider": str(row.get('Provider', 'Unknown')),
                    "model_size_gb": float(row.get('Model_Size_GB', 0)) if pd.notna(row.get('Model_Size_GB')) else 0.0,
                    "parameters": str(row.get('Parameters', 'Unknown')),
                    "type": "general"  # Could be enhanced based on model name
                },
                "performance_summary": {
                    "overall_accuracy": float(row.get('Overall_Accuracy', 0)) if pd.notna(row.get('Overall_Accuracy')) else 0.0,
                    "success_rate": float(row.get('Success_Rate', 0)) if pd.notna(row.get('Success_Rate')) else 0.0,
                    "reliability_score": float(row.get('Reliability_Score', 0)) if pd.notna(row.get('Reliability_Score')) else 0.0,
                    "overall_efficiency": float(row.get('Overall_Efficiency', 0)) if pd.notna(row.get('Overall_Efficiency')) else 0.0
                },
                "timing_metrics": {
                    "avg_duration_seconds": float(row.get('Avg_Duration_Seconds', 0)) if pd.notna(row.get('Avg_Duration_Seconds')) else 0.0,
                    "avg_load_time_seconds": float(row.get('Avg_Load_Time_Seconds', 0)) if pd.notna(row.get('Avg_Load_Time_Seconds')) else 0.0,
                    "median_response_time": float(row.get('Median_Response_Time', 0)) if pd.notna(row.get('Median_Response_Time')) else 0.0,
                    "p95_response_time": float(row.get('P95_Response_Time', 0)) if pd.notna(row.get('P95_Response_Time')) else 0.0,
                    "response_time_std": float(row.get('Response_Time_Std', 0)) if pd.notna(row.get('Response_Time_Std')) else 0.0,
                    "time_efficiency": float(row.get('Time_Efficiency', 0)) if pd.notna(row.get('Time_Efficiency')) else 0.0
                },
                "resource_usage": {
                    "avg_memory_usage_mb": float(row.get('Avg_Memory_Usage_MB', 0)) if pd.notna(row.get('Avg_Memory_Usage_MB')) else 0.0,
                    "memory_efficiency": float(row.get('Memory_Efficiency', 0)) if pd.notna(row.get('Memory_Efficiency')) else 0.0,
                    "tokens_per_second": float(row.get('Tokens_Per_Sec', 0)) if pd.notna(row.get('Tokens_Per_Sec')) else None
                },
                "cost_analysis": {
                    "total_cost": float(row.get('Total_Cost', 0)) if pd.notna(row.get('Total_Cost')) else 0.0,
                    "avg_cost_per_eval": float(row.get('Avg_Cost_Per_Eval', 0)) if pd.notna(row.get('Avg_Cost_Per_Eval')) else 0.0,
                    "cost_efficiency": float(row.get('Cost_Efficiency', 0)) if pd.notna(row.get('Cost_Efficiency')) else 0.0
                },
                "evaluation_stats": {
                    "evaluations_count": int(row.get('Evaluations_Count', 0)) if pd.notna(row.get('Evaluations_Count')) else 0,
                    "total_requests": int(row.get('Total_Requests', 0)) if pd.notna(row.get('Total_Requests')) else 0,
                    "total_errors": int(row.get('Total_Errors', 0)) if pd.notna(row.get('Total_Errors')) else 0,
                    "error_rate": float(row.get('Error_Rate', 0)) if pd.notna(row.get('Error_Rate')) else 0.0
                },
                "benchmark_results": {}
            }
            
            # Add benchmark results
            benchmark_cols = ['GSM8K', 'HUMANEVAL', 'MMLU', 'ARC', 'HELLASWAG', 'WINOGRANDE', 'TRUTHFULQA']
            for benchmark in benchmark_cols:
                accuracy_col = f"{benchmark}_Accuracy"
                time_col = f"{benchmark}_Time_Sec"
                success_col = f"{benchmark}_Success_Rate"
                tokens_col = f"{benchmark}_Tokens_Per_Sec"
                
                model_entry["benchmark_results"][benchmark] = {
                    "accuracy": float(row.get(accuracy_col, 0)) if pd.notna(row.get(accuracy_col)) else None,
                    "time_seconds": float(row.get(time_col, 0)) if pd.notna(row.get(time_col)) else None,
                    "success_rate": float(row.get(success_col, 0)) if pd.notna(row.get(success_col)) else None,
                    "tokens_per_second": float(row.get(tokens_col, 0)) if pd.notna(row.get(tokens_col)) else None
                }
            
            json_data["model_results"].append(model_entry)
        
        # Add summary statistics
        if not df.empty:
            best_accuracy_idx = df['Overall_Accuracy'].idxmax() if 'Overall_Accuracy' in df.columns else 0
            fastest_idx = df['Avg_Duration_Seconds'].idxmin() if 'Avg_Duration_Seconds' in df.columns else 0
            most_efficient_idx = df['Overall_Efficiency'].idxmax() if 'Overall_Efficiency' in df.columns else 0
            most_reliable_idx = df['Reliability_Score'].idxmax() if 'Reliability_Score' in df.columns else 0
            
            json_data["summary_statistics"] = {
                "best_overall_accuracy": {
                    "model": str(df.loc[best_accuracy_idx, 'Model']),
                    "score": float(df.loc[best_accuracy_idx, 'Overall_Accuracy']) if 'Overall_Accuracy' in df.columns else 0.0
                },
                "fastest_model": {
                    "model": str(df.loc[fastest_idx, 'Model']),
                    "avg_time": float(df.loc[fastest_idx, 'Avg_Duration_Seconds']) if 'Avg_Duration_Seconds' in df.columns else 0.0
                },
                "most_efficient": {
                    "model": str(df.loc[most_efficient_idx, 'Model']),
                    "efficiency_score": float(df.loc[most_efficient_idx, 'Overall_Efficiency']) if 'Overall_Efficiency' in df.columns else 0.0
                },
                "most_reliable": {
                    "model": str(df.loc[most_reliable_idx, 'Model']),
                    "reliability_score": float(df.loc[most_reliable_idx, 'Reliability_Score']) if 'Reliability_Score' in df.columns else 0.0
                }
            }
        
        # Export beautiful structured JSON
        json_file = run_output / f"{base_name}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        exported_files['json'] = str(json_file)
        print(f" ✅ Structured JSON exported: {json_file.name}")
        
        
        # 4. Simple HTML Export (for quick viewing)
        html_file = run_output / f"{base_name}.html"
        
        eval_context = ""
        if self.eval_metadata:
            eval_type = self.eval_metadata.get('evaluation_type', 'Unknown')
            cost = self.eval_metadata.get('total_cost', 0)
            duration = self.eval_metadata.get('total_duration', 0)
            run_folder = self.eval_metadata.get('run_folder', 'Unknown')
            eval_context = f"""
    <div style="background-color: #f0f8ff; padding: 15px; margin: 20px 0; border-radius: 5px;">
        <h3> Evaluation Context</h3>
        <p><strong>Type:</strong> {eval_type}</p>
        <p><strong>Run:</strong> {run_folder}</p>
        <p><strong>Total Cost:</strong> ${cost:.4f}</p>
        <p><strong>Duration:</strong> {duration:.1f} minutes</p>
        <p><strong>Export Path:</strong> {run_output}</p>
    </div>"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>smaLLMs Evaluation Results - {eval_type}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .number {{ text-align: right; }}
        .highlight {{ background-color: #e8f5e8; }}
    </style>
</head>
<body>
    <h1>smaLLMs Small Language Model Evaluation Results</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Models Evaluated:</strong> {len(df)} | <strong>Total Evaluations:</strong> {df['Evaluations'].sum()}</p>
    
    {eval_context}
    
    {df.to_html(classes='leaderboard', table_id='results', escape=False)}
    
    <h2>Export Files</h2>
    <ul>
        <li> <strong>CSV:</strong> Import into Excel or Google Sheets for detailed analysis</li>
        <li> <strong>JSON:</strong> Structured data for web applications, APIs, and programming</li>
    </ul>
    
    <h2>File Organization</h2>
    <p>All files for this evaluation are organized in: <code>{run_output}</code></p>
</body>
</html>"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        exported_files['html'] = str(html_file)
        print(f" HTML exported: {html_file.name}")
        
        # Create a README for the export folder
        readme_file = run_output / "README.md"
        with open(readme_file, 'w') as f:
            f.write(f"# smaLLMs Export - {eval_type}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Files in this export:\n\n")
            for file_type, file_path in exported_files.items():
                filename = Path(file_path).name
                f.write(f"- **{filename}** - {file_type.upper()} format\n")
            f.write(f"\n## Source Data\n\n")
            if self.eval_metadata:
                f.write(f"- Evaluation Type: {self.eval_metadata.get('evaluation_type', 'Unknown')}\n")
                f.write(f"- Models Tested: {len(df)}\n")
                f.write(f"- Total Cost: ${self.eval_metadata.get('total_cost', 0):.4f}\n")
                f.write(f"- Duration: {self.eval_metadata.get('total_duration', 0):.1f} minutes\n")
        
        exported_files['readme'] = str(readme_file)
        print(f" README created: {readme_file.name}")
        
        return exported_files
    
    def print_summary(self, df: pd.DataFrame):
        """Print a nice summary of the results."""
        print("\n" + "="*80)
        print(" SMALMS EVALUATION SUMMARY")
        print("="*80)
        
        if df.empty:
            print("No results to display")
            return
        
        print(f" {len(df)} models evaluated")
        print(f" Best accuracy: {df['Overall_Accuracy'].max():.3f} ({df.loc[df['Overall_Accuracy'].idxmax(), 'Model']})")
        print(f" Total cost: ${df['Total_Cost'].sum():.4f}")
        print(f" Most efficient: {df.loc[df['Cost_Efficiency'].idxmax(), 'Model']}")
        
        print(f"\n TOP 5 MODELS:")
        print("-"*60)
        top_5 = df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['Model']:<25} {row['Overall_Accuracy']:.3f} accuracy | ${row['Total_Cost']:.4f} cost")
        
        # Show benchmark breakdown if available
        benchmark_cols = [col for col in df.columns if col.isupper() and len(col) <= 10]
        if benchmark_cols:
            print(f"\n BENCHMARK PERFORMANCE:")
            print("-"*60)
            for benchmark in benchmark_cols:
                if benchmark in df.columns and df[benchmark].max() > 0:
                    best_model = df.loc[df[benchmark].idxmax()]
                    print(f"{benchmark:<10}: {best_model['Model']:<25} {best_model[benchmark]:.3f}")

def main():
    """Main function."""
    print(" smaLLMs Simple Results Exporter")
    print("="*50)
    
    exporter = SimpleResultsExporter()
    
    # Load and export
    data = exporter.load_latest_results()
    results = data.get('results', [])
    
    if not results:
        print(" No evaluation results found!")
        print(" Run some evaluations first using the intelligent evaluator.")
        return
    
    # Create leaderboard
    df = exporter.create_clean_leaderboard(results)
    
    # Print summary
    exporter.print_summary(df)
    
    # Export
    exported_files = exporter.export_for_website()
    
    if exported_files:
        print(f"\n EXPORT COMPLETE!")
        print("="*50)
        for file_type, file_path in exported_files.items():
            print(f" {file_type.upper()}: {Path(file_path).name}")
        
        print(f"\n Next Steps:")
        print("1. Open the HTML file to view results in browser")
        print("2. Copy the JSON file to your website project or for analysis")
        print("3. Import CSV into Excel for detailed analysis")

if __name__ == "__main__":
    main()
