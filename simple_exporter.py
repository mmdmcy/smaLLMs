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
                        
                        print(f"📊 Loaded {len(report.get('detailed_results', []))} results from organized run: {latest_run.name}")
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
            print("📂 No organized results found, checking legacy structure...")
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
                        
                        print(f"📊 Loaded {len(report.get('detailed_results', []))} results from legacy file: {latest_report.name}")
                except Exception as e:
                    print(f"Error loading legacy report {latest_report}: {e}")
        
        # Final fallback to cache
        if not all_results:
            print("📂 No reports found, loading from cache...")
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
        """Create a clean leaderboard from results."""
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
                    'evaluations': [],
                    'benchmarks': {},
                    'total_cost': 0.0,
                    'total_time': 0.0,
                    'eval_count': 0
                }
            
            model_data[model]['evaluations'].append(result)
            model_data[model]['benchmarks'][benchmark] = float(accuracy)
            model_data[model]['total_cost'] += float(cost)
            model_data[model]['total_time'] += float(duration)
            model_data[model]['eval_count'] += 1
        
        # Convert to leaderboard
        leaderboard_data = []
        
        for model, data in model_data.items():
            # Calculate metrics
            benchmark_scores = [score for score in data['benchmarks'].values() if score > 0]
            avg_accuracy = np.mean(benchmark_scores) if benchmark_scores else 0.0
            success_rate = len(benchmark_scores) / data['eval_count'] if data['eval_count'] > 0 else 0.0
            avg_cost = data['total_cost'] / data['eval_count'] if data['eval_count'] > 0 else 0.0
            avg_duration = data['total_time'] / data['eval_count'] if data['eval_count'] > 0 else 0.0
            
            # Clean model name for display
            clean_model_name = model.split('/')[-1] if '/' in model else model
            
            entry = {
                'Model': clean_model_name,
                'Full_Model_Name': model,
                'Overall_Accuracy': round(float(avg_accuracy), 3),
                'Success_Rate': round(float(success_rate), 3),
                'Evaluations': int(data['eval_count']),
                'Total_Cost': round(float(data['total_cost']), 4),
                'Avg_Cost': round(float(avg_cost), 6),
                'Avg_Duration_Seconds': round(float(avg_duration), 1),
                'Cost_Efficiency': round(float(success_rate / (avg_cost + 0.001)), 2)
            }
            
            # Add benchmark-specific scores
            for benchmark, score in data['benchmarks'].items():
                entry[f'{benchmark.upper()}'] = round(float(score), 3)
            
            leaderboard_data.append(entry)
        
        # Create DataFrame and sort
        df = pd.DataFrame(leaderboard_data)
        if not df.empty:
            df = df.sort_values(['Overall_Accuracy', 'Success_Rate'], ascending=[False, False])
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
        
        print(f"📤 Exporting to organized structure: {run_output.absolute()}")
        
        # Load results
        data = self.load_latest_results()
        results = data.get('results', [])
        
        if not results:
            print("❌ No results found to export!")
            return {}
        
        # Create leaderboard
        df = self.create_clean_leaderboard(results)
        
        if df.empty:
            print("❌ No valid data to export!")
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
        print(f"✅ CSV exported: {csv_file.name}")
        
        # 2. JSON Export (for web APIs)
        # Convert DataFrame to clean dict with Python types
        leaderboard_records = []
        for _, row in df.iterrows():
            record = {}
            for col, val in row.items():
                # Convert numpy types to Python types
                if pd.isna(val):
                    record[col] = None
                elif isinstance(val, (np.integer, np.int64, np.int32)):
                    record[col] = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    record[col] = float(val)
                elif isinstance(val, np.ndarray):
                    record[col] = val.tolist()
                else:
                    record[col] = val
            leaderboard_records.append(record)
        
        json_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_models': int(len(df)),
                'total_evaluations': int(df['Evaluations'].sum()),
                'evaluation_date': timestamp,
                'evaluation_details': self.eval_metadata,  # Include evaluation context
                'export_structure': 'organized',
                'export_path': str(run_output)
            },
            'leaderboard': leaderboard_records
        }
        
        json_file = run_output / f"{base_name}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        exported_files['json'] = str(json_file)
        print(f"✅ JSON exported: {json_file.name}")
        
        # 3. Markdown Export (for copy-paste to AI)
        md_file = run_output / f"{base_name}.md"
        with open(md_file, 'w') as f:
            f.write("# smaLLMs Small Language Model Evaluation Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Include evaluation context
            if self.eval_metadata:
                f.write("## Evaluation Details\n\n")
                f.write(f"- **Evaluation Type:** {self.eval_metadata.get('evaluation_type', 'Unknown')}\n")
                f.write(f"- **Run Folder:** {self.eval_metadata.get('run_folder', 'Unknown')}\n")
                f.write(f"- **Date:** {self.eval_metadata.get('date_folder', 'Unknown')}\n")
                f.write(f"- **Total Cost:** ${self.eval_metadata.get('total_cost', 0):.4f}\n")
                f.write(f"- **Duration:** {self.eval_metadata.get('total_duration', 0):.1f} minutes\n")
                f.write(f"- **Export Path:** {run_output}\n\n")
            
            f.write(f"**Models Evaluated:** {len(df)}\n")
            f.write(f"**Total Evaluations:** {df['Evaluations'].sum()}\n\n")
            
            f.write("## Leaderboard\n\n")
            f.write(df.to_markdown(index=True))
            
            f.write("\n\n## Key Insights\n\n")
            
            # Add some analysis
            best_model = df.iloc[0]
            f.write(f"- **Best Overall Model:** {best_model['Model']} ({best_model['Overall_Accuracy']:.3f} accuracy)\n")
            f.write(f"- **Most Cost Efficient:** {df.loc[df['Cost_Efficiency'].idxmax(), 'Model']}\n")
            f.write(f"- **Highest Success Rate:** {df.loc[df['Success_Rate'].idxmax(), 'Model']}\n")
            
            # Benchmark analysis
            benchmark_cols = [col for col in df.columns if col.isupper() and len(col) <= 10]
            for benchmark in benchmark_cols:
                if benchmark in df.columns:
                    best_in_benchmark = df.loc[df[benchmark].idxmax()]
                    f.write(f"- **Best {benchmark}:** {best_in_benchmark['Model']} ({best_in_benchmark[benchmark]:.3f})\n")
            
            f.write("\n## Evaluation Context\n\n")
            if self.eval_metadata:
                eval_type = self.eval_metadata.get('evaluation_type', 'Unknown')
                f.write(f"This evaluation used the **{eval_type}** preset ")
                if 'Lightning' in eval_type:
                    f.write("(quick 2-minute demo with 3 models, 10 samples each)")
                elif 'Quick' in eval_type:
                    f.write("(5-minute benchmark with 5 models, 25 samples each)")
                elif 'Standard' in eval_type:
                    f.write("(15-minute evaluation with 8 models, 50 samples each)")
                elif 'Comprehensive' in eval_type:
                    f.write("(45-minute thorough test with 12 models, 100 samples each)")
                else:
                    f.write("(custom configuration)")
                f.write(".\n\n")
                
                f.write(f"Results are organized in: `{run_output}`\n\n")
            
            f.write("## Notes\n\n")
            f.write("- Overall_Accuracy: Average performance across all successful evaluations\n")
            f.write("- Success_Rate: Percentage of evaluations that completed successfully\n")
            f.write("- Cost_Efficiency: Success rate per dollar spent\n")
            f.write("- Duration in seconds per evaluation\n")
        
        exported_files['markdown'] = str(md_file)
        print(f"✅ Markdown exported: {md_file.name}")
        
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
        <h3>📊 Evaluation Context</h3>
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
        <li>📊 <strong>CSV:</strong> Import into Excel or Google Sheets</li>
        <li>🔗 <strong>JSON:</strong> Use in web applications and APIs</li>
        <li>📝 <strong>Markdown:</strong> Copy-paste to AI assistants for analysis</li>
    </ul>
    
    <h2>File Organization</h2>
    <p>All files for this evaluation are organized in: <code>{run_output}</code></p>
</body>
</html>"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        exported_files['html'] = str(html_file)
        print(f"✅ HTML exported: {html_file.name}")
        
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
        print(f"✅ README created: {readme_file.name}")
        
        return exported_files
    
    def print_summary(self, df: pd.DataFrame):
        """Print a nice summary of the results."""
        print("\n" + "="*80)
        print("🏆 SMALMS EVALUATION SUMMARY")
        print("="*80)
        
        if df.empty:
            print("No results to display")
            return
        
        print(f"📊 {len(df)} models evaluated")
        print(f"🎯 Best accuracy: {df['Overall_Accuracy'].max():.3f} ({df.loc[df['Overall_Accuracy'].idxmax(), 'Model']})")
        print(f"💰 Total cost: ${df['Total_Cost'].sum():.4f}")
        print(f"⚡ Most efficient: {df.loc[df['Cost_Efficiency'].idxmax(), 'Model']}")
        
        print(f"\n📋 TOP 5 MODELS:")
        print("-"*60)
        top_5 = df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['Model']:<25} {row['Overall_Accuracy']:.3f} accuracy | ${row['Total_Cost']:.4f} cost")
        
        # Show benchmark breakdown if available
        benchmark_cols = [col for col in df.columns if col.isupper() and len(col) <= 10]
        if benchmark_cols:
            print(f"\n📈 BENCHMARK PERFORMANCE:")
            print("-"*60)
            for benchmark in benchmark_cols:
                if benchmark in df.columns and df[benchmark].max() > 0:
                    best_model = df.loc[df[benchmark].idxmax()]
                    print(f"{benchmark:<10}: {best_model['Model']:<25} {best_model[benchmark]:.3f}")

def main():
    """Main function."""
    print("📊 smaLLMs Simple Results Exporter")
    print("="*50)
    
    exporter = SimpleResultsExporter()
    
    # Load and export
    data = exporter.load_latest_results()
    results = data.get('results', [])
    
    if not results:
        print("❌ No evaluation results found!")
        print("💡 Run some evaluations first using the intelligent evaluator.")
        return
    
    # Create leaderboard
    df = exporter.create_clean_leaderboard(results)
    
    # Print summary
    exporter.print_summary(df)
    
    # Export
    exported_files = exporter.export_for_website()
    
    if exported_files:
        print(f"\n🎯 EXPORT COMPLETE!")
        print("="*50)
        for file_type, file_path in exported_files.items():
            print(f"📄 {file_type.upper()}: {Path(file_path).name}")
        
        print(f"\n💡 Next Steps:")
        print("1. Open the HTML file to view results in browser")
        print("2. Copy the JSON file to your website project")
        print("3. Use the Markdown file with AI assistants for analysis")
        print("4. Import CSV into Excel for detailed analysis")

if __name__ == "__main__":
    main()
