"""
Storage management for smaLLMs platform.
Handles local caching and cloud storage with minimal space usage.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from huggingface_hub import HfApi
import logging

class ResultStorage:
    """Manages storage of evaluation results with space optimization."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage configuration
        storage_config = config.get('storage', {})
        self.local_cache_mb = storage_config.get('local_cache_mb', 500)
        self.results_format = storage_config.get('results_format', 'json')
        self.cleanup_old_results = storage_config.get('cleanup_old_results', True)
        
        # Paths
        self.results_dir = Path('results')
        self.cache_dir = Path('results/cache')
        self.leaderboard_dir = Path('results/leaderboards')
        
        # Create directories
        self._create_directories()
        
        # Hugging Face integration
        hf_config = config.get('storage', {}).get('huggingface_datasets', {})
        self.upload_results = hf_config.get('upload_results', True)
        self.dataset_name = hf_config.get('dataset_name', 'smaLLMs-results')
        
        # Initialize HF API if uploading
        if self.upload_results:
            try:
                self.hf_api = HfApi(token=config.get('huggingface', {}).get('token'))
            except Exception as e:
                self.logger.warning(f"Failed to initialize HF API: {e}")
                self.upload_results = False
    
    def _create_directories(self):
        """Create necessary directories."""
        for directory in [self.results_dir, self.cache_dir, self.leaderboard_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def save_result(self, result) -> str:
        """Save evaluation result locally and optionally to cloud."""
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result.model_name.replace('/', '_')}_{result.benchmark_name}_{timestamp}"
        
        # Save locally
        local_path = self._save_local_result(result, filename)
        
        # Upload to cloud if enabled
        if self.upload_results:
            try:
                await self._upload_to_cloud(result, filename)
            except Exception as e:
                self.logger.warning(f"Failed to upload result to cloud: {e}")
        
        # Cleanup if needed
        if self.cleanup_old_results:
            await self._cleanup_if_needed()
        
        return str(local_path)
    
    def _save_local_result(self, result, filename: str) -> Path:
        """Save result locally in specified format."""
        if self.results_format == 'json':
            file_path = self.cache_dir / f"{filename}.json"
            result_dict = {
                'model_name': result.model_name,
                'benchmark_name': result.benchmark_name,
                'accuracy': result.accuracy,
                'latency': result.latency,
                'cost_estimate': result.cost_estimate,
                'timestamp': result.timestamp,
                'num_samples': result.num_samples,
                'detailed_results': result.detailed_results[:5] if result.detailed_results else [],  # Limit to save space
                'error': result.error
            }
            
            with open(file_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
        
        elif self.results_format == 'csv':
            file_path = self.cache_dir / f"{filename}.csv"
            result_df = pd.DataFrame([{
                'model_name': result.model_name,
                'benchmark_name': result.benchmark_name,
                'accuracy': result.accuracy,
                'latency': result.latency,
                'cost_estimate': result.cost_estimate,
                'timestamp': result.timestamp,
                'num_samples': result.num_samples,
                'has_error': bool(result.error)
            }])
            result_df.to_csv(file_path, index=False)
        
        self.logger.info(f"Saved result locally: {file_path}")
        return file_path
    
    async def _upload_to_cloud(self, result, filename: str):
        """Upload result to Hugging Face Dataset."""
        if not self.upload_results:
            return
        
        try:
            # Prepare data for upload
            result_data = {
                'model_name': result.model_name,
                'benchmark_name': result.benchmark_name,
                'accuracy': result.accuracy,
                'latency': result.latency,
                'cost_estimate': result.cost_estimate,
                'timestamp': result.timestamp,
                'num_samples': result.num_samples,
                'has_error': bool(result.error)
            }
            
            # Create/update dataset
            df = pd.DataFrame([result_data])
            
            # For now, save to a local file that can be uploaded
            # In a full implementation, this would use the datasets library
            cloud_file = self.results_dir / 'cloud_upload.jsonl'
            
            # Append to JSONL file
            with open(cloud_file, 'a') as f:
                f.write(json.dumps(result_data) + '\n')
            
            self.logger.info(f"Prepared result for cloud upload: {filename}")
        
        except Exception as e:
            self.logger.error(f"Failed to upload to cloud: {e}")
    
    async def save_leaderboard(self, leaderboard_data: Dict[str, Any]) -> str:
        """Save leaderboard data."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"leaderboard_{timestamp}.json"
        file_path = self.leaderboard_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(leaderboard_data, f, indent=2)
        
        # Also save as latest
        latest_path = self.leaderboard_dir / 'latest.json'
        with open(latest_path, 'w') as f:
            json.dump(leaderboard_data, f, indent=2)
        
        self.logger.info(f"Saved leaderboard: {file_path}")
        return str(file_path)
    
    def get_cached_results(self, model_name: str = None, benchmark_name: str = None) -> List[Dict[str, Any]]:
        """Retrieve cached results with optional filtering."""
        results = []
        
        # Scan cache directory
        for file_path in self.cache_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                
                # Apply filters
                if model_name and result.get('model_name') != model_name:
                    continue
                if benchmark_name and result.get('benchmark_name') != benchmark_name:
                    continue
                
                results.append(result)
            
            except Exception as e:
                self.logger.warning(f"Failed to load cached result {file_path}: {e}")
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return results
    
    def get_latest_leaderboard(self) -> Optional[Dict[str, Any]]:
        """Get the latest leaderboard."""
        latest_path = self.leaderboard_dir / 'latest.json'
        
        if latest_path.exists():
            try:
                with open(latest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load latest leaderboard: {e}")
        
        return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        def get_directory_size(path: Path) -> int:
            """Get total size of directory in bytes."""
            total = 0
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total += file_path.stat().st_size
            return total
        
        results_size = get_directory_size(self.results_dir)
        cache_size = get_directory_size(self.cache_dir)
        leaderboard_size = get_directory_size(self.leaderboard_dir)
        
        # Count files
        cache_files = len(list(self.cache_dir.glob('*.json')))
        leaderboard_files = len(list(self.leaderboard_dir.glob('*.json')))
        
        return {
            'total_size_mb': round(results_size / (1024 * 1024), 2),
            'cache_size_mb': round(cache_size / (1024 * 1024), 2),
            'leaderboard_size_mb': round(leaderboard_size / (1024 * 1024), 2),
            'cache_files': cache_files,
            'leaderboard_files': leaderboard_files,
            'cache_limit_mb': self.local_cache_mb,
            'cache_usage_percent': round((cache_size / (1024 * 1024)) / self.local_cache_mb * 100, 1)
        }
    
    async def cleanup_old_results(self):
        """Clean up old results to stay within storage limits."""
        stats = self.get_storage_stats()
        
        if stats['cache_size_mb'] > self.local_cache_mb:
            self.logger.info(f"Cache size ({stats['cache_size_mb']}MB) exceeds limit ({self.local_cache_mb}MB), cleaning up...")
            
            # Get all cache files with timestamps
            cache_files = []
            for file_path in self.cache_dir.glob('*.json'):
                try:
                    with open(file_path, 'r') as f:
                        result = json.load(f)
                    timestamp = result.get('timestamp', '')
                    cache_files.append((file_path, timestamp))
                except:
                    # If we can't read the file, consider it for deletion
                    cache_files.append((file_path, ''))
            
            # Sort by timestamp (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Delete oldest files until we're under the limit
            bytes_to_delete = (stats['cache_size_mb'] - self.local_cache_mb * 0.8) * 1024 * 1024  # Leave 20% buffer
            bytes_deleted = 0
            
            for file_path, timestamp in cache_files:
                if bytes_deleted >= bytes_to_delete:
                    break
                
                file_size = file_path.stat().st_size
                file_path.unlink()
                bytes_deleted += file_size
                self.logger.debug(f"Deleted old cache file: {file_path}")
            
            self.logger.info(f"Cleanup completed: deleted {bytes_deleted / (1024 * 1024):.2f}MB")
    
    async def _cleanup_if_needed(self):
        """Check if cleanup is needed and perform it."""
        stats = self.get_storage_stats()
        
        if stats['cache_usage_percent'] > 90:  # Cleanup when 90% full
            await self.cleanup_old_results()
    
    def export_results(self, format: str = 'csv', output_path: str = None) -> str:
        """Export all cached results to a single file."""
        all_results = self.get_cached_results()
        
        if not all_results:
            raise ValueError("No cached results found to export")
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"smaLLMs_results_export_{timestamp}.{format}"
        
        if format == 'csv':
            # Convert to DataFrame
            df_data = []
            for result in all_results:
                df_data.append({
                    'model_name': result.get('model_name', ''),
                    'benchmark_name': result.get('benchmark_name', ''),
                    'accuracy': result.get('accuracy', 0),
                    'latency': result.get('latency', 0),
                    'cost_estimate': result.get('cost_estimate', 0),
                    'timestamp': result.get('timestamp', ''),
                    'num_samples': result.get('num_samples', 0),
                    'has_error': bool(result.get('error'))
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
        
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(all_results)} results to {output_path}")
        return output_path
    
    def clear_cache(self):
        """Clear all cached results (use with caution)."""
        for file_path in self.cache_dir.glob('*'):
            if file_path.is_file():
                file_path.unlink()
        
        self.logger.warning("Cleared all cached results")
    
    def backup_results(self, backup_path: str = None) -> str:
        """Create a backup of all results."""
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"smaLLMs_backup_{timestamp}"
        
        # Create zip archive
        shutil.make_archive(backup_path, 'zip', self.results_dir)
        backup_file = f"{backup_path}.zip"
        
        self.logger.info(f"Created backup: {backup_file}")
        return backup_file
