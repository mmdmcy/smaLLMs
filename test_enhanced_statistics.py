#!/usr/bin/env python3
"""
Enhanced Statistics Test Module for smaLLMs
==========================================
Tests the new comprehensive statistics collection and export functionality.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

def test_enhanced_statistics():
    """Test enhanced statistics collection and export functionality."""
    print("Testing Enhanced Statistics...")
    
    try:
        from src.evaluator import EvaluationResult
        from simple_exporter import SimpleResultsExporter
        
        # Create sample evaluation results with enhanced statistics
        sample_results = [
            EvaluationResult(
                model_name="test-model-1",
                benchmark_name="gsm8k",
                accuracy=0.85,
                latency=120.5,
                cost_estimate=0.025,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                num_samples=50,
                provider="ollama",
                model_size_gb=2.3,
                model_parameters="1.5B",
                load_time=15.2,
                avg_response_time=2.4,
                min_response_time=1.1,
                max_response_time=5.8,
                tokens_per_second=45.2,
                memory_usage_mb=2800.0,
                error_count=2,
                total_requests=50,
                success_count=48
            ),
            EvaluationResult(
                model_name="test-model-2",
                benchmark_name="mmlu",
                accuracy=0.72,
                latency=95.3,
                cost_estimate=0.018,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                num_samples=50,
                provider="lm_studio",
                model_size_gb=1.8,
                model_parameters="1.1B",
                load_time=8.7,
                avg_response_time=1.9,
                min_response_time=0.8,
                max_response_time=4.2,
                tokens_per_second=52.1,
                memory_usage_mb=2200.0,
                error_count=1,
                total_requests=50,
                success_count=49
            )
        ]
        
        print("✓ Created sample evaluation results with enhanced statistics")
        
        # Test EvaluationResult.to_dict() method
        test_result = sample_results[0]
        result_dict = test_result.to_dict()
        
        # Verify all enhanced statistics are included
        expected_fields = [
            'model_name', 'benchmark_name', 'accuracy', 'provider', 'model_size_gb',
            'model_parameters', 'load_time', 'avg_response_time', 'tokens_per_second',
            'memory_usage_mb', 'error_count', 'total_requests', 'success_count'
        ]
        
        missing_fields = [field for field in expected_fields if field not in result_dict]
        
        if missing_fields:
            print(f"✗ Missing fields in to_dict(): {missing_fields}")
            return False
        
        print(f"✓ EvaluationResult.to_dict() includes all {len(expected_fields)} enhanced fields")
        
        # Test enhanced leaderboard creation
        exporter = SimpleResultsExporter()
        
        # Convert to dict format for exporter
        result_dicts = [result.to_dict() for result in sample_results]
        
        leaderboard = exporter.create_clean_leaderboard(result_dicts)
        
        # Verify enhanced columns are present
        expected_columns = [
            'Model', 'Provider', 'Model_Size_GB', 'Parameters', 'Overall_Accuracy',
            'Success_Rate', 'Reliability_Score', 'Avg_Duration_Seconds', 'Avg_Load_Time_Seconds',
            'Median_Response_Time', 'Avg_Memory_Usage_MB', 'Memory_Efficiency',
            'Cost_Efficiency', 'Time_Efficiency', 'Overall_Efficiency', 'Error_Rate'
        ]
        
        missing_columns = [col for col in expected_columns if col not in leaderboard.columns]
        
        if missing_columns:
            print(f"✗ Missing columns in leaderboard: {missing_columns}")
            return False
        
        print(f"✓ Leaderboard includes all {len(expected_columns)} enhanced columns")
        
        # Verify calculated metrics are reasonable
        first_row = leaderboard.iloc[0]
        
        # Test that efficiency calculations make sense
        if first_row['Overall_Efficiency'] <= 0:
            print("✗ Overall_Efficiency should be positive")
            return False
        
        if first_row['Error_Rate'] < 0 or first_row['Error_Rate'] > 1:
            print("✗ Error_Rate should be between 0 and 1")
            return False
        
        print(f"✓ Calculated metrics are reasonable (Overall_Efficiency: {first_row['Overall_Efficiency']:.3f})")
        
        # Test provider information is preserved
        providers = leaderboard['Provider'].unique()
        expected_providers = {'ollama', 'lm_studio'}
        if not expected_providers.issubset(set(providers)):
            print(f"✗ Provider information not preserved. Got: {list(providers)}")
            return False
        
        print(f"✓ Provider information preserved: {list(providers)}")
        
        # Test export functionality with enhanced statistics
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_results_dir = Path(temp_dir) / "test_results"
            temp_results_dir.mkdir()
            
            # Create sample result file with enhanced statistics
            sample_result_data = result_dicts[0]  # Use the enhanced dict
            
            # Save to temporary file
            result_file = temp_results_dir / "test_result.json"
            with open(result_file, 'w') as f:
                json.dump(sample_result_data, f)
            
            # Test exporter with enhanced data
            original_results_dir = exporter.results_dir
            exporter.results_dir = temp_results_dir
            
            try:
                # Test loading and processing
                data = exporter.load_latest_results()
                results = data.get('results', [])
                
                if not results:
                    print("✗ No results loaded from test data")
                    return False
                
                # Test leaderboard creation with enhanced statistics
                test_leaderboard = exporter.create_clean_leaderboard(results)
                
                if test_leaderboard.empty:
                    print("✗ Empty leaderboard created from file data")
                    return False
                
                print("✓ Successfully loaded and processed enhanced statistics from file")
                
            finally:
                # Restore original results directory
                exporter.results_dir = original_results_dir
        
        print("\n🎉 All enhanced statistics tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_export_integration():
    """Test that enhanced statistics work with the export system."""
    print("\nTesting Export Integration...")
    
    try:
        from simple_exporter import SimpleResultsExporter
        
        # Create test data with enhanced statistics
        enhanced_data = [
            {
                'model': 'test-model-enhanced',
                'benchmark': 'test-benchmark',
                'accuracy': 0.78,
                'provider': 'ollama',
                'model_size_gb': 3.1,
                'model_parameters': '2B',
                'load_time': 12.5,
                'avg_response_time': 2.1,
                'tokens_per_second': 48.3,
                'memory_usage_mb': 3200.0,
                'error_count': 1,
                'total_requests': 25,
                'success_count': 24,
                'cost_estimate': 0.03,
                'latency': 87.2,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_samples': 25
            }
        ]
        
        exporter = SimpleResultsExporter()
        leaderboard = exporter.create_clean_leaderboard(enhanced_data)
        
        if leaderboard.empty:
            print("✗ Empty leaderboard created from enhanced data")
            return False
        
        # Verify enhanced statistics are preserved
        row = leaderboard.iloc[0]
        
        checks = [
            ('Provider', 'ollama', row['Provider']),
            ('Model_Size_GB', 3.1, row['Model_Size_GB']),
            ('Parameters', '2B', row['Parameters']),
            ('Avg_Load_Time_Seconds', 12.5, row['Avg_Load_Time_Seconds'])
        ]
        
        for field, expected, actual in checks:
            if actual != expected:
                print(f"✗ {field} mismatch: expected {expected}, got {actual}")
                return False
        
        print("✓ All enhanced statistics preserved in export")
        
        # Test that efficiency metrics are calculated
        efficiency_fields = ['Cost_Efficiency', 'Time_Efficiency', 'Overall_Efficiency', 'Memory_Efficiency']
        for field in efficiency_fields:
            if field not in leaderboard.columns:
                print(f"✗ Missing efficiency field: {field}")
                return False
            if row[field] <= 0:
                print(f"✗ Efficiency field {field} should be positive, got {row[field]}")
                return False
        
        print(f"✓ All efficiency metrics calculated correctly")
        
        print("\n🎉 Export integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Export integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner."""
    print("=" * 60)
    print("Enhanced Statistics Test Suite")
    print("=" * 60)
    
    tests = [
        test_enhanced_statistics,
        test_export_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All enhanced statistics tests passed!")
        print("The new comprehensive statistics system is working correctly.")
    else:
        print(f"\n❌ {failed} tests failed. Check the output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
