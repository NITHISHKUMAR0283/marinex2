#!/usr/bin/env python3
"""
FloatChat Test Runner
Comprehensive test execution with detailed reporting
"""

import pytest
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

class TestReporter:
    """Custom test reporter for detailed results."""
    
    def __init__(self):
        self.results = {
            'start_time': None,
            'end_time': None,
            'duration': 0,
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'performance_metrics': {},
            'test_categories': {}
        }
    
    def start_testing(self):
        """Start test execution."""
        self.results['start_time'] = datetime.now()
        print("🌊 FloatChat Test Suite")
        print("=" * 60)
        print(f"Started at: {self.results['start_time']}")
        print()
    
    def end_testing(self):
        """End test execution."""
        self.results['end_time'] = datetime.now()
        self.results['duration'] = (
            self.results['end_time'] - self.results['start_time']
        ).total_seconds()
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        # Basic statistics
        print(f"Duration: {self.results['duration']:.2f} seconds")
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"⏭️ Skipped: {self.results['skipped']}")
        
        # Success rate
        if self.results['total_tests'] > 0:
            success_rate = (self.results['passed'] / self.results['total_tests']) * 100
            print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        print("\n" + "-" * 40)
        print("📋 TEST CATEGORIES")
        print("-" * 40)
        
        for category, stats in self.results['test_categories'].items():
            print(f"{category}:")
            print(f"  ✅ Passed: {stats['passed']}")
            print(f"  ❌ Failed: {stats['failed']}")
            print(f"  ⏭️ Skipped: {stats['skipped']}")
        
        # Performance metrics
        if self.results['performance_metrics']:
            print("\n" + "-" * 40)
            print("⚡ PERFORMANCE METRICS")
            print("-" * 40)
            
            for metric, value in self.results['performance_metrics'].items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.3f}s")
                else:
                    print(f"{metric}: {value}")
        
        # Failures and errors
        if self.results['errors']:
            print("\n" + "-" * 40)
            print("🚨 FAILURES AND ERRORS")
            print("-" * 40)
            
            for error in self.results['errors']:
                print(f"❌ {error['test']}: {error['error']}")
        
        print("\n" + "=" * 60)
        
        # Overall status
        if self.results['failed'] == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {self.results['failed']} TEST(S) FAILED")
        
        print("=" * 60)
    
    def save_report(self, output_file: str):
        """Save detailed report to JSON file."""
        report_data = {
            **self.results,
            'start_time': self.results['start_time'].isoformat() if self.results['start_time'] else None,
            'end_time': self.results['end_time'].isoformat() if self.results['end_time'] else None
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Detailed report saved to: {output_file}")

def run_unit_tests(reporter: TestReporter):
    """Run unit tests."""
    print("🧪 Running Unit Tests...")
    
    unit_test_files = [
        "test_embeddings.py",
        "test_vector_database.py"
    ]
    
    unit_results = {'passed': 0, 'failed': 0, 'skipped': 0}
    
    for test_file in unit_test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"  Running {test_file}...")
            
            # Run pytest and capture results
            exit_code = pytest.main([
                str(test_path),
                "-v",
                "--tb=short",
                "-x"  # Stop on first failure for quick feedback
            ])
            
            if exit_code == 0:
                unit_results['passed'] += 1
                print(f"  ✅ {test_file} passed")
            else:
                unit_results['failed'] += 1
                print(f"  ❌ {test_file} failed")
                reporter.results['errors'].append({
                    'test': test_file,
                    'error': f"Unit test failed with exit code {exit_code}"
                })
        else:
            print(f"  ⏭️ {test_file} not found, skipping")
            unit_results['skipped'] += 1
    
    reporter.results['test_categories']['Unit Tests'] = unit_results
    return unit_results

def run_integration_tests(reporter: TestReporter):
    """Run integration tests."""
    print("\n🔗 Running Integration Tests...")
    
    integration_results = {'passed': 0, 'failed': 0, 'skipped': 0}
    
    test_path = test_dir / "test_integration.py"
    if test_path.exists():
        print("  Running integration tests...")
        
        start_time = time.time()
        exit_code = pytest.main([
            str(test_path),
            "-v",
            "--tb=short"
        ])
        end_time = time.time()
        
        reporter.results['performance_metrics']['integration_test_time'] = end_time - start_time
        
        if exit_code == 0:
            integration_results['passed'] = 1
            print("  ✅ Integration tests passed")
        else:
            integration_results['failed'] = 1
            print("  ❌ Integration tests failed")
            reporter.results['errors'].append({
                'test': 'Integration Tests',
                'error': f"Integration tests failed with exit code {exit_code}"
            })
    else:
        print("  ⏭️ Integration tests not found, skipping")
        integration_results['skipped'] = 1
    
    reporter.results['test_categories']['Integration Tests'] = integration_results
    return integration_results

def run_performance_tests(reporter: TestReporter):
    """Run performance tests."""
    print("\n⚡ Running Performance Tests...")
    
    performance_results = {'passed': 0, 'failed': 0, 'skipped': 0}
    
    test_path = test_dir / "performance_tests.py"
    if test_path.exists():
        print("  Running performance benchmarks...")
        
        start_time = time.time()
        exit_code = pytest.main([
            str(test_path),
            "-v",
            "--tb=short",
            "-m", "not slow"  # Skip slow tests by default
        ])
        end_time = time.time()
        
        reporter.results['performance_metrics']['performance_test_time'] = end_time - start_time
        
        if exit_code == 0:
            performance_results['passed'] = 1
            print("  ✅ Performance tests passed")
        else:
            performance_results['failed'] = 1
            print("  ❌ Performance tests failed")
            reporter.results['errors'].append({
                'test': 'Performance Tests',
                'error': f"Performance tests failed with exit code {exit_code}"
            })
    else:
        print("  ⏭️ Performance tests not found, skipping")
        performance_results['skipped'] = 1
    
    reporter.results['test_categories']['Performance Tests'] = performance_results
    return performance_results

def run_system_validation():
    """Run system validation checks."""
    print("\n🔍 Running System Validation...")
    
    validation_results = []
    
    # Check required dependencies
    print("  Checking dependencies...")
    required_packages = [
        'numpy', 'pandas', 'streamlit', 'plotly', 
        'faiss-cpu', 'openai', 'anthropic', 'groq',
        'psycopg2-binary', 'sqlalchemy', 'asyncio'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"    ✅ {package}")
        except ImportError:
            print(f"    ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        validation_results.append({
            'check': 'Dependencies',
            'status': 'FAILED',
            'details': f"Missing packages: {', '.join(missing_packages)}"
        })
    else:
        validation_results.append({
            'check': 'Dependencies',
            'status': 'PASSED',
            'details': 'All required packages available'
        })
    
    # Check file structure
    print("  Checking file structure...")
    required_files = [
        'src/floatchat/__init__.py',
        'src/floatchat/ai/embeddings/multi_modal_embeddings.py',
        'src/floatchat/ai/vector_database/faiss_vector_store.py',
        'src/floatchat/ai/rag/rag_pipeline.py',
        'src/floatchat/ai/llm/llm_orchestrator.py',
        'src/floatchat/ai/nl2sql/nl2sql_engine.py',
        'src/floatchat/interface/streamlit_app.py',
        'main.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = test_dir.parent / file_path
        if full_path.exists():
            print(f"    ✅ {file_path}")
        else:
            print(f"    ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        validation_results.append({
            'check': 'File Structure',
            'status': 'FAILED',
            'details': f"Missing files: {', '.join(missing_files)}"
        })
    else:
        validation_results.append({
            'check': 'File Structure', 
            'status': 'PASSED',
            'details': 'All required files present'
        })
    
    # Check configuration
    print("  Checking configuration...")
    config_checks = []
    
    # Environment variables
    env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DATABASE_URL']
    for var in env_vars:
        if os.getenv(var):
            print(f"    ✅ {var} configured")
            config_checks.append(f"{var}: configured")
        else:
            print(f"    ⚠️ {var} not set (may use defaults)")
            config_checks.append(f"{var}: not set")
    
    validation_results.append({
        'check': 'Configuration',
        'status': 'PASSED',
        'details': '; '.join(config_checks)
    })
    
    print("\n  📋 Validation Summary:")
    for result in validation_results:
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"    {status_icon} {result['check']}: {result['status']}")
        if result['details']:
            print(f"       {result['details']}")
    
    return validation_results

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='FloatChat Test Suite')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--performance', action='store_true', help='Run only performance tests')
    parser.add_argument('--validation', action='store_true', help='Run only system validation')
    parser.add_argument('--all', action='store_true', help='Run all tests (default)')
    parser.add_argument('--report', type=str, help='Save detailed report to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Default to running all tests
    if not any([args.unit, args.integration, args.performance, args.validation]):
        args.all = True
    
    reporter = TestReporter()
    reporter.start_testing()
    
    try:
        # Run system validation first
        if args.validation or args.all:
            validation_results = run_system_validation()
        
        # Run unit tests
        if args.unit or args.all:
            unit_results = run_unit_tests(reporter)
            reporter.results['total_tests'] += sum(unit_results.values())
            reporter.results['passed'] += unit_results['passed']
            reporter.results['failed'] += unit_results['failed']
            reporter.results['skipped'] += unit_results['skipped']
        
        # Run integration tests
        if args.integration or args.all:
            integration_results = run_integration_tests(reporter)
            reporter.results['total_tests'] += sum(integration_results.values())
            reporter.results['passed'] += integration_results['passed']
            reporter.results['failed'] += integration_results['failed']
            reporter.results['skipped'] += integration_results['skipped']
        
        # Run performance tests
        if args.performance or args.all:
            performance_results = run_performance_tests(reporter)
            reporter.results['total_tests'] += sum(performance_results.values())
            reporter.results['passed'] += performance_results['passed']
            reporter.results['failed'] += performance_results['failed']
            reporter.results['skipped'] += performance_results['skipped']
        
    except KeyboardInterrupt:
        print("\n⏹️ Test execution interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        reporter.results['errors'].append({
            'test': 'Test Runner',
            'error': str(e)
        })
        return 1
    
    finally:
        reporter.end_testing()
        reporter.generate_report()
        
        # Save detailed report if requested
        if args.report:
            reporter.save_report(args.report)
    
    # Return exit code based on test results
    return 1 if reporter.results['failed'] > 0 else 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)