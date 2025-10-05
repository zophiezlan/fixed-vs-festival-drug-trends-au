"""
Basic tests to verify the drug checking analysis pipeline works correctly.
"""
import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate_data import generate_fixed_site_data, generate_festival_data
from analysis import DrugCheckingAnalyzer

def test_data_generation():
    """Test that data generation creates valid DataFrames."""
    print("Testing data generation...")
    
    # Generate small datasets
    fixed_df = generate_fixed_site_data(50)
    festival_df = generate_festival_data(40)
    
    # Check structure
    assert len(fixed_df) == 50, "Fixed-site data should have 50 samples"
    assert len(festival_df) == 40, "Festival data should have 40 samples"
    
    # Check columns
    expected_columns = ['date', 'service_type', 'substance_detected', 
                       'expected_substance', 'is_nps', 'sample_form', 
                       'adulterants', 'num_adulterants']
    assert list(fixed_df.columns) == expected_columns, "Fixed-site columns mismatch"
    assert list(festival_df.columns) == expected_columns, "Festival columns mismatch"
    
    # Check service types
    assert all(fixed_df['service_type'] == 'Fixed-site'), "Service type should be Fixed-site"
    assert all(festival_df['service_type'] == 'Festival'), "Service type should be Festival"
    
    print("✓ Data generation test passed")
    return True

def test_analysis_basic():
    """Test basic analysis functionality."""
    print("Testing analysis module...")
    
    # Create test data
    test_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'service_type': ['Fixed-site', 'Fixed-site', 'Festival', 'Festival'],
        'substance_detected': ['MDMA', '2C-B', 'MDMA', 'Cocaine'],
        'expected_substance': ['MDMA', 'MDMA', 'MDMA', 'Cocaine'],
        'is_nps': [False, True, False, False],
        'sample_form': ['Pill', 'Powder', 'Pill', 'Powder'],
        'adulterants': ['None', 'Caffeine', 'None', 'None'],
        'num_adulterants': [0, 1, 0, 0]
    })
    
    # Save test data
    os.makedirs('test_data', exist_ok=True)
    test_data.to_csv('test_data/test_combined.csv', index=False)
    
    # Create analyzer
    analyzer = DrugCheckingAnalyzer('test_data/test_combined.csv')
    
    # Test basic methods
    comparison = analyzer.get_service_comparison()
    assert 'Fixed-site' in comparison, "Fixed-site should be in comparison"
    assert 'Festival' in comparison, "Festival should be in comparison"
    
    diversity = analyzer.calculate_diversity_index('Fixed-site')
    assert 'shannon_diversity' in diversity, "Should calculate Shannon diversity"
    assert 'simpson_diversity' in diversity, "Should calculate Simpson diversity"
    
    top_substances = analyzer.get_top_substances('Fixed-site', top_n=5)
    assert len(top_substances) > 0, "Should return top substances"
    
    # Cleanup
    import shutil
    shutil.rmtree('test_data')
    
    print("✓ Analysis module test passed")
    return True

def test_full_pipeline():
    """Test that the full pipeline can run without errors."""
    print("Testing full pipeline...")
    
    # Import main components
    from generate_data import main as generate_data
    from visualization import DrugCheckingVisualizer
    
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Run data generation
    generate_data()
    
    # Check data files were created
    assert os.path.exists('data/fixed_site_data.csv'), "Fixed-site data not created"
    assert os.path.exists('data/festival_data.csv'), "Festival data not created"
    assert os.path.exists('data/combined_data.csv'), "Combined data not created"
    
    # Load and analyze
    analyzer = DrugCheckingAnalyzer('data/combined_data.csv')
    
    # Generate report
    report = analyzer.generate_summary_report()
    assert len(report) > 0, "Report should have content"
    assert "Fixed-site" in report, "Report should mention Fixed-site"
    assert "Festival" in report, "Report should mention Festival"
    
    # Test visualization creation (just instantiation)
    visualizer = DrugCheckingVisualizer(analyzer)
    assert visualizer is not None, "Visualizer should be created"
    
    print("✓ Full pipeline test passed")
    return True

def main():
    """Run all tests."""
    print("=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_data_generation,
        test_analysis_basic,
        test_full_pipeline
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
