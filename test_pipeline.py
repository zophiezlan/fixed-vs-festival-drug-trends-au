"""
Basic tests to verify the drug checking analysis pipeline works correctly.
"""
import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate_data import generate_fixed_site_data, generate_festival_data
from generate_qualitative_data import generate_service_provider_interviews, generate_service_user_interviews
from analysis import DrugCheckingAnalyzer
from qualitative_analysis import QualitativeAnalyzer
from mixed_methods import MixedMethodsIntegrator

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
    from generate_qualitative_data import main as generate_qualitative_data
    from visualization import DrugCheckingVisualizer
    
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Run data generation
    generate_data()
    generate_qualitative_data()
    
    # Check data files were created
    assert os.path.exists('data/fixed_site_data.csv'), "Fixed-site data not created"
    assert os.path.exists('data/festival_data.csv'), "Festival data not created"
    assert os.path.exists('data/combined_data.csv'), "Combined data not created"
    assert os.path.exists('data/all_interviews.csv'), "Interview data not created"
    
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

def test_qualitative_generation():
    """Test qualitative data generation."""
    print("Testing qualitative data generation...")
    
    # Generate small datasets
    providers_df = generate_service_provider_interviews(n_fixed=2, n_festival=2)
    users_df = generate_service_user_interviews(n_fixed=3, n_festival=3)
    
    # Check structure
    assert len(providers_df) == 4, "Should have 4 provider interviews"
    assert len(users_df) == 6, "Should have 6 user interviews"
    
    # Check columns
    assert 'participant_type' in providers_df.columns, "Should have participant_type"
    assert 'service_type' in providers_df.columns, "Should have service_type"
    
    # Check service types
    assert 'Fixed-site' in providers_df['service_type'].values, "Should have Fixed-site providers"
    assert 'Festival' in providers_df['service_type'].values, "Should have Festival providers"
    
    print("✓ Qualitative data generation test passed")
    return True

def test_qualitative_analysis():
    """Test qualitative analysis functionality."""
    print("Testing qualitative analysis...")
    
    # Create test data
    test_data = pd.DataFrame({
        'interview_id': [1, 2, 3, 4],
        'participant_id': ['P1', 'P2', 'P3', 'P4'],
        'participant_type': ['Service Provider', 'Service Provider', 'Service User', 'Service User'],
        'service_type': ['Fixed-site', 'Festival', 'Fixed-site', 'Festival'],
        'interview_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'interview_duration_minutes': [60, 45, 30, 25],
        'theme_detection_capabilities': [
            'We detect more substances',
            'We focus on common substances',
            'I got detailed results',
            'Quick results are great'
        ]
    })
    
    # Save test data
    os.makedirs('test_data', exist_ok=True)
    test_data.to_csv('test_data/test_interviews.csv', index=False)
    
    # Create analyzer
    qual_analyzer = QualitativeAnalyzer('test_data/test_interviews.csv')
    
    # Test basic methods
    summary = qual_analyzer.get_participant_summary()
    assert 'Service Provider' in summary, "Should have service providers"
    assert 'Service User' in summary, "Should have service users"
    
    themes = qual_analyzer.extract_themes()
    assert len(themes) > 0, "Should extract themes"
    
    report = qual_analyzer.generate_qualitative_summary()
    assert len(report) > 0, "Should generate report"
    
    # Cleanup
    import shutil
    shutil.rmtree('test_data')
    
    print("✓ Qualitative analysis test passed")
    return True

def test_mixed_methods_integration():
    """Test mixed-methods integration."""
    print("Testing mixed-methods integration...")
    
    # Create test quantitative data
    quant_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'service_type': ['Fixed-site', 'Fixed-site', 'Festival', 'Festival'],
        'substance_detected': ['MDMA', '2C-B', 'MDMA', 'Cocaine'],
        'expected_substance': ['MDMA', 'MDMA', 'MDMA', 'Cocaine'],
        'is_nps': [False, True, False, False],
        'sample_form': ['Pill', 'Powder', 'Pill', 'Powder'],
        'adulterants': ['None', 'Caffeine', 'None', 'None'],
        'num_adulterants': [0, 1, 0, 0]
    })
    
    # Create test qualitative data
    qual_data = pd.DataFrame({
        'interview_id': [1, 2],
        'participant_id': ['P1', 'P2'],
        'participant_type': ['Service Provider', 'Service User'],
        'service_type': ['Fixed-site', 'Festival'],
        'interview_date': ['2024-01-01', '2024-01-02'],
        'interview_duration_minutes': [60, 30],
        'theme_detection_capabilities': [
            'We detect more substances',
            'Quick results are great'
        ]
    })
    
    # Save test data
    os.makedirs('test_data', exist_ok=True)
    quant_data.to_csv('test_data/test_quant.csv', index=False)
    qual_data.to_csv('test_data/test_qual.csv', index=False)
    
    # Create analyzers
    quant_analyzer = DrugCheckingAnalyzer('test_data/test_quant.csv')
    qual_analyzer = QualitativeAnalyzer('test_data/test_qual.csv')
    
    # Create integrator
    integrator = MixedMethodsIntegrator(quant_analyzer, qual_analyzer)
    
    # Test integration methods
    findings = integrator.generate_integrated_findings()
    assert len(findings) > 0, "Should generate integrated findings"
    
    convergence = integrator.identify_convergent_findings()
    assert len(convergence) > 0, "Should identify convergent findings"
    
    report = integrator.generate_mixed_methods_report()
    assert len(report) > 0, "Should generate mixed-methods report"
    assert "MIXED-METHODS" in report, "Report should mention mixed-methods"
    
    # Cleanup
    import shutil
    shutil.rmtree('test_data')
    
    print("✓ Mixed-methods integration test passed")
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
        test_qualitative_generation,
        test_qualitative_analysis,
        test_mixed_methods_integration,
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
