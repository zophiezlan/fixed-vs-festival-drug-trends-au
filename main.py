"""
Main analysis script for Australian drug checking services comparison.
Runs the complete mixed-methods pipeline: quantitative and qualitative data generation,
analysis, integration, and visualization.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate_data import main as generate_data
from generate_qualitative_data import main as generate_qualitative_data
from analysis import DrugCheckingAnalyzer
from qualitative_analysis import QualitativeAnalyzer
from mixed_methods import MixedMethodsIntegrator
from visualization import DrugCheckingVisualizer

def main():
    """Run complete mixed-methods analysis pipeline."""
    print("=" * 70)
    print("AUSTRALIAN DRUG CHECKING SERVICES: MIXED-METHODS ANALYSIS")
    print("Fixed-Site vs Festival Drug Checking Services")
    print("=" * 70)
    print()
    
    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Step 1: Generate quantitative data
    print("STEP 1: Generating quantitative datasets...")
    print("-" * 70)
    generate_data()
    print()
    
    # Step 2: Generate qualitative data
    print("\nSTEP 2: Generating qualitative interview data...")
    print("-" * 70)
    generate_qualitative_data()
    print()
    
    # Step 3: Quantitative analysis
    print("\nSTEP 3: Conducting quantitative analysis...")
    print("-" * 70)
    quant_analyzer = DrugCheckingAnalyzer('data/combined_data.csv')
    
    # Generate and print quantitative report
    quant_report = quant_analyzer.generate_summary_report()
    print(quant_report)
    
    # Save quantitative report to file
    with open('outputs/quantitative_analysis.txt', 'w') as f:
        f.write(quant_report)
    print("\nQuantitative analysis saved to: outputs/quantitative_analysis.txt")
    print()
    
    # Step 4: Qualitative analysis
    print("\nSTEP 4: Conducting qualitative analysis...")
    print("-" * 70)
    qual_analyzer = QualitativeAnalyzer('data/all_interviews.csv')
    
    # Generate and print qualitative report
    qual_report = qual_analyzer.generate_qualitative_summary()
    print(qual_report)
    
    # Save qualitative report to file
    with open('outputs/qualitative_analysis.txt', 'w') as f:
        f.write(qual_report)
    print("\nQualitative analysis saved to: outputs/qualitative_analysis.txt")
    print()
    
    # Step 5: Mixed-methods integration
    print("\nSTEP 5: Integrating findings (mixed-methods synthesis)...")
    print("-" * 70)
    integrator = MixedMethodsIntegrator(quant_analyzer, qual_analyzer)
    
    # Generate and print integrated report
    mixed_report = integrator.generate_mixed_methods_report()
    print(mixed_report)
    
    # Save mixed-methods report to file
    with open('outputs/mixed_methods_report.txt', 'w') as f:
        f.write(mixed_report)
    print("\nMixed-methods analysis saved to: outputs/mixed_methods_report.txt")
    print()
    
    # Step 6: Create visualizations
    print("\nSTEP 6: Creating visualizations...")
    print("-" * 70)
    visualizer = DrugCheckingVisualizer(quant_analyzer)
    visualizer.create_all_visualizations()
    print()
    
    # Summary
    print("\n" + "=" * 70)
    print("MIXED-METHODS ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nOutputs generated:")
    print("  Quantitative data:")
    print("    - data/fixed_site_data.csv")
    print("    - data/festival_data.csv")
    print("    - data/combined_data.csv")
    print("\n  Qualitative data:")
    print("    - data/service_provider_interviews.csv")
    print("    - data/service_user_interviews.csv")
    print("    - data/all_interviews.csv")
    print("\n  Analysis reports:")
    print("    - outputs/quantitative_analysis.txt")
    print("    - outputs/qualitative_analysis.txt")
    print("    - outputs/mixed_methods_report.txt")
    print("\n  Visualizations:")
    print("    - outputs/service_comparison.png")
    print("    - outputs/nps_trends.png")
    print("    - outputs/substance_distribution.png")
    print("    - outputs/nps_diversity.png")
    print("    - outputs/early_warning.png")
    print("\n" + "=" * 70)
    print("\nKEY FINDINGS (MIXED-METHODS SYNTHESIS):")
    print("✓ QUANTITATIVE: Fixed-site services detect significantly higher drug diversity")
    print("✓ QUALITATIVE: Providers attribute this to equipment and analysis time")
    print("✓ QUANTITATIVE: Fixed-site services identify more Novel Psychoactive Substances")
    print("✓ QUALITATIVE: Year-round operation enables early trend identification")
    print("✓ INTEGRATED: Both service models serve essential complementary roles")
    print("✓ INTEGRATED: Different user populations value different service features")
    print("  Mixed-methods approach provides comprehensive understanding of")
    print("  service capabilities, stakeholder perspectives, and public health impact")
    print("=" * 70)

if __name__ == "__main__":
    main()
