# Fixed-Site vs Festival Drug Checking Trends in Australia

A comprehensive **mixed-methods** data analysis project comparing Australian fixed-site and festival drug checking services, combining quantitative analysis of drug checking data with qualitative interviews of stakeholders to identify distinct trends and insights.

## Overview

This project employs a **convergent parallel mixed-methods approach** that integrates:
- **Quantitative analysis**: Statistical analysis of drug checking data (900 samples)
- **Qualitative analysis**: Thematic analysis of stakeholder interviews (24 participants)
- **Mixed-methods synthesis**: Integration of findings to provide comprehensive understanding

The analysis demonstrates that **fixed-site drug checking services detect a higher diversity of drugs**, especially Novel Psychoactive Substances (NPS), while stakeholder interviews reveal the mechanisms and contextual factors behind these patterns.

### Key Findings

**Quantitative Analysis:**
✅ **Higher Drug Diversity**: Fixed-site services detect significantly more unique substances (38 vs 22)
✅ **Enhanced NPS Detection**: Fixed sites identify more NPS types (42.8% vs 23.2% detection rate)  
✅ **Early Warning Function**: Fixed sites detect emerging substances 3.4x more often  
✅ **Better Adulterant Detection**: 38% vs 13.5% detection rate at fixed sites

**Qualitative Analysis:**
✅ **Provider Perspectives**: Different capabilities attributed to equipment, time, and clientele
✅ **User Preferences**: Distinct needs based on context, timing, and information requirements
✅ **Service Complementarity**: Both models serve essential but different functions

**Mixed-Methods Synthesis:**
✅ **Strong Convergence**: Quantitative patterns explained by qualitative mechanisms
✅ **Comprehensive Understanding**: Integration reveals why differences exist and their implications
✅ **Policy Implications**: Evidence supports investment in both complementary service models

## Project Structure

```
fixed-vs-festival-drug-trends-au/
├── data/                      # Dataset directory
│   ├── fixed_site_data.csv    # Fixed-site service data
│   ├── festival_data.csv      # Festival service data
│   ├── combined_data.csv      # Combined quantitative dataset
│   ├── service_provider_interviews.csv  # Provider interview data
│   ├── service_user_interviews.csv      # User interview data
│   └── all_interviews.csv     # Combined qualitative dataset
├── src/                       # Source code
│   ├── generate_data.py       # Quantitative data generation
│   ├── generate_qualitative_data.py  # Qualitative data generation
│   ├── analysis.py            # Quantitative analysis module
│   ├── qualitative_analysis.py  # Qualitative analysis module
│   ├── mixed_methods.py       # Mixed-methods integration
│   └── visualization.py       # Visualization module
├── outputs/                   # Generated outputs
│   ├── quantitative_analysis.txt  # Quantitative report
│   ├── qualitative_analysis.txt   # Qualitative report
│   ├── mixed_methods_report.txt   # Integrated findings
│   ├── service_comparison.png # Key metrics comparison
│   ├── nps_trends.png         # NPS detection trends
│   ├── substance_distribution.png
│   ├── nps_diversity.png      # NPS diversity analysis
│   └── early_warning.png      # Early warning system visualization
├── main.py                    # Main mixed-methods analysis script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/zophiezlan/fixed-vs-festival-drug-trends-au.git
cd fixed-vs-festival-drug-trends-au
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Analysis

Execute the main analysis script to generate data, perform mixed-methods analysis, and create visualizations:

```bash
python main.py
```

This will:
1. Generate synthetic quantitative drug checking datasets (900 samples)
2. Generate synthetic qualitative interview data (24 stakeholder interviews)
3. Perform quantitative statistical analysis
4. Conduct qualitative thematic analysis
5. Integrate findings using mixed-methods synthesis
6. Calculate diversity indices (Shannon, Simpson)
7. Identify emerging substances and early warning indicators
8. Create comprehensive visualizations
9. Save all outputs to the `outputs/` directory

### Individual Components

You can also run individual components:

```bash
# Generate data only
python src/generate_data.py
python src/generate_qualitative_data.py

# Run custom analysis (in Python)
from src.analysis import DrugCheckingAnalyzer
from src.qualitative_analysis import QualitativeAnalyzer
from src.mixed_methods import MixedMethodsIntegrator

# Quantitative analysis
quant_analyzer = DrugCheckingAnalyzer('data/combined_data.csv')
print(quant_analyzer.generate_summary_report())

# Qualitative analysis
qual_analyzer = QualitativeAnalyzer('data/all_interviews.csv')
print(qual_analyzer.generate_qualitative_summary())

# Mixed-methods integration
integrator = MixedMethodsIntegrator(quant_analyzer, qual_analyzer)
print(integrator.generate_mixed_methods_report())
```

## Data Description

### Quantitative Datasets

The project uses synthetic drug checking datasets designed to reflect realistic patterns:

- **Fixed-Site Data** (500 samples): Year-round service with higher NPS detection
- **Festival Data** (400 samples): Event-based service focused on common substances

**Variables:**
- `date`: Date of sample testing
- `service_type`: Fixed-site or Festival
- `substance_detected`: Primary substance identified
- `expected_substance`: What the person thought they had
- `is_nps`: Boolean flag for Novel Psychoactive Substances
- `sample_form`: Physical form (powder, pill, crystal, etc.)
- `adulterants`: Additional substances detected
- `num_adulterants`: Count of adulterants

### Qualitative Datasets

Synthetic stakeholder interview data representing diverse perspectives:

- **Service Provider Interviews** (9 interviews): 5 fixed-site, 4 festival providers
- **Service User Interviews** (15 interviews): 8 fixed-site, 7 festival users

**Interview Themes:**
- Detection capabilities and limitations
- Early warning system functions
- User populations and access patterns
- Harm reduction impact
- Resource needs and service improvements
- Trust, comfort, and information needs

## Methodology

### Mixed-Methods Design

This project employs a **convergent parallel mixed-methods design** where quantitative and qualitative data are collected and analyzed independently, then integrated to provide comprehensive insights:

#### Quantitative Component

1. **Diversity Analysis**
   - Shannon Diversity Index: Measures species richness and evenness
   - Simpson Diversity Index: Probability of sampling different species
   - Species Richness: Total number of unique substances

2. **Temporal Analysis**
   - NPS detection rates over time
   - Cumulative discovery curves
   - Emerging substance identification

3. **Comparative Metrics**
   - Detection rate comparisons
   - First detection timing (early warning)
   - Adulterant detection capabilities

#### Qualitative Component

1. **Stakeholder Interviews**
   - Service providers (fixed-site and festival)
   - Service users (both service types)
   - Semi-structured interview approach

2. **Thematic Analysis**
   - Identification of recurring themes
   - Pattern recognition across stakeholder groups
   - Contextual understanding of service differences

#### Mixed-Methods Integration

1. **Convergence Analysis**
   - Identify where quantitative and qualitative findings align
   - Assess strength of triangulation

2. **Complementarity Analysis**
   - Explore how different methods provide unique insights
   - Understand mechanisms behind quantitative patterns

3. **Synthesis**
   - Integrated interpretation of findings
   - Policy and practice implications

### Visualizations

The project generates five key visualization types:

1. **Service Comparison**: Overview of key metrics
2. **NPS Trends**: Temporal patterns in NPS detection
3. **Substance Distribution**: Most commonly detected substances
4. **NPS Diversity**: Detailed NPS type comparison
5. **Early Warning System**: Detection timing and emerging threats

## Results & Implications

### Integrated Findings

The mixed-methods approach reveals:

1. **Detection Capabilities**: Quantitative data shows 73% more substance diversity at fixed sites. Qualitative interviews explain this through equipment sophistication, analysis time, and diverse clientele.

2. **Early Warning Function**: Fixed sites detect new substances first 3.4x more often. Providers confirm this stems from year-round operation and broader sample sources.

3. **User Populations**: Different service models attract distinct populations with varying needs—fixed-site users value thoroughness and privacy; festival users prioritize convenience and immediacy.

4. **Service Complementarity**: Both quantitative performance data and stakeholder perspectives emphasize that both service types are essential and complementary rather than competing.

### For Harm Reduction

- Fixed-site services provide continuous monitoring of drug market trends
- Festival services offer point-of-need accessible harm reduction
- Different user populations benefit from different service models
- Both approaches needed for comprehensive harm reduction coverage
- Qualitative data reveals user satisfaction and behavior change patterns

### For Policy Makers

- Evidence supporting investment in both fixed-site and festival services
- Quantitative data demonstrates detection capabilities and surveillance value
- Qualitative data reveals stakeholder perspectives on service value
- Mixed-methods approach provides robust evidence for resource allocation
- Both service models serve complementary rather than competing roles

### For Public Health

- Early warning system for Novel Psychoactive Substances validated through quantitative data
- Enhanced surveillance of drug market dynamics
- Stakeholder insights reveal mechanisms of service effectiveness
- Better adulterant detection at fixed sites protects public health
- User perspectives inform service design and accessibility improvements

## Technical Details

### Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scipy**: Scientific computing (diversity indices)

### Performance

- Analysis runtime: ~2-5 seconds on modern hardware
- Visualization generation: ~5-10 seconds
- Memory usage: <100MB

## Contributing

Contributions are welcome! Areas for enhancement:

- Integration with real-world drug checking datasets
- Additional statistical tests (Mann-Whitney U, Kruskal-Wallis)
- Interactive visualizations (Plotly, Dash)
- Longitudinal trend forecasting
- Geographic analysis (multi-site comparison)

## Ethics & Data Privacy

This project uses **synthetic data only** to demonstrate methodology. Real drug checking and interview data should:
- Be anonymized and de-identified
- Have informed consent from all participants
- Comply with privacy regulations (e.g., GDPR, local privacy laws)
- Have appropriate ethical approval from research ethics committees
- Follow data sharing agreements with service providers
- Protect participant confidentiality in qualitative data
- Use pseudonyms and remove identifying information in interview transcripts

## License

This project is provided for educational and research purposes. Please ensure appropriate attribution when using this code or methodology.

## Acknowledgments

This analysis framework supports the important work of drug checking services in Australia, including:
- Pill Testing Australia
- CanTEST Health and Drug Checking Service (ACT)
- Various festival-based harm reduction services

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

## Citation

If you use this methodology in your research, please cite:

```
Fixed-Site vs Festival Drug Checking Trends in Australia
https://github.com/zophiezlan/fixed-vs-festival-drug-trends-au
```

---

**Note**: This is a demonstration project using synthetic data. For real-world applications, partner with established drug checking services and follow appropriate research ethics protocols.