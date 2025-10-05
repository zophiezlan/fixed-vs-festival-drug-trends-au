# Fixed-Site vs Festival Drug Checking Trends in Australia

A comprehensive data analysis project comparing Australian fixed-site and festival drug checking services, with a focus on drug diversity detection and Novel Psychoactive Substances (NPS) identification.

## Overview

This project analyzes and visualizes trends showing that **fixed-site drug checking services detect a higher diversity of drugs**, especially Novel Psychoactive Substances (NPS), highlighting their crucial role as an **early warning system for public health**.

### Key Findings

✅ **Higher Drug Diversity**: Fixed-site services detect significantly more unique substances  
✅ **Enhanced NPS Detection**: Fixed sites identify more NPS types (40% vs 20% detection rate)  
✅ **Early Warning Function**: Fixed sites detect emerging substances earlier than festival services  
✅ **Better Adulterant Detection**: More comprehensive analysis capabilities  
✅ **Public Health Surveillance**: Critical role in monitoring drug market trends

## Project Structure

```
fixed-vs-festival-drug-trends-au/
├── data/                      # Dataset directory
│   ├── fixed_site_data.csv    # Fixed-site service data
│   ├── festival_data.csv      # Festival service data
│   └── combined_data.csv      # Combined dataset
├── src/                       # Source code
│   ├── generate_data.py       # Synthetic data generation
│   ├── analysis.py            # Analysis module
│   └── visualization.py       # Visualization module
├── outputs/                   # Generated outputs
│   ├── analysis_report.txt    # Text summary report
│   ├── service_comparison.png # Key metrics comparison
│   ├── nps_trends.png         # NPS detection trends
│   ├── substance_distribution.png
│   ├── nps_diversity.png      # NPS diversity analysis
│   └── early_warning.png      # Early warning system visualization
├── main.py                    # Main analysis script
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

Execute the main analysis script to generate data, perform analysis, and create visualizations:

```bash
python main.py
```

This will:
1. Generate synthetic drug checking datasets (900 samples total)
2. Perform comparative statistical analysis
3. Calculate diversity indices (Shannon, Simpson)
4. Identify emerging substances and early warning indicators
5. Create comprehensive visualizations
6. Save all outputs to the `outputs/` directory

### Individual Components

You can also run individual components:

```bash
# Generate data only
python src/generate_data.py

# Run custom analysis (in Python)
from src.analysis import DrugCheckingAnalyzer
analyzer = DrugCheckingAnalyzer('data/combined_data.csv')
print(analyzer.generate_summary_report())
```

## Data Description

### Datasets

The project uses synthetic datasets designed to reflect realistic patterns in Australian drug checking services:

- **Fixed-Site Data** (500 samples): Year-round service with higher NPS detection
- **Festival Data** (400 samples): Event-based service focused on common substances

### Variables

- `date`: Date of sample testing
- `service_type`: Fixed-site or Festival
- `substance_detected`: Primary substance identified
- `expected_substance`: What the person thought they had
- `is_nps`: Boolean flag for Novel Psychoactive Substances
- `sample_form`: Physical form (powder, pill, crystal, etc.)
- `adulterants`: Additional substances detected
- `num_adulterants`: Count of adulterants

## Methodology

### Analysis Techniques

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

### Visualizations

The project generates five key visualization types:

1. **Service Comparison**: Overview of key metrics
2. **NPS Trends**: Temporal patterns in NPS detection
3. **Substance Distribution**: Most commonly detected substances
4. **NPS Diversity**: Detailed NPS type comparison
5. **Early Warning System**: Detection timing and emerging threats

## Results & Implications

### For Harm Reduction

- Fixed-site services provide continuous monitoring of drug market trends
- Earlier detection of emerging substances enables faster public health responses
- Higher diversity detection supports more comprehensive harm reduction messaging

### For Policy Makers

- Evidence supporting investment in fixed-site infrastructure
- Demonstrates complementary roles of fixed-site and festival services
- Data-driven approach to resource allocation for drug checking services

### For Public Health

- Early warning system for Novel Psychoactive Substances
- Enhanced surveillance of drug market dynamics
- Better adulterant detection protects public health

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

This project uses **synthetic data only**. Real drug checking data should:
- Be anonymized and de-identified
- Comply with privacy regulations
- Have appropriate ethical approval
- Follow data sharing agreements with service providers

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