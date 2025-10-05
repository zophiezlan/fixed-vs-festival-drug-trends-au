# Quick Start Guide

Get started with the Australian Drug Checking Analysis project in 5 minutes!

## Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/zophiezlan/fixed-vs-festival-drug-trends-au.git
cd fixed-vs-festival-drug-trends-au

# Install dependencies
pip install -r requirements.txt
```

## Run Analysis (30 seconds)

```bash
python main.py
```

That's it! The script will:
1. Generate synthetic datasets (900 samples)
2. Perform comprehensive analysis
3. Create 5 visualizations
4. Save results to `outputs/` directory

## View Results

After running, check these files:

- **📊 Visualizations**: `outputs/*.png` (5 charts)
- **📄 Report**: `outputs/analysis_report.txt`
- **📁 Data**: `data/*.csv` (3 datasets)

## Key Outputs

### 1. Service Comparison
![Service Comparison](outputs/service_comparison.png)
*Compare samples, substances, NPS rates, and diversity*

### 2. NPS Trends
![NPS Trends](outputs/nps_trends.png)
*Track NPS detection over time*

### 3. Substance Distribution
![Substance Distribution](outputs/substance_distribution.png)
*Top substances by service type*

### 4. NPS Diversity
![NPS Diversity](outputs/nps_diversity.png)
*Detailed NPS type comparison*

### 5. Early Warning System
![Early Warning](outputs/early_warning.png)
*Detection timing and emerging threats*

## Key Findings

✅ Fixed-site services detect **38 unique substances** vs 22 for festivals  
✅ Fixed-site **NPS detection rate: 42.8%** vs 23.2% for festivals  
✅ Fixed-site services identify **24 NPS types** vs 8 for festivals  
✅ Fixed-site detected **17 substances first** vs 5 for festivals  

## Common Use Cases

### Generate New Data
```bash
python src/generate_data.py
```

### Custom Analysis
```python
from src.analysis import DrugCheckingAnalyzer

analyzer = DrugCheckingAnalyzer('data/combined_data.csv')
report = analyzer.generate_summary_report()
print(report)
```

### Test Everything
```bash
python test_pipeline.py
```

## Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**No visualizations?**
- Check `outputs/` directory exists
- Ensure matplotlib is installed
- Try running `python main.py` again

**Data files missing?**
- They're generated automatically by `main.py`
- Or run: `python src/generate_data.py`

## Next Steps

1. 📖 Read [README.md](README.md) for full documentation
2. 📊 Check [EXAMPLES.md](EXAMPLES.md) for detailed outputs
3. 🤝 See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Python API

```python
# Import modules
from src.analysis import DrugCheckingAnalyzer
from src.visualization import DrugCheckingVisualizer

# Load data
analyzer = DrugCheckingAnalyzer('data/combined_data.csv')

# Get metrics
comparison = analyzer.get_service_comparison()
diversity = analyzer.calculate_diversity_index('Fixed-site')
nps_data = analyzer.compare_nps_diversity()

# Create visualizations
visualizer = DrugCheckingVisualizer(analyzer)
visualizer.create_all_visualizations()
```

## File Structure

```
fixed-vs-festival-drug-trends-au/
├── main.py                    # ⚡ Run this!
├── test_pipeline.py           # Test suite
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── EXAMPLES.md               # Output examples
├── CONTRIBUTING.md           # Contribution guide
├── data/                     # Generated datasets
│   ├── fixed_site_data.csv
│   ├── festival_data.csv
│   └── combined_data.csv
├── outputs/                  # Generated outputs
│   ├── *.png                # Visualizations
│   └── analysis_report.txt  # Summary report
└── src/                     # Source code
    ├── generate_data.py     # Data generation
    ├── analysis.py          # Analysis module
    └── visualization.py     # Visualization module
```

## Requirements

- Python 3.8+
- pandas
- numpy  
- matplotlib
- seaborn
- scipy

## Time Requirements

- **Installation**: 2 minutes
- **Running analysis**: 30 seconds
- **Understanding outputs**: 5 minutes
- **Total**: ~8 minutes to full insights!

## Support

- 🐛 **Bug reports**: Open a GitHub issue
- 💡 **Feature ideas**: Open a GitHub issue
- 📚 **Documentation**: See README.md
- 🤝 **Contribute**: See CONTRIBUTING.md

---

**Ready to analyze drug checking trends?** Run `python main.py` now! 🚀
