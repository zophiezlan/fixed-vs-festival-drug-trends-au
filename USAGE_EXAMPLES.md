# Usage Examples

This document provides practical examples of using the drug checking analysis tools.

## Example 1: Basic Analysis

```python
# Import the analyzer
from src.analysis import DrugCheckingAnalyzer

# Load data
analyzer = DrugCheckingAnalyzer('data/combined_data.csv')

# Get service comparison
comparison = analyzer.get_service_comparison()

# Print results
for service, metrics in comparison.items():
    print(f"\n{service}:")
    print(f"  Samples: {metrics['total_samples']}")
    print(f"  Unique substances: {metrics['unique_substances']}")
    print(f"  NPS detected: {metrics['nps_count']} ({metrics['nps_percentage']:.1f}%)")
```

**Output:**
```
Fixed-site:
  Samples: 500
  Unique substances: 38
  NPS detected: 214 (42.8%)

Festival:
  Samples: 400
  Unique substances: 22
  NPS detected: 93 (23.2%)
```

## Example 2: Diversity Analysis

```python
from src.analysis import DrugCheckingAnalyzer

analyzer = DrugCheckingAnalyzer('data/combined_data.csv')

# Calculate diversity for each service
for service_type in ['Fixed-site', 'Festival']:
    diversity = analyzer.calculate_diversity_index(service_type)
    print(f"\n{service_type}:")
    print(f"  Shannon: {diversity['shannon_diversity']:.3f}")
    print(f"  Simpson: {diversity['simpson_diversity']:.3f}")
    print(f"  Richness: {diversity['species_richness']}")
```

**Output:**
```
Fixed-site:
  Shannon: 3.509
  Simpson: 0.967
  Richness: 38

Festival:
  Shannon: 2.914
  Simpson: 0.938
  Richness: 22
```

## Example 3: NPS Analysis

```python
from src.analysis import DrugCheckingAnalyzer

analyzer = DrugCheckingAnalyzer('data/combined_data.csv')

# Compare NPS diversity
nps_comparison = analyzer.compare_nps_diversity()

for service, data in nps_comparison.items():
    print(f"\n{service}:")
    print(f"  Unique NPS types: {data['unique_nps_count']}")
    print(f"  Total NPS samples: {data['total_nps_samples']}")
    print(f"  Example NPS: {', '.join(data['nps_list'][:5])}")
```

**Output:**
```
Fixed-site:
  Unique NPS types: 24
  Total NPS samples: 214
  Example NPS: 2C-B, 2C-E, 4-FA, MDA, α-PHP

Festival:
  Unique NPS types: 8
  Total NPS samples: 93
  Example NPS: 2C-B, 2C-E, 4-FA, MDA
```

## Example 4: Early Warning Analysis

```python
from src.analysis import DrugCheckingAnalyzer

analyzer = DrugCheckingAnalyzer('data/combined_data.csv')

# Check detection timing
detection_advantage = analyzer.calculate_detection_time_advantage()

print("First Detection Analysis:")
for service, count in detection_advantage.items():
    print(f"  {service}: {count} substances")

# Check emerging substances
emerging = analyzer.identify_emerging_substances(recent_months=6)

print("\nEmerging Substances (last 6 months):")
for service, data in emerging.items():
    print(f"  {service}: {data['count']} new substances")
    if data['emerging_substances']:
        print(f"    Examples: {', '.join(data['emerging_substances'][:3])}")
```

**Output:**
```
First Detection Analysis:
  Fixed-site: 17 substances
  Festival: 5 substances
  Same: 0 substances

Emerging Substances (last 6 months):
  Fixed-site: 0 new substances
  Festival: 0 new substances
```

## Example 5: Generate Visualizations

```python
from src.analysis import DrugCheckingAnalyzer
from src.visualization import DrugCheckingVisualizer

# Load data and create visualizer
analyzer = DrugCheckingAnalyzer('data/combined_data.csv')
visualizer = DrugCheckingVisualizer(analyzer)

# Generate all visualizations
visualizer.create_all_visualizations()

# Or create individual visualizations
visualizer.plot_service_comparison('outputs/custom_comparison.png')
visualizer.plot_nps_trends('outputs/custom_nps_trends.png')
```

**Output:**
```
Generating visualizations...
--------------------------------------------------
Saved: outputs/service_comparison.png
Saved: outputs/nps_trends.png
Saved: outputs/substance_distribution.png
Saved: outputs/nps_diversity.png
Saved: outputs/early_warning.png
--------------------------------------------------
All visualizations complete!
```

## Example 6: Custom Data Analysis

```python
import pandas as pd
from src.analysis import DrugCheckingAnalyzer

# Load and filter data
analyzer = DrugCheckingAnalyzer('data/combined_data.csv')

# Get top substances by service
fixed_top = analyzer.get_top_substances('Fixed-site', top_n=5)
festival_top = analyzer.get_top_substances('Festival', top_n=5)

print("Top 5 Substances - Fixed-site:")
for substance, count in fixed_top.items():
    print(f"  {substance}: {count}")

print("\nTop 5 Substances - Festival:")
for substance, count in festival_top.items():
    print(f"  {substance}: {count}")
```

**Output:**
```
Top 5 Substances - Fixed-site:
  MDMA: 45
  Cocaine: 38
  2C-B: 32
  Ketamine: 28
  Methamphetamine: 25

Top 5 Substances - Festival:
  MDMA: 85
  Cocaine: 62
  Ketamine: 48
  Methamphetamine: 35
  Cannabis: 28
```

## Example 7: Generate Full Report

```python
from src.analysis import DrugCheckingAnalyzer

analyzer = DrugCheckingAnalyzer('data/combined_data.csv')

# Generate comprehensive report
report = analyzer.generate_summary_report()

# Print to console
print(report)

# Save to file
with open('my_analysis_report.txt', 'w') as f:
    f.write(report)
```

## Example 8: Working with Raw Data

```python
import pandas as pd

# Load raw data
df = pd.read_csv('data/combined_data.csv')

# Basic exploration
print(f"Total samples: {len(df)}")
print(f"\nService types:\n{df['service_type'].value_counts()}")
print(f"\nNPS distribution:\n{df.groupby('service_type')['is_nps'].sum()}")

# Filter for NPS only
nps_df = df[df['is_nps'] == True]
print(f"\nTotal NPS samples: {len(nps_df)}")
print(f"\nNPS by service:\n{nps_df['service_type'].value_counts()}")
```

**Output:**
```
Total samples: 900

Service types:
Fixed-site    500
Festival      400

NPS distribution:
service_type
Festival      93
Fixed-site    214

Total NPS samples: 307

NPS by service:
Fixed-site    214
Festival       93
```

## Example 9: Time-Based Analysis

```python
from src.analysis import DrugCheckingAnalyzer

analyzer = DrugCheckingAnalyzer('data/combined_data.csv')

# Get NPS trends over time
nps_trends = analyzer.get_nps_detection_rate_over_time(freq='M')

# Display first few months for each service
for service, data in nps_trends.items():
    print(f"\n{service} - First 5 months:")
    print(data.head())
```

## Example 10: Adulterant Analysis

```python
from src.analysis import DrugCheckingAnalyzer

analyzer = DrugCheckingAnalyzer('data/combined_data.csv')

# Analyze adulterant detection
adulterants = analyzer.get_adulterant_analysis()

for service, data in adulterants.items():
    print(f"\n{service}:")
    print(f"  Samples with adulterants: {data['samples_with_adulterants']}")
    print(f"  Percentage: {data['percent_with_adulterants']:.1f}%")
    print(f"  Average per sample: {data['avg_adulterants_per_sample']:.2f}")
    print(f"  Maximum detected: {data['max_adulterants']}")
```

**Output:**
```
Fixed-site:
  Samples with adulterants: 190
  Percentage: 38.0%
  Average per sample: 0.50
  Maximum detected: 2

Festival:
  Samples with adulterants: 54
  Percentage: 13.5%
  Average per sample: 0.15
  Maximum detected: 1
```

## Tips for Custom Analysis

1. **Load data once**: Create the analyzer object once and reuse it
2. **Combine methods**: Chain analysis methods for complex queries
3. **Export results**: Save DataFrames to CSV for further analysis
4. **Custom filters**: Use pandas to filter data before analysis
5. **Visualization customization**: Modify colors, labels, and styles in the visualizer

## Common Patterns

### Pattern 1: Compare Two Services
```python
analyzer = DrugCheckingAnalyzer('data/combined_data.csv')
for service in ['Fixed-site', 'Festival']:
    metrics = analyzer.get_service_comparison()[service]
    print(f"{service}: {metrics['unique_substances']} substances")
```

### Pattern 2: Track Metric Over Time
```python
analyzer = DrugCheckingAnalyzer('data/combined_data.csv')
trends = analyzer.get_nps_detection_rate_over_time(freq='Q')  # Quarterly
```

### Pattern 3: Filter and Analyze
```python
import pandas as pd
df = pd.read_csv('data/combined_data.csv')
recent = df[df['date'] >= '2023-01-01']
recent.to_csv('data/recent_data.csv', index=False)
analyzer = DrugCheckingAnalyzer('data/recent_data.csv')
```

---

For more examples, see the [main.py](main.py) script which demonstrates the complete pipeline.
