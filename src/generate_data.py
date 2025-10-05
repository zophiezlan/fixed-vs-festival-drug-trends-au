"""
Generate synthetic drug checking datasets for analysis.
This creates realistic sample data for fixed-site and festival services.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define drug categories
COMMON_DRUGS = [
    'MDMA', 'Cocaine', 'Methamphetamine', 'Ketamine', 
    'Cannabis', 'LSD', 'Psilocybin', 'Amphetamine'
]

NPS_DRUGS = [
    '2C-B', '2C-E', '2C-I', '4-FA', '4-FMA', '5-MeO-DALT',
    'DOC', 'DOI', 'N-Ethylpentylone', 'Eutylone', 
    '25I-NBOMe', '25B-NBOMe', 'MDA', '3-MMC', '4-MMC',
    'α-PHP', 'α-PVP', 'Synthetic cannabinoids', 
    'Benzofurans', 'Phenethylamines', 'Tryptamines',
    'Cathinones', 'DMT analogues', '5-MeO-DMT'
]

OTHER_SUBSTANCES = [
    'Caffeine', 'Paracetamol', 'Glucose', 'Lactose',
    'Unknown', 'No substance detected'
]

def generate_fixed_site_data(n_samples=500):
    """Generate fixed-site drug checking data with higher NPS diversity."""
    data = []
    start_date = datetime(2022, 1, 1)
    
    for i in range(n_samples):
        # Fixed sites have more time for detailed analysis
        days_offset = random.randint(0, 730)  # 2 years
        date = start_date + timedelta(days=days_offset)
        
        # Higher probability of detecting NPS at fixed sites (40%)
        rand = random.random()
        if rand < 0.35:
            substance = random.choice(COMMON_DRUGS)
            is_nps = False
        elif rand < 0.75:
            substance = random.choice(NPS_DRUGS)
            is_nps = True
        else:
            substance = random.choice(OTHER_SUBSTANCES)
            is_nps = False
        
        # Sample characteristics
        sample_form = random.choice(['Powder', 'Crystal', 'Pill', 'Liquid', 'Blotter', 'Capsule'])
        expected_substance = random.choice(COMMON_DRUGS + NPS_DRUGS[:5])
        
        # Fixed sites can detect more adulterants
        adulterants = []
        if random.random() < 0.3:
            adulterants.append(random.choice(OTHER_SUBSTANCES[:4]))
        if random.random() < 0.15:
            adulterants.append(random.choice(COMMON_DRUGS[:3]))
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'service_type': 'Fixed-site',
            'substance_detected': substance,
            'expected_substance': expected_substance,
            'is_nps': is_nps,
            'sample_form': sample_form,
            'adulterants': ', '.join(adulterants) if adulterants else 'None',
            'num_adulterants': len(adulterants)
        })
    
    return pd.DataFrame(data)

def generate_festival_data(n_samples=400):
    """Generate festival drug checking data with lower NPS diversity."""
    data = []
    start_date = datetime(2022, 1, 1)
    
    # Festivals happen at specific times
    festival_dates = [
        start_date + timedelta(days=random.randint(0, 730))
        for _ in range(20)
    ]
    
    for i in range(n_samples):
        # Pick a random festival date
        base_date = random.choice(festival_dates)
        date = base_date + timedelta(days=random.randint(0, 2))
        
        # Lower probability of detecting NPS at festivals (20%)
        rand = random.random()
        if rand < 0.65:
            substance = random.choice(COMMON_DRUGS)
            is_nps = False
        elif rand < 0.85:
            # Festival NPS detection focuses on more common ones
            substance = random.choice(NPS_DRUGS[:8])
            is_nps = True
        else:
            substance = random.choice(OTHER_SUBSTANCES)
            is_nps = False
        
        # Sample characteristics
        sample_form = random.choice(['Powder', 'Crystal', 'Pill', 'Capsule'])
        expected_substance = random.choice(COMMON_DRUGS + NPS_DRUGS[:3])
        
        # Festivals detect fewer adulterants due to time constraints
        adulterants = []
        if random.random() < 0.15:
            adulterants.append(random.choice(OTHER_SUBSTANCES[:4]))
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'service_type': 'Festival',
            'substance_detected': substance,
            'expected_substance': expected_substance,
            'is_nps': is_nps,
            'sample_form': sample_form,
            'adulterants': ', '.join(adulterants) if adulterants else 'None',
            'num_adulterants': len(adulterants)
        })
    
    return pd.DataFrame(data)

def main():
    """Generate and save datasets."""
    print("Generating synthetic drug checking datasets...")
    
    # Generate datasets
    fixed_site_df = generate_fixed_site_data(500)
    festival_df = generate_festival_data(400)
    
    # Combine datasets
    combined_df = pd.concat([fixed_site_df, festival_df], ignore_index=True)
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    # Save datasets
    fixed_site_df.to_csv('data/fixed_site_data.csv', index=False)
    festival_df.to_csv('data/festival_data.csv', index=False)
    combined_df.to_csv('data/combined_data.csv', index=False)
    
    print(f"\nDatasets created:")
    print(f"  - Fixed-site samples: {len(fixed_site_df)}")
    print(f"  - Festival samples: {len(festival_df)}")
    print(f"  - Combined samples: {len(combined_df)}")
    print(f"\nFiles saved to data/ directory")
    
    # Print summary statistics
    print("\n=== Fixed-Site Summary ===")
    print(f"Unique substances: {fixed_site_df['substance_detected'].nunique()}")
    print(f"NPS detected: {fixed_site_df['is_nps'].sum()} ({fixed_site_df['is_nps'].sum()/len(fixed_site_df)*100:.1f}%)")
    
    print("\n=== Festival Summary ===")
    print(f"Unique substances: {festival_df['substance_detected'].nunique()}")
    print(f"NPS detected: {festival_df['is_nps'].sum()} ({festival_df['is_nps'].sum()/len(festival_df)*100:.1f}%)")

if __name__ == "__main__":
    main()
