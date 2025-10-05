"""
Generate synthetic qualitative interview data for stakeholder analysis.
This creates realistic interview transcripts for service providers and users.
"""
import pandas as pd
import random
from datetime import datetime

# Set seed for reproducibility
random.seed(42)

# Interview themes and responses
SERVICE_PROVIDER_THEMES = {
    'detection_capabilities': {
        'Fixed-site': [
            "We have more time for detailed analysis, so we can detect a wider range of substances including novel psychoactive substances that might be missed in a festival setting.",
            "The laboratory setting allows us to use more sophisticated testing equipment, which is crucial for identifying new and emerging substances.",
            "We see a broader range of substances because people come to us year-round with whatever they're concerned about, not just festival drugs."
        ],
        'Festival': [
            "At festivals, we focus on quick turnaround times for common substances like MDMA and cocaine that people are most likely to use at these events.",
            "The festival environment is more limited - we use portable equipment that's reliable but not as comprehensive as fixed-site labs.",
            "We see mainly recreational substances that are popular in festival settings."
        ]
    },
    'early_warning': {
        'Fixed-site': [
            "We often detect new substances months before they appear at festivals because we serve a diverse client base throughout the year.",
            "Our continuous operation means we can track emerging trends in real-time and alert public health authorities early.",
            "We've been first to identify several novel substances that later became more widespread."
        ],
        'Festival': [
            "By the time substances appear at festivals, they're often already circulating in the community.",
            "We do catch things, but our seasonal operation means we might miss early trends between events.",
            "We're great for snapshot monitoring during peak use times."
        ]
    },
    'user_populations': {
        'Fixed-site': [
            "We see a more diverse user population - not just recreational users but also people with substance use concerns and those experimenting with new substances.",
            "People bring us things they've purchased online or through unconventional channels.",
            "We get samples from people who are specifically concerned about adulterants or want to know exactly what they have."
        ],
        'Festival': [
            "Festival users are typically recreational users who want to make safer choices for immediate use.",
            "The population is younger on average and more focused on party drugs.",
            "People usually test substances they plan to use that day or weekend."
        ]
    },
    'harm_reduction_impact': {
        'Fixed-site': [
            "We can provide in-depth education and counseling because we're not under time pressure.",
            "Our year-round presence allows us to build relationships with regular clients and track patterns.",
            "We can refer people to other services if needed - it's a more holistic approach."
        ],
        'Festival': [
            "We provide immediate harm reduction when people need it most - right before they might use.",
            "The festival setting allows us to reach people who might not seek out a fixed service.",
            "We create a harm reduction culture at events which has broader community impact."
        ]
    },
    'resource_needs': {
        'Fixed-site': [
            "We need sustained funding for equipment, staff, and facility costs throughout the year.",
            "Investing in fixed sites provides continuous surveillance capability for public health.",
            "The cost per sample is higher but the intelligence value is greater."
        ],
        'Festival': [
            "We need support for portable equipment and trained volunteer teams.",
            "Festival services are cost-effective for reaching many people in a short time.",
            "We need better coordination between festival and fixed services for comprehensive coverage."
        ]
    }
}

SERVICE_USER_THEMES = {
    'access_preferences': {
        'Fixed-site': [
            "I prefer the fixed site because I can go anytime and don't have to wait for an event.",
            "The fixed location feels more private and professional.",
            "I like that I can take my time and ask questions without the festival chaos."
        ],
        'Festival': [
            "It's convenient to test right before the event when I need it.",
            "The festival service is easier to access - I'm already there.",
            "I appreciate the immediate results so I can make decisions quickly."
        ]
    },
    'trust_and_comfort': {
        'Fixed-site': [
            "The clinical setting makes me trust the results more - it seems more thorough.",
            "Staff seem more knowledgeable and have time to explain things properly.",
            "I feel I can be more honest about what I'm taking and why."
        ],
        'Festival': [
            "The festival vibe makes it less intimidating - feels more peer-based.",
            "I trust the service because they're embedded in the harm reduction community.",
            "It's easier to approach without feeling judged at a festival."
        ]
    },
    'information_needs': {
        'Fixed-site': [
            "I wanted detailed information about an unfamiliar substance I got online.",
            "I was concerned about potential adulterants and wanted comprehensive testing.",
            "I needed to understand the risks of something I'd been using regularly."
        ],
        'Festival': [
            "I just wanted to know if my MDMA was actually MDMA before the set.",
            "Quick confirmation of purity was all I needed.",
            "I wanted to know if what I had matched what my friends had."
        ]
    },
    'behavior_change': {
        'Fixed-site': [
            "Learning about adulterants made me more careful about sources.",
            "The detailed results changed my whole approach to substance use.",
            "I now test everything before using, even from trusted sources."
        ],
        'Festival': [
            "I adjusted my dose based on the purity information.",
            "Finding out it wasn't what I expected made me dispose of it.",
            "It confirmed what I hoped but made me more cautious generally."
        ]
    },
    'service_improvements': {
        'Fixed-site': [
            "More locations would make it more accessible across the city.",
            "Extended hours would help people with work schedules.",
            "Online booking would make it easier to plan visits."
        ],
        'Festival': [
            "More staff would reduce wait times during peak hours.",
            "Better signage would help people find the service.",
            "Having services at more festivals would reach more people."
        ]
    }
}

def generate_service_provider_interviews(n_fixed=5, n_festival=4):
    """Generate synthetic service provider interview data."""
    interviews = []
    interview_id = 1
    
    # Generate fixed-site provider interviews
    for i in range(n_fixed):
        provider_id = f"SP_FIXED_{i+1:02d}"
        interview = {
            'interview_id': interview_id,
            'participant_id': provider_id,
            'participant_type': 'Service Provider',
            'service_type': 'Fixed-site',
            'role': random.choice(['Lab Technician', 'Harm Reduction Worker', 'Program Coordinator', 'Pharmacist']),
            'years_experience': random.randint(2, 15),
            'interview_date': datetime(2024, random.randint(1, 3), random.randint(1, 28)).strftime('%Y-%m-%d'),
            'interview_duration_minutes': random.randint(45, 90)
        }
        
        # Add responses for each theme
        for theme, responses in SERVICE_PROVIDER_THEMES.items():
            interview[f'theme_{theme}'] = random.choice(responses['Fixed-site'])
        
        interviews.append(interview)
        interview_id += 1
    
    # Generate festival provider interviews
    for i in range(n_festival):
        provider_id = f"SP_FEST_{i+1:02d}"
        interview = {
            'interview_id': interview_id,
            'participant_id': provider_id,
            'participant_type': 'Service Provider',
            'service_type': 'Festival',
            'role': random.choice(['Volunteer Coordinator', 'Testing Technician', 'Harm Reduction Peer', 'Event Coordinator']),
            'years_experience': random.randint(1, 10),
            'interview_date': datetime(2024, random.randint(1, 3), random.randint(1, 28)).strftime('%Y-%m-%d'),
            'interview_duration_minutes': random.randint(30, 75)
        }
        
        # Add responses for each theme
        for theme, responses in SERVICE_PROVIDER_THEMES.items():
            interview[f'theme_{theme}'] = random.choice(responses['Festival'])
        
        interviews.append(interview)
        interview_id += 1
    
    return pd.DataFrame(interviews)

def generate_service_user_interviews(n_fixed=8, n_festival=7):
    """Generate synthetic service user interview data."""
    interviews = []
    interview_id = 100
    
    # Generate fixed-site user interviews
    for i in range(n_fixed):
        user_id = f"SU_FIXED_{i+1:02d}"
        interview = {
            'interview_id': interview_id,
            'participant_id': user_id,
            'participant_type': 'Service User',
            'service_type': 'Fixed-site',
            'age_group': random.choice(['18-25', '26-35', '36-45', '46+']),
            'times_used_service': random.randint(1, 12),
            'interview_date': datetime(2024, random.randint(1, 3), random.randint(1, 28)).strftime('%Y-%m-%d'),
            'interview_duration_minutes': random.randint(20, 45)
        }
        
        # Add responses for each theme
        for theme, responses in SERVICE_USER_THEMES.items():
            interview[f'theme_{theme}'] = random.choice(responses['Fixed-site'])
        
        interviews.append(interview)
        interview_id += 1
    
    # Generate festival user interviews
    for i in range(n_festival):
        user_id = f"SU_FEST_{i+1:02d}"
        interview = {
            'interview_id': interview_id,
            'participant_id': user_id,
            'participant_type': 'Service User',
            'service_type': 'Festival',
            'age_group': random.choice(['18-25', '26-35', '36-45']),
            'times_used_service': random.randint(1, 5),
            'interview_date': datetime(2024, random.randint(1, 3), random.randint(1, 28)).strftime('%Y-%m-%d'),
            'interview_duration_minutes': random.randint(15, 35)
        }
        
        # Add responses for each theme
        for theme, responses in SERVICE_USER_THEMES.items():
            interview[f'theme_{theme}'] = random.choice(responses['Festival'])
        
        interviews.append(interview)
        interview_id += 1
    
    return pd.DataFrame(interviews)

def main():
    """Generate and save qualitative interview datasets."""
    print("Generating synthetic qualitative interview data...")
    
    # Generate datasets
    providers_df = generate_service_provider_interviews(n_fixed=5, n_festival=4)
    users_df = generate_service_user_interviews(n_fixed=8, n_festival=7)
    
    # Combine datasets
    combined_df = pd.concat([providers_df, users_df], ignore_index=True)
    
    # Save datasets
    providers_df.to_csv('data/service_provider_interviews.csv', index=False)
    users_df.to_csv('data/service_user_interviews.csv', index=False)
    combined_df.to_csv('data/all_interviews.csv', index=False)
    
    print(f"\nQualitative datasets created:")
    print(f"  - Service provider interviews: {len(providers_df)} (Fixed-site: 5, Festival: 4)")
    print(f"  - Service user interviews: {len(users_df)} (Fixed-site: 8, Festival: 7)")
    print(f"  - Total interviews: {len(combined_df)}")
    print(f"\nFiles saved to data/ directory")
    
    # Print summary statistics
    print("\n=== Service Provider Summary ===")
    print(f"Average years of experience:")
    print(f"  Fixed-site: {providers_df[providers_df['service_type']=='Fixed-site']['years_experience'].mean():.1f} years")
    print(f"  Festival: {providers_df[providers_df['service_type']=='Festival']['years_experience'].mean():.1f} years")
    
    print("\n=== Service User Summary ===")
    print(f"Average times used service:")
    print(f"  Fixed-site: {users_df[users_df['service_type']=='Fixed-site']['times_used_service'].mean():.1f} times")
    print(f"  Festival: {users_df[users_df['service_type']=='Festival']['times_used_service'].mean():.1f} times")

if __name__ == "__main__":
    main()
