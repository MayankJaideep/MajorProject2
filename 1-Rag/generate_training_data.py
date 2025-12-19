"""
Generate synthetic training data for case outcome prediction.
This creates realistic Indian Commercial Court cases with outcomes.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define realistic data distributions
COURTS = [
    "Supreme Court of India",
    "Delhi High Court",
    "Bombay High Court",
    "Madras High Court",
    "Calcutta High Court",
    "Karnataka High Court",
    "Gujarat High Court",
    "Allahabad High Court",
    "Rajasthan High Court",
    "Madhya Pradesh High Court"
]

CASE_TYPES = [
    "Breach of Contract",
    "Trademark Infringement",
    "Copyright Violation",
    "Patent Dispute",
    "Partnership Dispute",
    "Arbitration Matter",
    "Insolvency Proceedings",
    "Shareholder Dispute",
    "Intellectual Property",
    "Commercial Fraud",
    "Joint Venture Dispute",
    "Licensing Agreement",
    "Supply Chain Dispute",
    "Merger & Acquisition Dispute"
]

JUDGES = [
    "Justice A. Kumar",
    "Justice S. Sharma",
    "Justice R. Patel",
    "Justice M. Singh",
    "Justice V. Reddy",
    "Justice P. Mehta",
    "Justice N. Gupta",
    "Justice K. Rao",
    "Justice D. Verma",
    "Justice T. Iyer",
    "Justice L. Desai",
    "Justice B. Nair",
    "Justice C. Joshi",
    "Justice H. Kapoor"
]

LEGAL_DOMAINS = [
    "Contract Law",
    "Intellectual Property",
    "Corporate Law",
    "Arbitration",
    "Insolvency",
    "Competition Law",
    "Securities Law",
    "Banking Law"
]

OUTCOMES = [
    "plaintiff_win",
    "defendant_win",
    "settlement",
    "dismissed"
]

def generate_case_complexity():
    """Generate case complexity score (1-10)"""
    return np.random.choice([3, 4, 5, 6, 7, 8], p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])

def calculate_outcome_probability(court, case_type, year, complexity, judge):
    """
    Calculate outcome probability based on features.
    This simulates realistic patterns in legal outcomes.
    """
    base_prob = 0.5
    
    # Supreme Court tends to favor well-argued cases (slightly higher plaintiff win)
    if court == "Supreme Court of India":
        base_prob += 0.05
    elif court in ["Bombay High Court", "Delhi High Court"]:
        base_prob += 0.03
    
    # Certain case types have higher plaintiff success rates
    if case_type in ["Breach of Contract", "Trademark Infringement", "Copyright Violation"]:
        base_prob += 0.1
    elif case_type in ["Commercial Fraud", "Patent Dispute"]:
        base_prob += 0.15
    
    # Recent cases (2020+) have slightly different outcomes due to legal reforms
    if year >= 2020:
        base_prob += 0.05
    
    # Complexity affects outcomes (very complex cases harder to win)
    if complexity >= 8:
        base_prob -= 0.1
    elif complexity <= 4:
        base_prob += 0.05
    
    # Judge experience (simulated by name - some judges more plaintiff-friendly)
    if judge in ["Justice A. Kumar", "Justice S. Sharma", "Justice M. Singh"]:
        base_prob += 0.08
    
    # Add some randomness
    base_prob += np.random.normal(0, 0.1)
    
    # Clamp between 0.2 and 0.8
    return max(0.2, min(0.8, base_prob))

def determine_outcome(plaintiff_prob):
    """Determine case outcome based on probability"""
    rand = random.random()
    
    if rand < plaintiff_prob:
        return "plaintiff_win"
    elif rand < plaintiff_prob + 0.25:
        return "defendant_win"
    elif rand < plaintiff_prob + 0.35:
        return "settlement"
    else:
        return "dismissed"

def generate_case_description(case_type, court, year, outcome):
    """Generate a brief case description"""
    descriptions = {
        "Breach of Contract": f"Commercial contract dispute regarding non-performance of obligations filed in {year}.",
        "Trademark Infringement": f"Trademark infringement case involving brand protection and market confusion filed in {year}.",
        "Copyright Violation": f"Copyright infringement matter concerning unauthorized reproduction filed in {year}.",
        "Patent Dispute": f"Patent validity and infringement dispute in technology sector filed in {year}.",
        "Partnership Dispute": f"Partnership dissolution and profit-sharing dispute filed in {year}.",
        "Arbitration Matter": f"Arbitration award enforcement and challenge proceedings filed in {year}.",
        "Insolvency Proceedings": f"Corporate insolvency resolution process under IBC filed in {year}.",
        "Shareholder Dispute": f"Shareholder rights and corporate governance dispute filed in {year}.",
        "Intellectual Property": f"Comprehensive IP rights protection and licensing dispute filed in {year}.",
        "Commercial Fraud": f"Commercial fraud and misrepresentation case filed in {year}.",
        "Joint Venture Dispute": f"Joint venture agreement breach and profit distribution dispute filed in {year}.",
        "Licensing Agreement": f"Technology licensing and royalty payment dispute filed in {year}.",
        "Supply Chain Dispute": f"Supply chain contract breach and delivery obligations dispute filed in {year}.",
        "Merger & Acquisition Dispute": f"M&A transaction dispute regarding representations and warranties filed in {year}."
    }
    
    return descriptions.get(case_type, f"{case_type} case filed in {court} in {year}.")

def generate_training_data(num_cases=250):
    """Generate synthetic training dataset"""
    
    cases = []
    
    for i in range(num_cases):
        case_id = f"CASE_{year}_{i+1:04d}" if (year := random.randint(2015, 2024)) else f"CASE_2020_{i+1:04d}"
        court = random.choice(COURTS)
        judge = random.choice(JUDGES)
        case_type = random.choice(CASE_TYPES)
        year = random.randint(2015, 2024)
        complexity = generate_case_complexity()
        domain = random.choice(LEGAL_DOMAINS)
        
        # Calculate outcome probability
        plaintiff_prob = calculate_outcome_probability(court, case_type, year, complexity, judge)
        
        # Determine outcome
        outcome = determine_outcome(plaintiff_prob)
        
        # Generate description
        description = generate_case_description(case_type, court, year, outcome)
        
        cases.append({
            'case_id': case_id,
            'court': court,
            'judge': judge,
            'case_type': case_type,
            'year': year,
            'complexity': complexity,
            'legal_domain': domain,
            'outcome': outcome,
            'plaintiff_win_probability': round(plaintiff_prob, 3),
            'description': description
        })
    
    return pd.DataFrame(cases)

def main():
    """Main function to generate and save training data"""
    
    print("🔄 Generating synthetic training data for case outcome prediction...")
    
    # Generate data
    df = generate_training_data(num_cases=250)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(data_dir, 'training_cases.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} synthetic cases")
    print(f"📁 Saved to: {output_path}")
    print("\n📊 Dataset Statistics:")
    print(f"   - Date Range: {df['year'].min()} - {df['year'].max()}")
    print(f"   - Unique Courts: {df['court'].nunique()}")
    print(f"   - Unique Case Types: {df['case_type'].nunique()}")
    print(f"\n⚖️ Outcome Distribution:")
    print(df['outcome'].value_counts())
    print(f"\n🏛️ Court Distribution:")
    print(df['court'].value_counts().head())
    
    return df

if __name__ == "__main__":
    df = main()
