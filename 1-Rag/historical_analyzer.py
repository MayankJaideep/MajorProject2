"""
Historical Win Rate Analysis Module
Analyze historical case outcomes to provide context for predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import os

class HistoricalAnalyzer:
    """
    Analyze historical case data to calculate win rates and provide context
    """
    
    def __init__(self, data_path: str = "1-Rag/data/training_cases.csv"):
        """
        Initialize with historical case data
        
        Args:
            data_path: Path to CSV file with historical cases
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load historical case data"""
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} historical cases")
        else:
            print(f"⚠️  No historical data found at {self.data_path}")
            self.df = pd.DataFrame()
    
    def get_win_rate(
        self,
        case_type: Optional[str] = None,
        court: Optional[str] = None,
        year_range: Optional[tuple] = None
    ) -> Dict[str, any]:
        """
        Calculate plaintiff win rate for similar cases
        
        Args:
            case_type: Filter by case type
            court: Filter by court
            year_range: Tuple of (min_year, max_year)
        
        Returns:
            Dictionary with win rate statistics
        """
        if self.df.empty:
            return {
                'win_rate': 0.0,
                'total_cases': 0,
                'plaintiff_wins': 0,
                'error': 'No historical data available'
            }
        
        # Filter data
        filtered_df = self.df.copy()
        
        if case_type:
            filtered_df = filtered_df[filtered_df['case_type'] == case_type]
        
        if court:
            filtered_df = filtered_df[filtered_df['court'] == court]
        
        if year_range:
            min_year, max_year = year_range
            filtered_df = filtered_df[
                (filtered_df['year'] >= min_year) & 
                (filtered_df['year'] <= max_year)
            ]
        
        total_cases = len(filtered_df)
        
        if total_cases == 0:
            return {
                'win_rate': 0.0,
                'total_cases': 0,
                'plaintiff_wins': 0,
                'defendant_wins': 0,
                'settlements': 0,
                'dismissed': 0,
                'error': 'No matching cases found'
            }
        
        # Count outcomes
        outcome_counts = filtered_df['outcome'].value_counts().to_dict()
        
        plaintiff_wins = outcome_counts.get('plaintiff_win', 0)
        defendant_wins = outcome_counts.get('defendant_win', 0)
        settlements = outcome_counts.get('settlement', 0)
        dismissed = outcome_counts.get('dismissed', 0)
        
        win_rate = plaintiff_wins / total_cases if total_cases > 0 else 0.0
        
        return {
            'win_rate': round(win_rate, 3),
            'total_cases': total_cases,
            'plaintiff_wins': plaintiff_wins,
            'defendant_wins': defendant_wins,
            'settlements': settlements,
            'dismissed': dismissed,
            'breakdown': {
                'plaintiff_win': round(plaintiff_wins / total_cases, 3) if total_cases > 0 else 0,
                'defendant_win': round(defendant_wins / total_cases, 3) if total_cases > 0 else 0,
                'settlement': round(settlements / total_cases, 3) if total_cases > 0 else 0,
                'dismissed': round(dismissed / total_cases, 3) if total_cases > 0 else 0
            },
            'error': None
        }
    
    def get_court_statistics(self, court: str) -> Dict[str, any]:
        """Get detailed statistics for a specific court"""
        if self.df.empty:
            return {'error': 'No data available'}
        
        court_cases = self.df[self.df['court'] == court]
        
        if len(court_cases) == 0:
            return {'error': f'No cases found for {court}'}
        
        stats = {
            'total_cases': len(court_cases),
            'win_rate': self.get_win_rate(court=court)['win_rate'],
            'avg_complexity': court_cases['complexity'].mean(),
            'case_types': court_cases['case_type'].value_counts().to_dict(),
            'yearly_trend': court_cases.groupby('year')['outcome'].apply(
                lambda x: (x == 'plaintiff_win').sum() / len(x)
            ).to_dict()
        }
        
        return stats
    
    def find_similar_cases(
        self,
        case_type: str,
        court: str,
        year: int,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Find most similar historical cases
        
        Args:
            case_type: Current case type
            court: Current court
            year: Current year
            top_n: Number of similar cases to return
        
        Returns:
            List of similar cases with details
        """
        if self.df.empty:
            return []
        
        # Calculate similarity score
        df_copy = self.df.copy()
        
        # Score based on matching criteria
        df_copy['similarity'] = 0
        df_copy.loc[df_copy['case_type'] == case_type, 'similarity'] += 3
        df_copy.loc[df_copy['court'] == court, 'similarity'] += 2
        df_copy.loc[abs(df_copy['year'] - year) <= 2, 'similarity'] += 1
        
        # Sort by similarity
        similar = df_copy.nlargest(top_n, 'similarity')
        
        # Convert to list of dicts
        results = []
        for _, row in similar.iterrows():
            results.append({
                'case_id': row.get('case_id', 'Unknown'),
                'case_type': row['case_type'],
                'court': row['court'],
                'year': row['year'],
                'outcome': row['outcome'],
                'similarity_score': row['similarity'],
                'description': row.get('description', '')
            })
        
        return results
    
    def get_trend_analysis(
        self,
        case_type: str,
        years: int = 5
    ) -> Dict[str, any]:
        """
        Analyze trends over recent years
        
        Args:
            case_type: Case type to analyze
            years: Number of recent years to analyze
        
        Returns:
            Trend analysis data
        """
        if self.df.empty:
            return {'error': 'No data available'}
        
        # Filter by case type
        filtered = self.df[self.df['case_type'] == case_type]
        
        # Get recent years
        max_year = filtered['year'].max()
        min_year = max_year - years + 1
        
        recent = filtered[filtered['year'] >= min_year]
        
        # Calculate yearly win rates
        yearly_stats = recent.groupby('year').apply(
            lambda x: pd.Series({
                'total': len(x),
                'plaintiff_wins': (x['outcome'] == 'plaintiff_win').sum(),
                'win_rate': (x['outcome'] == 'plaintiff_win').sum() / len(x)
            })
        ).to_dict('index')
        
        return {
            'case_type': case_type,
            'years_analyzed': years,
            'yearly_stats': yearly_stats,
            'overall_trend': 'increasing' if list(yearly_stats.values())[-1]['win_rate'] > list(yearly_stats.values())[0]['win_rate'] else 'decreasing'
        }

# Test function
def test_historical_analyzer():
    """Test the historical analyzer"""
    
    print("=" * 80)
    print("HISTORICAL WIN RATE ANALYSIS TEST")
    print("=" * 80)
    
    analyzer = HistoricalAnalyzer()
    
    # Test 1: Overall win rate
    print("\n🔍 Test 1: Overall Win Rate")
    overall = analyzer.get_win_rate()
    print(f"   Total Cases: {overall['total_cases']}")
    print(f"   Plaintiff Win Rate: {overall['win_rate']:.1%}")
    print(f"   Breakdown:")
    for outcome, rate in overall['breakdown'].items():
        print(f"      • {outcome.replace('_', ' ').title()}: {rate:.1%}")
    
    # Test 2: Win rate by case type
    print("\n🔍 Test 2: Trademark Cases Win Rate")
    trademark = analyzer.get_win_rate(case_type="Trademark Infringement")
    print(f"   Total Trademark Cases: {trademark['total_cases']}")
    print(f"   Win Rate: {trademark['win_rate']:.1%}")
    
    # Test 3: Win rate by court
    print("\n🔍 Test 3: Bombay High Court Win Rate")
    bombay = analyzer.get_win_rate(court="Bombay High Court")
    print(f"   Total Cases: {bombay['total_cases']}")
    print(f"   Win Rate: {bombay['win_rate']:.1%}")
    
    # Test 4: Find similar cases
    print("\n🔍 Test 4: Find Similar Cases")
    similar = analyzer.find_similar_cases(
        case_type="Trademark Infringement",
        court="Bombay High Court",
        year=2023,
        top_n=3
    )
    print(f"   Found {len(similar)} similar cases:")
    for case in similar:
        print(f"      • {case['case_type']} ({case['year']}) - {case['outcome']} (similarity: {case['similarity_score']})")
    
    print("\n" + "=" * 80)
    print("✅ Historical analysis tests complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_historical_analyzer()
