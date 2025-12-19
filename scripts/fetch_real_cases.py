"""
Fetch real legal cases from Indian Kanoon API to expand the training dataset.
"""

import os
import requests
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("INDIAN_KANOON_API_TOKEN")
BASE_URL = "https://api.indiankanoon.org"

def fetch_cases(query, max_pages=2):
    """Fetch cases for a query"""
    if not API_TOKEN:
        print("❌ Error: INDIAN_KANOON_API_TOKEN not set.")
        return []

    print(f"🔍 Fetching cases for query: '{query}'...")
    cases = []
    
    headers = {"Authorization": f"Token {API_TOKEN}"}
    
    for page in range(max_pages):
        try:
            url = f"{BASE_URL}/search/"
            params = {
                "formInput": query,
                "pagenum": page
            }
            
            resp = requests.post(url, headers=headers, data=params, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                docs = data.get('docs', [])
                
                for doc in docs:
                    # Basic extraction
                    case = {
                        'title': doc.get('title', ''),
                        'description': doc.get('headline', '') + " " + doc.get('doc', ''), # 'doc' might not be full text in search
                        'court': doc.get('docsource', 'Unknown Court'),
                        'date': doc.get('publishdate', ''),
                        'outcome': infer_outcome(doc.get('headline', '')) # Heuristic
                    }
                    cases.append(case)
                
                print(f"   Page {page}: Found {len(docs)} cases")
                time.sleep(1) # Rate limit
            else:
                print(f"   Error fetching page {page}: {resp.status_code}")
                
        except Exception as e:
            print(f"   Exception: {e}")
            
    return cases

def infer_outcome(text):
    """Heuristic to label outcome based on text snippet"""
    text = text.lower()
    if 'dismissed' in text or 'reject' in text or 'denied' in text or 'convicted' in text or 'dismissal' in text:
        return 'dismissed' # roughly maps to negative for appellant/petitioner
    elif 'allowed' in text or 'granted' in text or 'decreed' in text or 'acquitted' in text or 'quashed' in text or 'set aside' in text:
        return 'allowed' # roughly maps to positive
    elif 'partly allowed' in text or 'modified' in text or 'partly' in text:
        return 'partly_allowed'
    else:
        return 'unknown'

def main():
    # Expanded query list for diverse case types
    queries = [
        "breach of contract damages",
        "trademark infringement injunction",
        "murder appeal supreme court",
        "property dispute partition",
        "divorce cruelty grounds",
        "anticipatory bail application",
        "cheque bounce 138 ni act",
        "consumer protection deficiency service",
        "motor accident claim tribunal",
        "arbitration award challenge",
        "specific performance agreement",
        "quashing fir 482 crpc",
        "custody minor child",
        "medical negligence compensation"
    ]
    
    all_cases = []
    # Fetch 3 pages per query -> ~30-40 cases per query * 14 queries ~ 400-500 potential raw cases
    for q in queries:
        all_cases.extend(fetch_cases(q, max_pages=3))
        
    if all_cases:
        df = pd.DataFrame(all_cases)
        # Filter unknown outcomes
        df_clean = df[df['outcome'] != 'unknown']
        
        output_path = "1-Rag/data/real_cases_fetched.csv"
        
        # Check if file exists to append or create new
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                final_df = pd.concat([existing_df, df_clean]).drop_duplicates(subset=['title'])
            except:
                final_df = df_clean
        else:
            final_df = df_clean
            
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Total valid cases in dataset: {len(final_df)} (Added {len(df_clean)} new)")
    else:
        print("\n⚠️ No cases found.")

if __name__ == "__main__":
    main()
