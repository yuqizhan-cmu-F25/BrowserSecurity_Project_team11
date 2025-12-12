#!/usr/bin/env python3
"""
Extract VisibleV8 data from PostgreSQL database.
"""
import psycopg2
import pandas as pd
from pathlib import Path
import os

# Get the script's directory and resolve the output path relative to project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Database connection parameters
DB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5434,
    'dbname': 'vv8_backend',
    'user': 'vv8',
    'password': 'vv8'
}

def connect_to_db( ):
    """Connect to PostgreSQL database."""
    return psycopg2.connect(**DB_CONFIG)

def extract_all_crawls():
    """Extract all crawl data."""
    conn = connect_to_db()
    query = """
    SELECT id, url, start_time, end_time, postprocessor_used
    FROM submissions ORDER BY start_time DESC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def extract_mega_features(submission_id):
    """Extract Mfeatures data (mega_usages) for a specific submission."""
    conn = connect_to_db()
    query = """
    SELECT 
        mu.instance_id, mu.feature_id, mf.full_name as feature_name,
        mf.receiver_name, mf.member_name, mu.usage_offset, mu.usage_mode,
        mu.usage_count, mi.script_id, u_script.url_full as script_url,
        u_origin.url_full as origin_url
    FROM mega_usages mu
    JOIN mega_instances mi ON mu.instance_id = mi.id
    JOIN logfile l ON mi.logfile_id = l.id
    JOIN submissions s ON l.submissionid = s.id
    LEFT JOIN mega_features mf ON mu.feature_id = mf.id
    LEFT JOIN urls u_script ON mi.script_url_id = u_script.id
    LEFT JOIN urls u_origin ON mu.origin_url_id = u_origin.id
    WHERE s.id = %s ORDER BY mu.instance_id, mu.feature_id;
    """
    df = pd.read_sql_query(query, conn, params=(submission_id,))
    conn.close()
    return df

def save_extracted_data(output_dir=None):
    """Extract and save all data."""
    if output_dir is None:
        # Default to project root's data/processed directory
        output_path = PROJECT_ROOT / 'data' / 'processed'
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Extracting crawl data...")
    crawls_df = extract_all_crawls()
    crawls_df.to_csv(output_path / 'crawls_summary.csv', index=False)
    print(f"Saved {len(crawls_df)} crawls to {output_path / 'crawls_summary.csv'}")
    
    print("\nExtracting feature usage data...")
    for idx, row in crawls_df.iterrows():
        submission_id = row['id']
        url = row['url']
        postprocessor = row['postprocessor_used']
        print(f"Processing {url}...")
        
        try:
            if postprocessor and 'Mfeatures' in postprocessor:
                mega_df = extract_mega_features(submission_id)
                if not mega_df.empty:
                    safe_filename = url.replace('https://', '' ).replace('http://', '' ).replace('/', '_').replace(':', '_')
                    mega_df.to_csv(output_path / f'mega_features_{safe_filename}.csv', index=False)
                    print(f"  Found and saved {len(mega_df)} Mfeatures")
                else:
                    print("  No Mfeatures found")
            else:
                print("  Mfeatures not used for this crawl")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nData extraction complete! Files saved to {output_path}")

if __name__ == '__main__':
    save_extracted_data()