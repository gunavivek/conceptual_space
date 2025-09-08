#!/usr/bin/env python3
"""
Extract single record from parquet file
"""

import pandas as pd
from pathlib import Path

def extract_single_record(record_id="finqa_test_617"):
    """Extract single record and create new parquet file"""
    
    # Load original parquet file
    data_path = Path(__file__).parent.parent / "data" / "test_mode_5_records.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"Original file has {len(df)} records")
    print(f"Looking for record: {record_id}")
    
    # Filter for specific record
    filtered_df = df[df['id'] == record_id]
    
    if len(filtered_df) == 0:
        print(f"Record {record_id} not found!")
        print("Available records:")
        print(df['id'].tolist())
        return
    
    print(f"Found record: {record_id}")
    print(f"Filtered to {len(filtered_df)} record(s)")
    
    # Save single record file
    output_path = Path(__file__).parent.parent / "data" / f"single_record_{record_id}.parquet"
    filtered_df.to_parquet(output_path, index=False)
    
    print(f"Saved single record to: {output_path}")
    
    # Display record info
    record = filtered_df.iloc[0]
    print(f"\nRecord details:")
    print(f"  ID: {record['id']}")
    print(f"  Question: {record.get('question', 'N/A')}")
    print(f"  Documents length: {len(str(record.get('documents', [])))}")
    
    return output_path

if __name__ == "__main__":
    extract_single_record("finqa_test_617")