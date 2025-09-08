import pandas as pd

df = pd.read_parquet('sample_20_records.parquet')
print('Actual questions in sample data:')
print('='*60)
for i in range(min(5, len(df))):
    print(f"Record {i+1}: {df.iloc[i]['id']}")
    print(f"Question: {df.iloc[i]['question']}")
    print('-'*40)