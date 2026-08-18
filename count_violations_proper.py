#!/usr/bin/env python
import pandas as pd

print('='*80)
print('PA VIOLATIONS BY INSTANCE')
print('='*80)

for dataset in ['law', 'saudi']:
    csv_file = f'results/fairness_eval/per_narrative_metrics_{dataset}.csv'
    
    df = pd.read_csv(csv_file)
    exclude_pa = df[df['condition'] == 'exclude_pa']
    
    # Find all pa_*_mentioned columns
    pa_mentioned_cols = [col for col in df.columns if 'pa_' in col and '_mentioned' in col]
    
    print(f'\n{dataset.upper()}:')
    
    # For each instance, count how many PA are mentioned
    instance_pa_counts = {}
    for _, row in exclude_pa.iterrows():
        instance_id = row['instance_id']
        provider = row['provider']
        
        key = f'{provider}_instance_{int(instance_id)}'
        pa_count = sum(1 for col in pa_mentioned_cols if row[col] == 1)
        
        if pa_count > 0:
            if key not in instance_pa_counts:
                instance_pa_counts[key] = {'count': 0, 'names': []}
            instance_pa_counts[key]['count'] += pa_count
            
            for col in pa_mentioned_cols:
                if row[col] == 1:
                    pa_name = col.replace('pa_', '').replace('_mentioned', '')
                    if pa_name not in instance_pa_counts[key]['names']:
                        instance_pa_counts[key]['names'].append(pa_name)
    
    print(f'Total affected instances: {len(instance_pa_counts)}')
    total_mentions = sum(v['count'] for v in instance_pa_counts.values())
    print(f'Total PA mentions: {total_mentions}')
    print(f'\nBreakdown:')
    
    for key in sorted(instance_pa_counts.keys()):
        v = instance_pa_counts[key]
        pa_list = ' + '.join(v['names'])
        print(f'  {key}: {v["count"]} mention(s) - {pa_list}')
