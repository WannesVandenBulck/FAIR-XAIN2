#!/usr/bin/env python
"""
Analyze all PA mentions in exclude_pa narratives.
"""
import json
from pathlib import Path

# Track all violations with details
violations = []

datasets_info = {
    'credit': {'pa_list': ['age', 'sex', 'foreign_worker']},
    'law': {'pa_list': ['gender', 'race']},
    'saudi': {'pa_list': ['gender', 'age', 'health_issues']},
    'student': {'pa_list': ['sex', 'age', 'health']}
}

print('='*100)
print('ANALYZING PA MENTIONS IN EXCLUDE_PA NARRATIVES')
print('='*100)

for dataset, info in datasets_info.items():
    print(f'\n{dataset.upper()}')
    print('-'*100)
    
    pa_list = info['pa_list']
    exclude_dir = Path(f'results/extractions/{dataset}/exclude_pa')
    
    if not exclude_dir.exists():
        continue
    
    # Check all providers
    for provider_dir in sorted(exclude_dir.glob('*')):
        if not provider_dir.is_dir():
            continue
        
        provider = provider_dir.name
        
        # Navigate through nested subdirectories
        for nested_dir in sorted(provider_dir.glob('*')):
            if not nested_dir.is_dir():
                continue
            
            for json_file in sorted(nested_dir.glob('instance_*.json')):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    # Check for PA mentions
                    instance_id = int(json_file.stem.replace('instance_', ''))
                    mentioned_pas = []
                    
                    for feat in data.get('features', []):
                        fname = feat.get('name', '').lower()
                        if fname in pa_list and feat.get('mentioned', 0) == 1:
                            mentioned_pas.append({
                                'name': feat.get('name'),
                                'value': feat.get('value')
                            })
                    
                    if mentioned_pas:
                        narrative = data.get('narrative', '').strip()
                        
                        violations.append({
                            'dataset': dataset,
                            'provider': provider,
                            'instance_id': instance_id,
                            'pa_mentioned': mentioned_pas,
                            'narrative': narrative
                        })
                        
                        print(f'\n  {provider:10s} | Instance {instance_id:3d}')
                        print(f'  PA mentioned: {", ".join([f"{p["name"]}={p["value"]}" for p in mentioned_pas])}')
                        print(f'  Narrative (first 500 chars):')
                        print(f'  {narrative[:500]}')
                        if len(narrative) > 500:
                            print(f'  ...')
                        
                except Exception as e:
                    pass

print(f'\n\n' + '='*100)
print(f'TOTAL VIOLATIONS: {len(violations)}')
print('='*100)

# Group by dataset and provider
print('\nSUMMARY BY DATASET & PROVIDER:')
for dataset in datasets_info.keys():
    dataset_viols = [v for v in violations if v['dataset'] == dataset]
    if dataset_viols:
        print(f'\n  {dataset.upper()}: {len(dataset_viols)} total')
        for provider in ['grok', 'openai', 'deepseek']:
            prov_viols = [v for v in dataset_viols if v['provider'] == provider]
            if prov_viols:
                print(f'    {provider:10s}: {len(prov_viols)} violations')

# Detailed analysis by PA attribute
print('\n' + '='*100)
print('ANALYSIS BY PA ATTRIBUTE')
print('='*100)

for dataset, info in datasets_info.items():
    dataset_viols = [v for v in violations if v['dataset'] == dataset]
    if dataset_viols:
        print(f'\n{dataset.upper()}:')
        pa_counts = {}
        for v in dataset_viols:
            for pa in v['pa_mentioned']:
                pa_name = pa['name']
                if pa_name not in pa_counts:
                    pa_counts[pa_name] = []
                pa_counts[pa_name].append({
                    'provider': v['provider'],
                    'instance': v['instance_id'],
                    'value': pa['value']
                })
        
        for pa_name in sorted(pa_counts.keys()):
            mentions = pa_counts[pa_name]
            print(f'\n  {pa_name}:')
            print(f'    Total mentions: {len(mentions)}')
            
            # Show value distribution
            values = {}
            for m in mentions:
                val = str(m['value'])
                if val not in values:
                    values[val] = []
                values[val].append(m)
            
            for val in sorted(values.keys()):
                val_mentions = values[val]
                print(f'    Value "{val}": {len(val_mentions)} mentions')
                for m in val_mentions[:3]:  # Show first 3
                    print(f'      - {m["provider"]} instance {m["instance"]}')
                if len(val_mentions) > 3:
                    print(f'      ... and {len(val_mentions)-3} more')
