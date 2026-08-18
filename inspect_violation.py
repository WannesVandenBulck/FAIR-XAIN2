#!/usr/bin/env python
import json
from pathlib import Path

violations = [
    ('law', 'deepseek', 2),
    ('law', 'grok', 113),
    ('saudi', 'deepseek', 0)
]

for dataset, provider, instance in violations:
    exclude_dir = Path(f'results/extractions/{dataset}/exclude_pa/{provider}')
    
    # Try to find the instance file
    json_file = None
    for nested_dir in exclude_dir.glob('*'):
        candidate = nested_dir / f'instance_{instance}.json'
        if candidate.exists():
            json_file = candidate
            break
    
    if json_file:
        with open(json_file) as f:
            data = json.load(f)
        
        print(f'\n{"="*80}')
        print(f'{dataset.upper()} | {provider} | Instance {instance}')
        print(f'File: {json_file}')
        print(f'{"="*80}')
        print(f'Keys in JSON: {list(data.keys())}')
        print(f'Narrative length: {len(data.get("narrative", ""))}')
        print(f'Narrative first 200 chars: {repr(data.get("narrative", "")[:200])}')
        
        # Show features marked as mentioned
        print(f'\nFeatures marked as mentioned (mentioned=1):')
        for feat in data.get('features', []):
            if feat.get('mentioned') == 1:
                fname = feat.get('name', '')
                fvalue = feat.get('value')
                print(f'  - {fname} = {fvalue}')
        
        # Show most important features
        print(f'\nMost important features:')
        for i, feat in enumerate(data.get('most_important_features', [])[:3]):
            fname = feat.get('name', '')
            fvalue = feat.get('value')
            print(f'  {i+1}. {fname} = {fvalue}')
    else:
        print(f'\nFile not found: {dataset}/{provider}/{instance}')
