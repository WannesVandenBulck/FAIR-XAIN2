#!/usr/bin/env python
import json
from pathlib import Path

violations = {
    'law': {'deepseek': [2, 212, 283, 96], 'grok': [113, 293, 85]},
    'saudi': {'deepseek': [0, 12, 51, 55, 78]}
}

for dataset, providers in violations.items():
    print(f'\n{"="*100}')
    print(f'{dataset.upper()} VIOLATIONS - NARRATIVES')
    print(f'{"="*100}')
    
    for provider, instances in providers.items():
        for instance_id in instances[:2]:  # Show first 2 per provider
            narr_file = Path(f'results/narratives/{dataset}/exclude_pa/{provider}/{provider}-*/instance_{instance_id}.json')
            
            # Find actual file
            actual_file = None
            narr_dir = Path(f'results/narratives/{dataset}/exclude_pa/{provider}')
            for subdir in narr_dir.glob('*'):
                candidate = subdir / f'instance_{instance_id}.json'
                if candidate.exists():
                    actual_file = candidate
                    break
            
            if actual_file:
                with open(actual_file) as f:
                    data = json.load(f)
                
                narrative = data.get('narrative', '')
                
                print(f'\n{provider.upper()} | Instance {instance_id}')
                print('-' * 80)
                
                # Find any references to PA
                pa_keywords = {'gender', 'race', 'age', 'sex', 'foreign', 'health'}
                if dataset == 'saudi':
                    pa_keywords = {'gender', 'age', 'health', 'Health_Issues', 'Gender', 'Age'}
                elif dataset == 'law':
                    pa_keywords = {'gender', 'race'}
                
                # Search for PA mentions in narrative
                lines = narrative.split('\n')
                for line in lines:
                    line_lower = line.lower()
                    if any(kw.lower() in line_lower for kw in pa_keywords):
                        print(line)
                
                if not any(any(kw.lower() in line.lower() for kw in pa_keywords) for line in lines):
                    print('(No PA keywords found in narrative)')
                
                print(f'\nNarrative length: {len(narrative)} characters')
                print(f'Full narrative:\n{narrative}\n')
