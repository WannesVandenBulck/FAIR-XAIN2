#!/usr/bin/env python
import re
from pathlib import Path

prompt_files = [
    'llm_tools/prompts/prompt_credit.py',
    'llm_tools/prompts/prompt_law.py',
    'llm_tools/prompts/prompt_saudi.py',
    'llm_tools/prompts/prompt_student.py'
]

print('='*80)
print('CHECKING FOR PA MENTIONS IN DATASET DESCRIPTIONS')
print('='*80)

for pfile in prompt_files:
    dataset = pfile.split('prompt_')[1].split('.')[0]
    
    with open(pfile) as f:
        content = f.read()
    
    # Look for the line that says "Protected attributes" or similar
    pa_lines = []
    for line in content.split('\n'):
        if 'protected' in line.lower() and ('attribute' in line.lower() or 'sex' in line.lower() or 'age' in line.lower() or 'gender' in line.lower() or 'race' in line.lower() or 'health' in line.lower()):
            pa_lines.append(line.strip())
    
    print(f'\n{dataset.upper()}:')
    
    if pa_lines:
        for line in pa_lines[:2]:
            print(f'  {line}')
    else:
        print('  No explicit PA mentions found')
