import json
from pathlib import Path

# Check young instances for Gemini
young_instances = [1, 3, 8, 9, 11, 14, 17, 19, 21, 22, 24, 25, 27, 29, 30, 32, 33]

provider = 'gemini'

print("="*100)
print("CHECKING: Credit History and Property mentions in YOUNG group for Gemini")
print("="*100)

for instance_idx in young_instances:
    json_file = Path(f"results/extractions/majority/{provider}/instance_{instance_idx}.json")
    
    try:
        with open(json_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Instance {instance_idx}: FILE NOT FOUND")
        continue
    
    # Find credit_history and property
    credit_history_mentioned = False
    property_mentioned = False
    
    for feature in data.get('features', []):
        if feature['name'] == 'credit_history':
            credit_history_mentioned = feature.get('mentioned', 0) == 1
        if feature['name'] == 'property':
            property_mentioned = feature.get('mentioned', 0) == 1
    
    ch_status = "✓ MENTIONED" if credit_history_mentioned else "✗ not mentioned"
    p_status = "✓ MENTIONED" if property_mentioned else "✗ not mentioned"
    
    print(f"Instance {instance_idx:2d}: credit_history {ch_status:20s} | property {p_status}")

print("\n" + "="*100)
print("CHECKING: Credit History and Property mentions in OLD group for Gemini")
print("="*100)

old_instances = [0, 2, 4, 5, 6, 7, 10, 12, 13, 15, 16, 18, 20, 23, 26, 28, 31]

for instance_idx in old_instances:
    json_file = Path(f"results/extractions/majority/{provider}/instance_{instance_idx}.json")
    
    try:
        with open(json_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Instance {instance_idx}: FILE NOT FOUND")
        continue
    
    # Find credit_history and property
    credit_history_mentioned = False
    property_mentioned = False
    
    for feature in data.get('features', []):
        if feature['name'] == 'credit_history':
            credit_history_mentioned = feature.get('mentioned', 0) == 1
        if feature['name'] == 'property':
            property_mentioned = feature.get('mentioned', 0) == 1
    
    ch_status = "✓ MENTIONED" if credit_history_mentioned else "✗ not mentioned"
    p_status = "✓ MENTIONED" if property_mentioned else "✗ not mentioned"
    
    print(f"Instance {instance_idx:2d}: credit_history {ch_status:20s} | property {p_status}")
