import json

with open('results/extractions/credit/include_pa/deepseek/grok/instance_0.json') as f:
    data = json.load(f)

print('Full extraction JSON structure:')
print()
print('Keys:', list(data.keys()))
print()
print('most_important_features:')
for feat in data['most_important_features']:
    print(f'  {feat}')
print()
print('features array (length={}, first 3):'.format(len(data['features'])))
if isinstance(data['features'], list):
    for feat in data['features'][:3]:
        print(f'  {feat}')
else:
    print(f'  {data["features"]}')
print()
print('Sample features with count:', len(data['features']))
