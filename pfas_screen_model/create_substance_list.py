import os
import json

all_names = set()

# First, process root PFAS files
root_files = [f for f in os.listdir('.') if f.startswith('PFAS_detect_target_') and f.endswith('.json')]
for f in root_files:
    with open(f, 'r') as file:
        data = json.load(file)
        for item in data.get('detect_target', []):
            if item.get('substance_name'):
                all_names.add(item['substance_name'])

# Second, process the JSON_data directory
data_dir = 'JSON_data/retrieved_substances_data_enhanced_json_files_2nd_attempt_11052024'
if os.path.isdir(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    for section in ['detect_target', 'probe_material', 'test_medium_electrolyte']:
                        for item in data.get(section, []):
                            if item.get('substance_name'):
                                all_names.add(item['substance_name'])
            except Exception:
                continue # Ignore broken json files

# Write to file
with open('available_substances.txt', 'w') as f:
    for name in sorted(list(all_names)):
        f.write(name + '\n')

print(f"Successfully created available_substances.txt with {len(all_names)} unique substance names.")
