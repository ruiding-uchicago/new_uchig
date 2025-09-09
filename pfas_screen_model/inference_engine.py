import os
import json
import argparse
from collections import Counter
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric imports might need to be installed: pip install torch_geometric
try:
    from torch_geometric.data import Data, DataLoader
except ImportError:
    print("PyTorch Geometric not found. Please install it: pip install torch_geometric")
    Data = None
    DataLoader = None
from torch_geometric.nn import GraphConv, global_mean_pool

# --- Constants ---
DEFAULT_MODEL_DIR = "saved-models"
DEFAULT_DATA_DIR = "JSON_data/retrieved_substances_data_enhanced_json_files_2nd_attempt_11052024"
MODEL_FILENAME = "pfas_multimodal.pt"
CONFIG_FILENAME = "config.json"

# --- Model Class Definitions (Copied from training/evaluation scripts) ---
class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, beta):
        super().__init__()
        self.beta = beta
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = self.fc1(x); x = self.relu(x)
        x = self.beta * self.fc2(x)
        return x

class GNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers):
        super().__init__()
        self.convs = nn.ModuleList([
            GraphConv(num_features if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])
        self.lin = nn.Linear(hidden_channels, num_classes)
    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index); x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

class MultiModalModel(nn.Module):
    def __init__(self, snn_input_size, snn_hidden_size, gnn_input_size, gnn_hidden_size,
                 num_classes, snn_beta=0.85, gnn_layers=7):
        super().__init__()
        self.snn = SNN(snn_input_size, snn_hidden_size, num_classes, beta=snn_beta)
        self.gnn = GNN(gnn_input_size, gnn_hidden_size, num_classes, num_layers=gnn_layers)
        self.fusion_layer = nn.Linear(num_classes * 2, num_classes)
    def forward(self, snn_batch, gnn_batch):
        snn_logits = self.snn(snn_batch)
        gnn_logits = self.gnn(gnn_batch)
        fused = torch.cat([snn_logits, gnn_logits], dim=-1)
        return self.fusion_layer(fused)

# --- Data Processing Functions (Copied and adapted) ---
def extract_features_with_magpie(descriptors: dict, substance_type: str) -> torch.Tensor:
    numerical_values = []
    # Simplified extraction logic from eval script
    for key, value in descriptors.items():
        if key == "CID" or (substance_type == "inorganic solid" and "MagpieData" in key):
            continue
        if isinstance(value, (int, float)):
            numerical_values.append(float(value))
        elif isinstance(value, str):
            try:
                numerical_values.append(float(value))
            except ValueError:
                pass
    if substance_type == "inorganic solid" and descriptors.get("Magpie Descriptors") is not None:
        for k, v in descriptors["Magpie Descriptors"].items():
            if isinstance(v, (int, float)):
                numerical_values.append(float(v))
    return torch.tensor(numerical_values, dtype=torch.float)

def aggregate_by_type(items: list, max_feature_length: int) -> torch.Tensor:
    types = {'small molecule': [], 'inorganic solid': [], 'polymer': []}
    for item in items:
        substance_type = item.get('substance_type', '').lower()
        if substance_type in types:
            feats = extract_features_with_magpie(item.get('substance_descriptors', {}), substance_type)
            # Pad or truncate features
            if feats.shape[0] < max_feature_length:
                feats = F.pad(feats, (0, max_feature_length - feats.shape[0]))
            else:
                feats = feats[:max_feature_length]
            types[substance_type].append(feats)
    
    aggregated = []
    for key in ['small molecule', 'inorganic solid', 'polymer']:
        if types[key]:
            aggregated.append(torch.mean(torch.stack(types[key]), dim=0))
        else:
            aggregated.append(torch.zeros(max_feature_length))
    return torch.cat(aggregated)

def generate_graph_from_json(data: dict, max_feature_length: int) -> Data:
    target_node = aggregate_by_type(data.get('detect_target', []), max_feature_length)
    probe_node  = aggregate_by_type(data.get('probe_material', []), max_feature_length)
    medium_node = aggregate_by_type(data.get('test_medium_electrolyte', []), max_feature_length)

    cond_feats_list = [
        data.get("test_operating_temperature_celsius", 25.0),
        data.get("min_pH_when_testing", 7.0),
        data.get("max_pH_when_testing", 7.0),
    ]
    cond_tensor = torch.tensor(cond_feats_list, dtype=torch.float)
    
    # Pad or truncate conditions node
    needed_len = 3 * max_feature_length
    if cond_tensor.numel() < needed_len:
        cond_tensor = F.pad(cond_tensor, (0, needed_len - cond_tensor.numel()))
    else:
        cond_tensor = cond_tensor[:needed_len]

    x = torch.stack([target_node, probe_node, medium_node, cond_tensor])
    edges = [(0, 1), (0, 2), (1, 2), (3, 2)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def extract_and_concatenate_vectors(substances: list) -> np.ndarray:
    vector_data = []
    for substance in substances:
        descriptors = substance.get('substance_descriptors', {})
        for key in ["Morgan_128", "maccs_fp", "morgan_fp_128"]:
            value = descriptors.get(key, [])
            if isinstance(value, list):
                vector_data.extend(value)
    return np.array(vector_data, dtype=float)

# --- Core Engine Functions ---

def load_model_and_config(model_dir=DEFAULT_MODEL_DIR):
    """Loads the trained model and its configuration."""
    print("Loading model and configuration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = os.path.join(model_dir, CONFIG_FILENAME)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    model = MultiModalModel(
        snn_input_size=int(cfg["snn"]["input_size"]),
        snn_hidden_size=int(cfg["snn"]["hidden_size"]),
        gnn_input_size=int(cfg["gnn"]["input_size"]),
        gnn_hidden_size=int(cfg["gnn"]["hidden_size"]),
        num_classes=int(cfg["num_classes"]),
        snn_beta=float(cfg["snn"]["beta"]),
        gnn_layers=int(cfg["gnn"]["layers"])
    ).to(device)

    state_path = os.path.join(model_dir, MODEL_FILENAME)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    return model, cfg, device

def load_database(data_dir=DEFAULT_DATA_DIR):
    """Loads all substances, conditions, etc., from the JSON data directory."""
    print("Loading substance database...")
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found at '{data_dir}'")
        return None

    substances = {}
    probes, mediums, targets = [], [], []
    conditions = set()

    for filename in os.listdir(data_dir):
        if filename.endswith('_original.json'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for item in data.get('probe_material', []):
                name = item.get('substance_name')
                if name and name not in substances:
                    substances[name] = item
                    probes.append(item)

            for item in data.get('test_medium_electrolyte', []):
                name = item.get('substance_name')
                if name and name not in substances:
                    substances[name] = item
                    mediums.append(item)
            
            for item in data.get('detect_target', []):
                 name = item.get('substance_name')
                 if name and name not in substances:
                    substances[name] = item
                    targets.append(item)

            cond = (
                data.get("test_operating_temperature_celsius", 25.0),
                data.get("min_pH_when_testing", 7.0),
                data.get("max_pH_when_testing", 7.0)
            )
            conditions.add(cond)
    
    print(f"Database loaded: {len(substances)} unique substances, {len(conditions)} unique conditions.")
    return {
        "substances_by_name": substances,
        "all_probes": probes,
        "all_mediums": mediums,
        "all_targets": targets, # All substances can be targets
        "all_conditions": list(conditions)
    }

def run_inference(model, config, device, database, target=None, probe=None, medium=None, condition=None):
    """
    Runs flexible inference. Provide any combination of T, P, M, C.
    If all are provided, it performs scoring.
    If some are missing, it performs screening for the missing parts.
    """
    # --- Determine Mode: Scoring vs. Screening ---
    inputs = {'target': target, 'probe': probe, 'medium': medium, 'condition': condition}
    knowns = {k: v for k, v in inputs.items() if v is not None}
    unknowns = [k for k, v in inputs.items() if v is None]

    if not unknowns:
        # --- SCORING MODE ---
        # print("Mode: Scoring")
        virtual_exp = {
            'detect_target': [target],
            'probe_material': [probe],
            'test_medium_electrolyte': [medium],
            'test_operating_temperature_celsius': condition[0],
            'min_pH_when_testing': condition[1],
            'max_pH_when_testing': condition[2]
        }
        
        # Preprocess
        snn_input_size = config['snn']['input_size']
        max_feature_length = config['max_feature_length']

        vec = np.concatenate([
            extract_and_concatenate_vectors(virtual_exp['detect_target']),
            extract_and_concatenate_vectors(virtual_exp['probe_material']),
            extract_and_concatenate_vectors(virtual_exp['test_medium_electrolyte'])
        ]).astype(float)
        vec = np.pad(vec, (0, snn_input_size - vec.shape[0]), 'constant') if vec.shape[0] < snn_input_size else vec[:snn_input_size]
        
        snn_tensor = torch.tensor(vec, dtype=torch.float).unsqueeze(0).to(device)
        gnn_data = generate_graph_from_json(virtual_exp, max_feature_length).to(device)
        gnn_loader = DataLoader([gnn_data], batch_size=1) # Create a loader for one item

        # Predict
        with torch.no_grad():
           for gnn_batch in gnn_loader: # A bit of a hack to get the batched graph
                output = model(snn_tensor, gnn_batch)
                prediction = output.argmax(dim=1).item()
        
        return {"prediction_label": prediction}

    else:
        # --- SCREENING MODE ---
        # print(f"Mode: Screening for {unknowns}")
        
        # Define iteration space
        iteration_map = {
            'target': [target] if target else database['all_targets'],
            'probe': [probe] if probe else database['all_probes'],
            'medium': [medium] if medium else database['all_mediums'],
            'condition': [condition] if condition else database['all_conditions']
        }
        
        # Get iterators for unknown variables
        iterators = [iteration_map[k] for k in unknowns]
        
        snn_input_size = config['snn']['input_size']
        max_feature_length = config['max_feature_length']
        
        all_combinations = []
        all_snn_vectors = []
        all_gnn_graphs = []

        # print("Generating and processing combinations...")
        # This can be very slow if multiple fields are unknown!
        for combo in product(*iterators):
            # Create the virtual experiment from knowns and the current combo
            virtual_exp_knowns = knowns.copy()
            for i, unknown_key in enumerate(unknowns):
                virtual_exp_knowns[unknown_key] = combo[i]

            t, p, m, c = virtual_exp_knowns['target'], virtual_exp_knowns['probe'], virtual_exp_knowns['medium'], virtual_exp_knowns['condition']

            # Prepare data for model
            exp_dict = {
                'detect_target': [t], 'probe_material': [p], 'test_medium_electrolyte': [m],
                'test_operating_temperature_celsius': c[0], 'min_pH_when_testing': c[1], 'max_pH_when_testing': c[2]
            }

            vec = np.concatenate([
                extract_and_concatenate_vectors(exp_dict['detect_target']),
                extract_and_concatenate_vectors(exp_dict['probe_material']),
                extract_and_concatenate_vectors(exp_dict['test_medium_electrolyte'])
            ]).astype(float)
            vec = np.pad(vec, (0, snn_input_size - vec.shape[0]), 'constant') if vec.shape[0] < snn_input_size else vec[:snn_input_size]
            
            all_snn_vectors.append(vec)
            all_gnn_graphs.append(generate_graph_from_json(exp_dict, max_feature_length))
            all_combinations.append(combo)

        if not all_combinations:
            return "No combinations found to test."

        # print(f"Predicting on {len(all_combinations)} combinations...")
        # Batch predict
        snn_tensor = torch.tensor(np.stack(all_snn_vectors, axis=0), dtype=torch.float)
        snn_loader = DataLoader(list(zip(snn_tensor, torch.zeros(len(snn_tensor)))), batch_size=64)
        gnn_loader = DataLoader(all_gnn_graphs, batch_size=64)
        
        all_preds = []
        with torch.no_grad():
            for (snn_batch, _), gnn_batch in zip(snn_loader, gnn_loader):
                snn_batch = snn_batch.to(device)
                gnn_batch = gnn_batch.to(device)
                out = model(snn_batch, gnn_batch)
                all_preds.extend(out.argmax(dim=1).cpu().tolist())

        # Filter for best results (label 0) and rank
        successful_combos = []
        for i, pred in enumerate(all_preds):
            if pred == 0:
                successful_combos.append(all_combinations[i])
        
        # Rank results
        results = {}
        for i, unknown_key in enumerate(unknowns):
            # Extract the successful values for this unknown key
            successful_values = []
            for combo in successful_combos:
                item = combo[i]
                # Get a readable name
                if isinstance(item, dict):
                    name = item.get('substance_name', 'Unknown Substance')
                else: # It's a condition tuple
                    name = str(item)
                successful_values.append(name)

            results[f"best_{unknown_key}s"] = Counter(successful_values).most_common()

        return results

def format_and_print_results(results, title="Screening Results", top_n=10):
    """Formats and prints screening results in a user-friendly way."""
    print(title)
    print('-' * len(title))

    if not isinstance(results, dict) or not results:
        print("No successful combinations found or an error occurred.")
        return

    for key, values in results.items():
        # E.g., key = "best_probes"
        category_name = key.replace("best_", "").replace("s", "").capitalize()
        print(f"\n--- Top {top_n} {category_name} ---")
        
        if not values:
            print("No items found for this category.")
            continue

        for i, (name, count) in enumerate(values[:top_n]):
            print(f"{i+1}. {name} (found {count} times)")
        
        if len(values) > top_n:
            print(f"... and {len(values) - top_n} more.")

if __name__ == '__main__':
    if Data is None:
        exit()

    # --- 1. Load model and database once ---
    model, config, device = load_model_and_config()
    database = load_database()

    if database is None:
        exit()

    # --- 2. Define substances of interest from the database ---
    # Helper to get a substance by name
    def get_sub(name):
        sub = database['substances_by_name'].get(name)
        if not sub:
            print(f"Warning: Substance '{name}' not found in database.")
        return sub

    # --- 3. Run Inference with different scenarios ---
    print("\n" + "="*50)
    print("SCENARIO 1: SCORING - All inputs provided")
    print("="*50)
    score_result = run_inference(
        model, config, device, database,
        target=get_sub("perfluorooctanesulfonic acid"),
        probe=get_sub("graphene"),
        medium=get_sub("water"),
        condition=(25.0, 7.0, 7.0)
    )
    print(f"Scoring Result: Predicted LDL Label = {score_result['prediction_label']} (0 is best)")


    print("\n" + "="*50)
    # SCENARIO 2: Find best Probes (constrained)
    screening_result_p = run_inference(
        model, config, device, database,
        target=get_sub("perfluorooctanesulfonic acid"),
        medium=get_sub("water"),
        condition=(25.0, 7.0, 7.0)
    )
    format_and_print_results(screening_result_p, title="SCENARIO 2: Best Probes for water at 25C, pH 7")


    print("\n" + "="*50)
    # SCENARIO 3: Find best Targets (Reverse search)
    screening_result_t = run_inference(
        model, config, device, database,
        probe=get_sub("graphene"),
        medium=get_sub("water"),
        condition=(25.0, 7.0, 7.0)
    )
    format_and_print_results(screening_result_t, title="SCENARIO 3: Best Targets for Graphene probe in water at 25C, pH 7")


    print("\n" + "="*50)
    # SCENARIO 4: Find best Medium and Condition
    screening_result_mc = run_inference(
        model, config, device, database,
        target=get_sub("perfluorooctanesulfonic acid"),
        probe=get_sub("graphene")
    )
    format_and_print_results(screening_result_mc, title="SCENARIO 4: Best Mediums & Conditions for Graphene to detect PFOS", top_n=20)

    # --- 4. Additional Lightweight Use Cases ---
    print("\n" + "="*50)
    print("ADDITIONAL USE CASES")
    print("="*50)

    # SCENARIO 5: Compare a few specific probes
    print("\n--- SCENARIO 5: Comparing specific probes for detecting Dopamine ---")
    target_to_test = get_sub("dopamine")
    probes_to_compare = ["graphene", "carbon nanotube", "gold", "silver"]
    for probe_name in probes_to_compare:
        result = run_inference(
            model, config, device, database,
            target=target_to_test,
            probe=get_sub(probe_name),
            medium=get_sub("water"),
            condition=(25.0, 7.0, 7.0)
        )
        print(f"  - Result for probe '{probe_name}': LDL Label = {result['prediction_label']}")

    # SCENARIO 6: Test the effect of pH
    print("\n--- SCENARIO 6: Testing pH effect for a Graphene/PFOS system ---")
    conditions_to_test = [
        (25.0, 4.0, 4.0), # Acidic
        (25.0, 7.0, 7.0), # Neutral
        (25.0, 9.0, 9.0)  # Basic
    ]
    for cond in conditions_to_test:
        result = run_inference(
            model, config, device, database,
            target=get_sub("perfluorooctanesulfonic acid"),
            probe=get_sub("graphene"),
            medium=get_sub("water"),
            condition=cond
        )
        print(f"  - Result for condition {cond}: LDL Label = {result['prediction_label']}")

    # SCENARIO 7: Find a suitable medium for a Target/Probe pair
    print("\n--- SCENARIO 7: Finding the best medium for Dopamine/Carbon Nanotube ---")
    screening_result_m = run_inference(
        model, config, device, database,
        target=get_sub("dopamine"),
        probe=get_sub("carbon nanotube"),
        # medium is None, so we screen for it
        condition=(25.0, 7.0, 7.0)
    )
    format_and_print_results(screening_result_m, title="Best Mediums for Dopamine/Carbon Nanotube at 25C, pH 7")
