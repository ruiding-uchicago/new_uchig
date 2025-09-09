import os
import json
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader
except Exception:
    from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool

# ---- Defaults to match your environment ----
DEFAULT_MODEL_DIR = "saved-models"
MODEL_FILENAME = "pfas_multimodal.pt"
CONFIG_FILENAME = "config.json"

ldl_mapping = {
    'Very Low Detection (High Sensitivity)': 0,
    'Low Detection': 1,
    'Medium Detection': 2,
    'High Detection': 3,
    'Very High Detection': 4
}

def extract_features_with_magpie(descriptors: dict, substance_type: str) -> torch.Tensor:
    numerical_values = []
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

    cond_feats = [
        data.get("test_operating_temperature_celsius", 0.0),
        data.get("min_pH_when_testing", -1.0),
        data.get("max_pH_when_testing", 0.0),
    ]
    cond = torch.tensor(cond_feats, dtype=torch.float)
    need = 3*max_feature_length - cond.numel()
    if need > 0:
        cond = F.pad(cond, (0, need))
    else:
        cond = cond[:3*max_feature_length]

    x = torch.stack([target_node, probe_node, medium_node, cond])
    edges = [(0, 1), (0, 2), (1, 2), (3, 2)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    ldl_label = ldl_mapping.get(data.get("LDL_category", ""), 0)
    return Data(x=x, edge_index=edge_index, y=torch.tensor(ldl_label, dtype=torch.long))

def extract_and_concatenate_vectors(substances: list) -> np.ndarray:
    vector_data = []
    for substance in substances:
        descriptors = substance.get('substance_descriptors', {})
        for key in ["Morgan_128", "maccs_fp", "morgan_fp_128"]:
            value = descriptors.get(key, [])
            if isinstance(value, list):
                vector_data.extend(value)
    return np.array(vector_data, dtype=float)

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
        # Match training: 'fusion_layer'
        self.fusion_layer = nn.Linear(num_classes * 2, num_classes)
    def forward(self, snn_batch, gnn_batch):
        snn_logits = self.snn(snn_batch)
        gnn_logits = self.gnn(gnn_batch)
        fused = torch.cat([snn_logits, gnn_logits], dim=-1)
        return self.fusion_layer(fused)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfas_file", type=str, required=True,
                        help="Path to PFAS detect_target JSON (e.g. PFAS_Screening/PFAS_detect_target_6.json)")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing saved model + config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config + model
    cfg_path = os.path.join(args.model_dir, CONFIG_FILENAME)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    snn_input_size = int(cfg["snn"]["input_size"])
    gnn_input_size = int(cfg["gnn"]["input_size"])
    max_feature_length = int(cfg["max_feature_length"])
    num_classes = int(cfg["num_classes"])
    snn_hidden = int(cfg["snn"]["hidden_size"])
    gnn_hidden = int(cfg["gnn"]["hidden_size"])
    snn_beta = float(cfg["snn"]["beta"])
    gnn_layers = int(cfg["gnn"]["layers"])
    positive_idx = int(cfg.get("positive_class_index", 0))

    model = MultiModalModel(
        snn_input_size=snn_input_size,
        snn_hidden_size=snn_hidden,
        gnn_input_size=gnn_input_size,
        gnn_hidden_size=gnn_hidden,
        num_classes=num_classes,
        snn_beta=snn_beta,
        gnn_layers=gnn_layers
    ).to(device)

    # Safe state load
    state_path = os.path.join(args.model_dir, MODEL_FILENAME)
    try:
        state = torch.load(state_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location=device)
    # Remap if necessary
    if "fusion_layer.weight" not in state and "fuse.weight" in state:
        state["fusion_layer.weight"] = state.pop("fuse.weight")
        state["fusion_layer.bias"] = state.pop("fuse.bias")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Load PFAS detect_target to screen against
    with open(args.pfas_file, 'r') as pfas_file:
        pfas_data = json.load(pfas_file)
    fixed_detect_target = pfas_data.get('detect_target', [])

    folder_path = 'JSON_data/retrieved_substances_data_enhanced_json_files_2nd_attempt_11052024'
    json_files_org = [file for file in os.listdir(folder_path) if file.endswith('_original.json')]

    snn_vectors, gnn_graphs = [], []
    for f in json_files_org:
        file_path = os.path.join(folder_path, f)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        # Inject fixed PFAS target
        data['detect_target'] = fixed_detect_target

        # SNN vector
        vec = np.concatenate([
            extract_and_concatenate_vectors(data.get('detect_target', [])),
            extract_and_concatenate_vectors(data.get('probe_material', [])),
            extract_and_concatenate_vectors(data.get('test_medium_electrolyte', []))
        ]).astype(float)

        if vec.shape[0] < snn_input_size:
            vec = np.pad(vec, (0, snn_input_size - vec.shape[0]), 'constant')
        else:
            vec = vec[:snn_input_size]
        snn_vectors.append(vec)

        # GNN graph (uses training max_feature_length for padding)
        graph = generate_graph_from_json(data, max_feature_length)
        if graph is not None:
            gnn_graphs.append(graph)
        else:
            snn_vectors.pop()

    # Build loaders
    snn_tensor = torch.tensor(np.stack(snn_vectors, axis=0), dtype=torch.float)
    snn_loader = DataLoader(list(zip(snn_tensor, torch.zeros(len(snn_vectors), dtype=torch.long))), batch_size=64)
    gnn_loader = DataLoader(gnn_graphs, batch_size=64)

    # Predict
    preds = []
    with torch.no_grad():
        for (snn_batch, _), gnn_batch in zip(snn_loader, gnn_loader):
            snn_batch = snn_batch.to(device)
            gnn_batch = gnn_batch.to(device)
            out = model(snn_batch, gnn_batch)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().tolist())

    # Collect details for predicted positives
    zero_pred_details = []
    for idx, pred in enumerate(preds):
        if pred == positive_idx:
            file_name = json_files_org[idx]
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as jf:
                jd = json.load(jf)

            probe_material = jd.get('probe_material', [])
            probe_material_names = [item.get('substance_name', 'Unknown') for item in probe_material]

            conditions_vector = [
                jd.get("test_operating_temperature_celsius", 0.0),
                jd.get("min_pH_when_testing", -1.0),
                jd.get("max_pH_when_testing", 0.0)
            ]

            zero_pred_details.append({
                "file_name": file_name,
                "probe_material_names": probe_material_names,
                "conditions_vector": conditions_vector
            })

    # ---- Match original-style output ----
    total_zero = len(zero_pred_details)
    print(f"Number of predictions == 0: {total_zero}")

    # Minimal "filter": drop entries with no probe names and dedupe by file_name
    filtered_zero_pred_details = []
    seen_files = set()
    for d in zero_pred_details:
        if not d['probe_material_names']:
            continue
        if d['file_name'] in seen_files:
            continue
        seen_files.add(d['file_name'])
        filtered_zero_pred_details.append(d)

    print(f"Size of filtered dataset: {len(filtered_zero_pred_details)}")

    # Recurrent probe materials (filtered)
    filtered_probe_names = [name for d in filtered_zero_pred_details for name in d['probe_material_names'] if name and name.lower() != 'unknown']
    filtered_name_counts = Counter(filtered_probe_names)
    sorted_filtered_name_counts = sorted(filtered_name_counts.items(), key=lambda x: x[1], reverse=True)

    print("Recurrent Probe Material Names (Filtered) (Decreasing Order):")
    for name, count in sorted_filtered_name_counts:
        print(f"{name}: {count} occurrences")

    # Recurrent conditions vectors (filtered)
    filtered_conditions = [tuple(d['conditions_vector']) for d in filtered_zero_pred_details]
    filtered_conditions_counts = Counter(filtered_conditions)
    sorted_filtered_conditions_counts = sorted(filtered_conditions_counts.items(), key=lambda x: x[1], reverse=True)

    print("Recurrent Conditions Vectors (Filtered) (Decreasing Order):")
    for conditions, count in sorted_filtered_conditions_counts:
        print(f"{conditions}: {count} occurrences")

if __name__ == "__main__":
    main()
