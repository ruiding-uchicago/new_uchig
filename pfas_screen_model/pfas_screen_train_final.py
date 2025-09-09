import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import snntorch as snn

# ==== Persisted model/config defaults (added) ====
from pathlib import Path
DEFAULT_SAVE_DIR = "saved-models"
MODEL_FILENAME = "pfas_multimodal.pt"
CONFIG_FILENAME = "config.json"

def save_config(cfg: dict, save_dir: str = DEFAULT_SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, CONFIG_FILENAME), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved to {os.path.join(save_dir, CONFIG_FILENAME)}")
# ==== end added ====


# Category mappings for LDL as integers
ldl_mapping = {
    'Very Low Detection (High Sensitivity)': 0,
    'Low Detection': 1,
    'Moderate Detection': 2,
    'High Detection (Lower Sensitivity)': 3,
    'Very High Detection (Low Sensitivity)': 4
}

def extract_features(descriptors):
    numerical_values = []
    for key, value in descriptors.items():
        if key == "CID":
            continue
        if isinstance(value, (int, float)):
            numerical_values.append(float(value))
        elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
            numerical_values.append(float(value))
    return torch.tensor(numerical_values, dtype=torch.float)

# Step 1: Find maximum feature length across JSON files
def find_max_feature_length(json_files, folder_path):
    max_feature_length = 0
    for f in json_files:
        data = json.load(open(os.path.join(folder_path, f), 'r'))
        for section in ['detect_target', 'probe_material', 'test_medium_electrolyte']:
            for item in data.get(section, []):
                feature_length = len(extract_features(item['substance_descriptors']))
                max_feature_length = max(max_feature_length, feature_length)
    return max_feature_length

# Updated helper function to extract features and include Magpie Descriptors for "inorganic solid"
def extract_features_with_magpie(descriptors, substance_type):
    numerical_values = []
    for key, value in descriptors.items():
        if key == "CID" or (substance_type == "inorganic solid" and "MagpieData" in key):
            continue
        if isinstance(value, (int, float)):
            numerical_values.append(float(value))
        elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
            numerical_values.append(float(value))
    
    # Add Magpie Descriptors for inorganic solids if they exist
    if substance_type == "inorganic solid" and descriptors.get("Magpie Descriptors") is not None:
        magpie_descriptors = [
            float(v) for k, v in descriptors["Magpie Descriptors"].items() if isinstance(v, (int, float))
        ]
        numerical_values.extend(magpie_descriptors)
    
    return torch.tensor(numerical_values, dtype=torch.float)


# Updated aggregation function to use the new extract_features_with_magpie function
def aggregate_by_type(items, max_feature_length):
    types = {'small molecule': [], 'inorganic solid': [], 'polymer': []}
    for item in items:
        substance_type = item.get('substance_type', '').lower()
        if substance_type in types:
            # Extract features, considering Magpie Descriptors if the type is "inorganic solid"
            feature_tensor = F.pad(
                extract_features_with_magpie(item['substance_descriptors'], substance_type),
                (0, max_feature_length - len(extract_features_with_magpie(item['substance_descriptors'], substance_type)))
            )
            types[substance_type].append(feature_tensor)
    
    aggregated_features = []
    for type_key in types:
        if types[type_key]:
            aggregated_features.append(torch.mean(torch.stack(types[type_key]), dim=0))
        else:
            aggregated_features.append(torch.zeros(max_feature_length))
    
    return torch.cat(aggregated_features)

# Generate graph from JSON data (no changes needed here)
def generate_graph_from_json(data, max_feature_length):
    nodes = []
    edges = []

    # Aggregate target, probe, medium nodes with updated aggregation function
    target_node = aggregate_by_type(data.get('detect_target', []), max_feature_length)
    probe_node = aggregate_by_type(data.get('probe_material', []), max_feature_length)
    medium_node = aggregate_by_type(data.get('test_medium_electrolyte', []), max_feature_length)

    # Conditions node
    conditions_features = [
        data.get("test_operating_temperature_celsius", 0.0),
        data.get("min_pH_when_testing", -1.0),
        data.get("max_pH_when_testing", 0.0)
    ]
    conditions_node = F.pad(torch.tensor(conditions_features, dtype=torch.float),
                            (0, max_feature_length * 3 - len(conditions_features)))

    nodes.extend([target_node, probe_node, medium_node, conditions_node])

    edges = [
        (0, 1),  # Target -> Probe
        (0, 2),  # Target -> Medium
        (1, 2),  # Probe -> Medium
        (3, 2)   # Conditions -> Medium
    ]

    x = torch.stack(nodes)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ldl_label = ldl_mapping.get(data.get("LDL_category", ""), 0)
    return Data(x=x, edge_index=edge_index, y=torch.tensor(ldl_label, dtype=torch.long))

# Load and process data
folder_path = 'JSON_data/retrieved_substances_data_enhanced_json_files_2nd_attempt_11052024'
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
max_feature_length = find_max_feature_length(json_files, folder_path)

# The rest of the data loading and processing code remains unchanged
def extract_and_concatenate_vectors(substances):
    vector_data = []
    for substance in substances:
        descriptors = substance.get('substance_descriptors', {})
        for key in ["Morgan_128", "maccs_fp", "morgan_fp_128"]:
            value = descriptors.get(key, [])
            if isinstance(value, list):
                vector_data.extend(value)
    return np.array(vector_data)

snn_data = []
gnn_data = []
labels = []

# Find the maximum vector length across all files
max_vector_length = 0
all_vectors = []

for f in json_files:
    # Load JSON data
    data = json.load(open(os.path.join(folder_path, f), 'r'))

    # Generate graph and append to gnn_data
    graph = generate_graph_from_json(data, max_feature_length)
    if graph is not None:
        gnn_data.append(graph)

        # Generate the spiking vector by concatenating target, probe, and medium
        spiking_vector = np.concatenate([
            extract_and_concatenate_vectors(data.get('detect_target', [])),
            extract_and_concatenate_vectors(data.get('probe_material', [])),
            extract_and_concatenate_vectors(data.get('test_medium_electrolyte', []))
        ])
        all_vectors.append(spiking_vector)

        # Update the maximum vector length if needed
        max_vector_length = max(max_vector_length, len(spiking_vector))
        # Append the label for classification
        labels.append(graph.y.item())

# Pad all_vectors to the maximum vector length
padded_vectors = [np.pad(vec, (0, max_vector_length - len(vec)), 'constant') for vec in all_vectors]

# Append each padded vector directly to snn_data
snn_data.extend(padded_vectors)

snn_data = torch.FloatTensor(snn_data)
labels = torch.LongTensor(labels)

X_train, X_test, y_train, y_test = train_test_split(snn_data, labels, test_size=0.2, random_state=42)
train_data, test_data = train_test_split(gnn_data, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# Define SNN model with customizable beta
class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, beta):
        super(SNN, self).__init__()
        self.beta = beta
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.beta * self.fc2(x)
        return x

# Define GNN model with customizable number of layers
class GNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList([GraphConv(num_features if i == 0 else hidden_channels, hidden_channels) for i in range(num_layers)])
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

class MultiModalModel(nn.Module):
    def __init__(self, snn_input_size, snn_hidden_size, gnn_input_size, gnn_hidden_size, num_classes, snn_beta=0.9, gnn_layers=2):
        super(MultiModalModel, self).__init__()
        self.snn = SNN(snn_input_size, snn_hidden_size, num_classes, beta=snn_beta)
        self.gnn = GNN(gnn_input_size, gnn_hidden_size, num_classes, num_layers=gnn_layers)
        self.fusion_layer = nn.Linear(num_classes * 2, num_classes)

    def forward(self, snn_input, gnn_data):
        snn_output = self.snn(snn_input)
        gnn_output = self.gnn(gnn_data)
        combined_output = torch.cat((snn_output, gnn_output), dim=1)
        final_output = self.fusion_layer(combined_output)
        return final_output

def train_multimodal(snn_loader, gnn_loader):
    model.train()
    total_loss = 0

    for (snn_batch, (gnn_data)) in zip(snn_loader, gnn_loader):
        snn_input, snn_labels = snn_batch
        snn_input, snn_labels = snn_input.to(device), snn_labels.to(device)
        gnn_data = gnn_data.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        output = model(snn_input, gnn_data)
        
        # Calculate loss
        loss = criterion(output, gnn_data.y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(snn_loader)

def test_multimodal(snn_loader, gnn_loader):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    
    with torch.no_grad():
        for (snn_batch, gnn_data) in zip(snn_loader, gnn_loader):
            snn_data, snn_labels = snn_batch
            snn_data, snn_labels = snn_data.to(device), snn_labels.to(device)
            gnn_data = gnn_data.to(device)
            
            # Forward pass through multi-modal model
            output = model(snn_data, gnn_data)
            
            # Compute loss
            loss = criterion(output, gnn_data.y)
            total_loss += loss.item()
            
            # Predictions and labels for metrics
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())  # Save predicted labels
            labels.extend(gnn_data.y.cpu().numpy())  # Save true labels

    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    avg_loss = total_loss / len(snn_loader)
    
    # Return evaluation metrics, predictions, and true labels
    return avg_loss, accuracy, f1, precision, recall, preds, labels

# Save models and their configurations
def save_model(model, file_path, description):
    """
    Save the trained model and print its characteristics.
    """
    # Save the model
    torch.save(model.state_dict(), file_path)
    
    # Print model characteristics
    print(f"Model saved to {file_path}")
    print("Model Description:")
    print(description)
    print("Number of Parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# training loop below
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn_input_size = gnn_data[0].x.shape[1]

model = MultiModalModel(
    snn_input_size=snn_data.shape[1],  # 1358 based on snn_data.shape
    snn_hidden_size=64,
    gnn_input_size=gnn_input_size,     # Updated to match the actual feature size of gnn_data.x
    gnn_hidden_size=64,
    num_classes=5,
    snn_beta=0.85,
    gnn_layers=7
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Create DataLoader for SNN data with matching batch size
snn_train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)
snn_test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=64)

# Training loop
num_epochs = 400
for epoch in range(num_epochs):
    train_loss = train_multimodal(snn_train_loader, train_loader)
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.4f}")

# Final evaluation using test data loaders
test_loss, accuracy, f1, precision, recall, preds, labels = test_multimodal(snn_test_loader, test_loader)
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# ==== Added: persist trained model and its config (robust) ====
try:
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
    _model_path = os.path.join(DEFAULT_SAVE_DIR, MODEL_FILENAME)

    # Derive sizes robustly from the model if not bound in this scope
    try:
        _snn_in = getattr(model.snn.fc1, "in_features", None)
    except Exception:
        _snn_in = None
    try:
        _snn_hid = getattr(model.snn.fc1, "out_features", None)
    except Exception:
        _snn_hid = None
    try:
        _num_classes = getattr(model.snn.fc2, "out_features", None)
        if _num_classes is None and hasattr(model, "fuse"):
            _num_classes = getattr(model.fuse, "out_features", None)
    except Exception:
        _num_classes = None
    try:
        # torch_geometric GraphConv usually exposes in_channels / out_channels
        _gnn_in = None
        if hasattr(model.gnn.convs[0], "in_channels"):
            _gnn_in = int(model.gnn.convs[0].in_channels)
        elif hasattr(model.gnn.convs[0], "lin_rel") and hasattr(model.gnn.convs[0].lin_rel, "in_features"):
            _gnn_in = int(model.gnn.convs[0].lin_rel.in_features)
    except Exception:
        _gnn_in = None
    try:
        _gnn_hid = None
        if hasattr(model.gnn.convs[-1], "out_channels"):
            _gnn_hid = int(model.gnn.convs[-1].out_channels)
    except Exception:
        _gnn_hid = None
    try:
        _gnn_layers = len(model.gnn.convs)
    except Exception:
        _gnn_layers = None
    try:
        _snn_beta = getattr(model.snn, "beta", None)
    except Exception:
        _snn_beta = None

    # Fallbacks if derivation failed
    if _num_classes is None:
        _num_classes = 5
    if _snn_hid is None:
        _snn_hid = 64
    if _gnn_hid is None:
        _gnn_hid = 64
    if _gnn_layers is None:
        _gnn_layers = 7
    if _snn_beta is None:
        _snn_beta = 0.85

    # max_feature_length assumed from gnn_in = 3 * L (target/probe/medium aggregation length)
    try:
        _max_feat_len = int(_gnn_in // 3) if _gnn_in is not None else int(max_feature_length)
    except Exception:
        _max_feat_len = 128  # reasonable default

    # Build config
    _cfg = {
        "arch": "MultiModal(SNN+GNN)",
        "num_classes": int(_num_classes),
        "snn": {
            "input_size": int(_snn_in) if _snn_in is not None else int(getattr(snn_data, "shape", [None, 0])[1]),
            "hidden_size": int(_snn_hid),
            "beta": float(_snn_beta)
        },
        "gnn": {
            "input_size": int(_gnn_in) if _gnn_in is not None else int(3 * _max_feat_len),
            "hidden_size": int(_gnn_hid),
            "layers": int(_gnn_layers)
        },
        "max_feature_length": int(_max_feat_len),
        "positive_class_index": int(0)
    }
    # If a label mapping exists, include it
    try:
        _cfg["ldl_mapping"] = ldl_mapping
    except Exception:
        pass

    # Save weights
    torch.save(model.state_dict(), _model_path)
    print(f"Model saved to {_model_path}")

    # Save config
    save_config(_cfg, DEFAULT_SAVE_DIR)

except Exception as _e:
    print("Warning: failed to save model/config:", _e)
# ==== End added ====
