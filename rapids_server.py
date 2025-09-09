#!/usr/bin/env python3
"""
RAPIDS Web Server - Minimal Flask backend for RAPIDS molecular simulation GUI
Provides API endpoints for running single and batch simulations via web interface
"""

from flask import Flask, request, Response, jsonify, send_file, stream_with_context
from flask_cors import CORS
import subprocess
import json
import os
import tempfile
import threading
import queue
import time
import uuid
from pathlib import Path
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
RAPIDS_PATH = os.environ.get('RAPIDS_PATH', '/path/to/auto_CPT_uma_simul')
SIMULATIONS_DIR = Path('simulations_web')
SIMULATIONS_DIR.mkdir(exist_ok=True)

# Active simulations tracking
active_simulations = {}

@app.route('/')
def index():
    """Serve the main HTML interface"""
    html_path = 'myportal/templates/globus-portal-framework/v2/RAPIDS_Simulator.html'
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            return f.read()
    else:
        return "RAPIDS_Simulator.html not found. Please ensure it's in the correct location.", 404

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run a single molecule simulation with real-time output streaming"""
    config = request.json
    sim_id = str(uuid.uuid4())
    
    def generate():
        try:
            # Create temporary directory for this simulation
            sim_dir = SIMULATIONS_DIR / sim_id
            sim_dir.mkdir(exist_ok=True)
            
            # Write configuration file
            config_file = sim_dir / 'config.json'
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            yield f"data: Starting simulation {sim_id}\n\n"
            yield f"data: Configuration saved to {config_file}\n\n"
            
            # Prepare command
            cmd = [
                sys.executable,
                os.path.join(RAPIDS_PATH, 'smart_fairchem_flow.py'),
                str(config_file)
            ]
            
            yield f"data: Running command: {' '.join(cmd)}\n\n"
            
            # Run RAPIDS simulation
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=RAPIDS_PATH
            )
            
            # Store process for potential cancellation
            active_simulations[sim_id] = process
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Parse progress if present
                    if 'Step' in line or 'Energy' in line or '%' in line:
                        yield f"data: {line.strip()}\n\n"
                    else:
                        yield f"data: {line.strip()}\n\n"
            
            process.wait()
            
            # Check for completion
            if process.returncode == 0:
                yield f"data: Simulation completed successfully!\n\n"
                
                # Read results if available
                results_file = sim_dir / 'simulations' / config.get('run_name', 'default') / 'interactions.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    yield f"data: RESULTS:{json.dumps(results)}\n\n"
            else:
                yield f"data: ERROR: Simulation failed with code {process.returncode}\n\n"
                
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
        finally:
            # Clean up
            if sim_id in active_simulations:
                del active_simulations[sim_id]
            yield f"data: COMPLETE\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/batch', methods=['POST'])
def batch_simulate():
    """Run batch screening for multiple molecules"""
    data = request.json
    molecules = data.get('molecules', [])
    target = data.get('target', None)
    substrate = data.get('substrate', 'vacuum')
    
    batch_id = str(uuid.uuid4())
    
    def generate():
        try:
            # Create batch configuration
            batch_config = {
                "probes": molecules,
                "substrate": substrate,
                "comparison_name": f"batch_{batch_id}"
            }
            
            if target:
                batch_config["target"] = target
            
            # Create temporary directory
            batch_dir = SIMULATIONS_DIR / f"batch_{batch_id}"
            batch_dir.mkdir(exist_ok=True)
            
            # Write batch config
            config_file = batch_dir / 'batch_config.json'
            with open(config_file, 'w') as f:
                json.dump(batch_config, f, indent=2)
            
            yield f"data: Starting batch screening with {len(molecules)} molecules\n\n"
            
            # Run batch comparison
            cmd = [
                sys.executable,
                os.path.join(RAPIDS_PATH, 'batch_comparison.py'),
                str(config_file)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=RAPIDS_PATH
            )
            
            active_simulations[batch_id] = process
            
            # Stream output
            current_molecule = None
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Parse which molecule is being processed
                    if 'Processing' in line or 'Simulating' in line:
                        for mol in molecules:
                            if mol in line:
                                current_molecule = mol
                                yield f"data: MOLECULE:{mol}\n\n"
                                break
                    
                    yield f"data: {line.strip()}\n\n"
            
            process.wait()
            
            if process.returncode == 0:
                # Read comparison results
                results_file = batch_dir / 'simulations' / batch_config['comparison_name'] / 'comparison_results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    yield f"data: RESULTS:{json.dumps(results)}\n\n"
                
                yield f"data: Batch screening completed!\n\n"
            else:
                yield f"data: ERROR: Batch screening failed\n\n"
                
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
        finally:
            if batch_id in active_simulations:
                del active_simulations[batch_id]
            yield f"data: COMPLETE\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop a running simulation"""
    sim_id = request.json.get('simulation_id')
    
    if sim_id in active_simulations:
        process = active_simulations[sim_id]
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            process.kill()
        del active_simulations[sim_id]
        return jsonify({"status": "stopped", "simulation_id": sim_id})
    else:
        return jsonify({"error": "Simulation not found"}), 404

@app.route('/api/molecule/search', methods=['GET'])
def search_molecule():
    """Search for molecule information from PubChem"""
    name = request.args.get('name')
    
    try:
        import pubchempy as pcp
        
        # Search PubChem
        compounds = pcp.get_compounds(name, 'name')
        
        if compounds:
            compound = compounds[0]
            return jsonify({
                "cid": compound.cid,
                "smiles": compound.canonical_smiles,
                "formula": compound.molecular_formula,
                "weight": compound.molecular_weight,
                "iupac_name": compound.iupac_name
            })
        else:
            return jsonify({"error": "Molecule not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download/<sim_id>/<file_type>')
def download_results(sim_id, file_type):
    """Download simulation results"""
    sim_dir = SIMULATIONS_DIR / sim_id
    
    if file_type == 'json':
        file_path = sim_dir / 'simulations' / 'default' / 'interactions.json'
    elif file_type == 'vasp':
        file_path = sim_dir / 'simulations' / 'default' / 'optimized.vasp'
    elif file_type == 'report':
        file_path = sim_dir / 'simulations' / 'default' / 'smart_report.txt'
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

@app.route('/api/status')
def status():
    """Get server status and active simulations"""
    return jsonify({
        "status": "running",
        "active_simulations": list(active_simulations.keys()),
        "rapids_path": RAPIDS_PATH,
        "simulations_dir": str(SIMULATIONS_DIR)
    })

@app.route('/api/substrates')
def get_substrates():
    """Get available substrates"""
    substrates = [
        {"name": "vacuum", "display": "Vacuum", "description": "No substrate"},
        {"name": "Graphene", "display": "Graphene", "description": "2D carbon sheet"},
        {"name": "Co_HHTP", "display": "Co-HHTP", "description": "Metal-organic framework"},
        {"name": "Cu_BTC", "display": "Cu-BTC", "description": "Metal-organic framework"},
        {"name": "MoS2", "display": "MoS₂", "description": "2D transition metal dichalcogenide"},
        {"name": "h-BN", "display": "h-BN", "description": "Hexagonal boron nitride"}
    ]
    return jsonify(substrates)

@app.route('/api/examples')
def get_examples():
    """Get example configurations"""
    examples = [
        {
            "name": "Glucose on Graphene",
            "config": {
                "probe": "glucose",
                "substrate": "Graphene",
                "fmax": 0.05,
                "max_steps": 100
            }
        },
        {
            "name": "PFAS Detection",
            "config": {
                "probe": "PFOA",
                "target": "antibody",
                "substrate": "vacuum",
                "fmax": 0.03,
                "max_steps": 200
            }
        },
        {
            "name": "Drug-Protein Interaction",
            "config": {
                "probe": "aspirin",
                "target": "COX-2",
                "substrate": "vacuum",
                "fmax": 0.05,
                "max_steps": 150
            }
        }
    ]
    return jsonify(examples)

@app.route('/api/validate', methods=['POST'])
def validate_config():
    """Validate simulation configuration before running"""
    config = request.json
    
    errors = []
    warnings = []
    
    # Check required fields
    if not config.get('probe'):
        errors.append("Probe molecule is required")
    
    # Check numerical parameters
    fmax = config.get('fmax', 0.05)
    if fmax <= 0 or fmax > 1:
        errors.append("Force convergence (fmax) should be between 0.01 and 1.0")
    elif fmax > 0.1:
        warnings.append("High fmax value may lead to poor convergence")
    
    max_steps = config.get('max_steps', 100)
    if max_steps < 10:
        errors.append("Maximum steps should be at least 10")
    elif max_steps > 1000:
        warnings.append("Large max_steps may take a long time")
    
    # Check device availability
    device = config.get('device', 'cpu')
    if device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA device requested but not available")
        except ImportError:
            errors.append("PyTorch not installed, cannot check CUDA availability")
    
    return jsonify({
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    })

if __name__ == '__main__':
    print(f"RAPIDS Web Server starting...")
    print(f"RAPIDS path: {RAPIDS_PATH}")
    print(f"Simulations directory: {SIMULATIONS_DIR}")
    
    # Check if RAPIDS is accessible
    if not os.path.exists(RAPIDS_PATH):
        print(f"WARNING: RAPIDS path not found at {RAPIDS_PATH}")
        print("Please set RAPIDS_PATH environment variable to the location of auto_CPT_uma_simul")
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)