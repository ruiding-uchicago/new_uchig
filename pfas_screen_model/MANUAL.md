
# User Manual: The `run_inference` Function

## 1. Overview

The `run_inference` function is the core of the `inference_engine.py` script. It serves as a flexible and intelligent interface to the trained multi-modal (SNN+GNN) chemical sensing model.

Its primary purpose is to predict the effectiveness of a chemical detection system. The system is defined by four key components, which we can abbreviate as **C-P-T-M**:

- **T (Target)**: The substance you want to detect.
- **P (Probe)**: The material used to detect the target.
- **M (Medium)**: The electrolyte or medium in which the detection takes place.
- **C (Condition)**: The environmental conditions, specifically temperature and pH.

The function operates in two primary modes, depending on the inputs you provide:

1.  **Scoring Mode**: When you provide all four C-P-T-M components, the function predicts a single effectiveness score (the LDL or Limit of Detection level) for that specific experimental setup.
2.  **Screening Mode**: When you provide only a subset of the C-P-T-M components (e.g., only a Target and a Probe), the function intelligently searches the entire database for the best possible values for the missing components, returning a ranked list of a successful combinations.

## 2. Prerequisites

Before using the `run_inference` function, ensure the following are in place:

1.  **Trained Model**: The `saved-models` directory must exist and contain the `pfas_multimodal.pt` model file and the `config.json` configuration file.
2.  **Substance Database**: The `JSON_data` directory must be populated with the JSON files containing substance descriptors. The engine reads from this to find candidate materials.
3.  **Python Environment**: Your Python environment must have all the necessary libraries installed. You can set this up easily by running:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Function Signature & Parameters

```python
def run_inference(model, config, device, database, target=None, probe=None, medium=None, condition=None):
```

- **`model`** (torch.nn.Module, required): The loaded PyTorch model object, returned by the `load_model_and_config()` helper function.
- **`config`** (dict, required): The loaded model configuration dictionary, also returned by `load_model_and_config()`.
- **`device`** (torch.device, required): The PyTorch device (e.g., 'cpu' or 'cuda') on which the model is running.
- **`database`** (dict, required): The pre-loaded substance database, returned by the `load_database()` helper function.
- **`target`** (dict, optional): A dictionary containing the substance information for the **Target**. Defaults to `None`.
- **`probe`** (dict, optional): A dictionary containing the substance information for the **Probe**. Defaults to `None`.
- **`medium`** (dict, optional): A dictionary containing the substance information for the **Medium**. Defaults to `None`.
- **`condition`** (tuple, optional): A tuple representing the conditions `(temperature, min_pH, max_pH)`. Defaults to `None`.


## 4. Modes of Operation

### 4.1. Scoring Mode

This mode is activated when **all four** optional parameters (`target`, `probe`, `medium`, `condition`) are provided.

- **Purpose**: To evaluate a single, fully-defined experimental setup.
- **Example Call**:
  ```python
  # Assuming model, config, device, database are loaded
  # and get_sub is the helper function to retrieve substance dicts.

  score = run_inference(
      model, config, device, database,
      target=get_sub("perfluorooctanesulfonic acid"),
      probe=get_sub("graphene"),
      medium=get_sub("water"),
      condition=(25.0, 7.0, 7.0)
  )
  # Expected output: {'prediction_label': 0}
  ```
- **Return Value**: A dictionary containing a single key, `prediction_label`. The value is an integer from 0 to 4, representing the predicted LDL level. **A score of `0` is the best possible result**, indicating high detection sensitivity.

### 4.2. Screening Mode

This mode is activated when **one or more** of the optional parameters are omitted (left as `None`).

- **Purpose**: To discover the best unknown components that can successfully be combined with the known components you provide.
- **Example Calls**:

  **A) Find Best Probe (Forward Screening)**
  *Question: "What is the best probe to detect PFOS in water at neutral pH?"*
  ```python
  results = run_inference(
      model, config, device, database,
      target=get_sub("perfluorooctanesulfonic acid"),
      medium=get_sub("water"),
      condition=(25.0, 7.0, 7.0)
      # probe is None
  )
  ```

  **B) Find Best Target (Reverse Screening)**
  *Question: "What targets can a graphene-based sensor in water detect well?"*
  ```python
  results = run_inference(
      model, config, device, database,
      probe=get_sub("graphene"),
      medium=get_sub("water"),
      condition=(25.0, 7.0, 7.0)
      # target is None
  )
  ```

  **C) Find Best Medium and Condition**
  *Question: "I want to use Graphene to detect PFOS. What is the best medium and what are the best conditions?"*
  ```python
  results = run_inference(
      model, config, device, database,
      target=get_sub("perfluorooctanesulfonic acid"),
      probe=get_sub("graphene")
      # medium and condition are None
  )
  ```

- **Return Value**: A dictionary where keys correspond to the unknown parameters. The values are ranked lists of successful items. Each item is a tuple containing the name and the number of times it appeared in a successful combination.

  *Example output for case (C):*
  ```json
  {
    "best_mediums": [
      ["acetate", 108],
      ["oxygen", 106],
      ["water", 106]
    ],
    "best_conditions": [
      ["(25.0, 0.059..., 14.0)", 50],
      ["(25.0, -0.0, 14.0)", 50]
    ]
  }
  ```

## 5. Practical Usage

The `inference_engine.py` script includes a runnable `if __name__ == '__main__':` block that demonstrates all these scenarios. To use `run_inference` in your own code, you should follow a similar structure:

```python
# 1. Import necessary functions and classes from inference_engine.py
# (Or simply use the script as a module)

# 2. Load the model and database once at the start
model, config, device = load_model_and_config()
database = load_database()

# 3. Create a helper function to easily get substance data
def get_sub(name):
    return database['substances_by_name'].get(name)

# 4. Call run_inference with your desired scenario
my_results = run_inference(
    model, config, device, database,
    target=get_sub("dopamine"),
    probe=get_sub("carbon nanotube"),
    condition=(25.0, 7.0, 7.0)
    # Screening for the best medium
)

# 5. Process or print your results
format_and_print_results(my_results, title="My Custom Search")
```

## 6. Performance Considerations

The power of the Screening Mode comes at a computational cost. The number of combinations to test can grow exponentially with the number of unknown parameters.

- **Fast**: Leaving 1 parameter unknown (e.g., finding the best probe) is generally fast.
- **Slow**: Leaving 2 parameters unknown (e.g., finding the best medium and condition) is slower but often manageable.
- **Very Slow**: Leaving 3 or 4 parameters unknown will test millions of combinations and may take an extremely long time or exhaust your system's memory. **This is not recommended for practical use.**

For best results, always provide as much known information as possible to constrain the search space.
