# ByteSEM: A Semantically-Enhanced Hybrid Graph Neural Network for Smart Contract Bytecode Clone Detection- Code Documentation


## Main Entry Points

### 1. Model Training Entry - `train.py`

**Purpose**: Core training script for Siamese GNN models for smart contract clone detection

**Complete Command Examples**:
```bash
# Train hierarchical GNN model
python train.py --model_type hierarchical --epochs 20 --batch_size 32 --lr 0.001 --margin 1.0 --model_path models/hierarchical

# Train flat GNN model (for ablation study)
python train.py --model_type base --epochs 20 --batch_size 32 --lr 0.001 --margin 1.0 --model_path models/base
```

**Detailed Parameter Specifications**:
- `--model_type` (str, default='base'): Model architecture selection
  - `'hierarchical'`: Hierarchical GNN model with gated fusion of hierarchical and flat features
  - `'base'`: Flat GNN model for ablation experiments
- `--processed_dir` (str, default='GNNdata/noAbstract_ptdata'): Path to preprocessed graph data directory
- `--train_csv` (str, default='train.csv'): Training set CSV file path
- `--val_csv` (str, default='val.csv'): Validation set CSV file path
- `--epochs` (int, default=20): Number of training epochs
- `--batch_size` (int, default=32): Batch size for training
- `--lr` (float, default=0.001): Learning rate for Adam optimizer
- `--margin` (float, default=1.0): Margin value for ContrastiveLoss function
- `--model_path` (str, default='noAbstractbasemodel'): Directory to save trained models

**Training Pipeline**:
1. Automatically checks CFG file directory and filters valid contract IDs
2. Loads training and validation datasets using PairedGraphDataset
3. Initializes model with fixed dimensions (output_dim=128, gnn_hidden_dim=128)
4. Uses Adam optimizer and ContrastiveLoss for training
5. Saves model after each epoch to `{model_path}/{epoch}.pth`
6. Saves best model to `{model_path}/best_model.pth`
7. Generates training log `training_log.json`

**Output Files**:
- `{model_path}/best_model.pth`: Best model weights based on validation loss
- `{model_path}/{epoch}.pth`: Model weights for each training epoch
- `training_log.json`: Training and validation loss curves data

**Key Implementation Details**:
- Fixed random seed (42) for reproducibility
- Automatic GPU/CPU device selection
- Data filtering based on available CFG files in `GNNdata/noAbstact_proccessed_cfg`
- Hierarchical batch processing with `collate_fn_hierarchical`

### 2. Model Evaluation Entry - `evaluation.py`

**Purpose**: Comprehensive evaluation of trained models with detailed performance reports and publication-quality visualizations

**Complete Command Examples**:
```bash
# Standard evaluation mode
python evaluation.py --model_type hierarchical --model_path models/hierarchical/best_model.pth --test_csv test.csv --output_dir results/hierarchical_eval

# Robustness testing mode
python evaluation.py --model_type hierarchical --model_path models/hierarchical/best_model.pth --test_csv test.csv --robust True --output_dir results/robustness_test
```

**Detailed Parameter Specifications**:
- `--model_type` (str, default='hierarchical'): Model architecture type
- `--model_path` (str, default='noAbstractmodel/best_model.pth'): Path to trained model file
- `--processed_dir` (str, default='GNNdata/noABstract_ptdata'): Preprocessed graph data directory
- `--train_csv` (str, default='train.csv'): Training set CSV file (for dataset distribution plots)
- `--val_csv` (str, default='val.csv'): Validation set CSV file (for dataset distribution plots)
- `--test_csv` (str, default='test.csv'): Test set CSV file
- `--log_path` (str, default='training_log.json'): Training log file path
- `--batch_size` (int, default=128): Batch size for evaluation
- `--robust` (bool, default=False): Enable robustness testing mode
- `--output_dir` (str, default='noAbstracthierReport'): Output directory for results

**Evaluation Capabilities**:

**Standard Evaluation Mode**:
- Performs predictions on test set using trained model
- Uses Youden's J statistic to find optimal classification threshold
- Calculates overall performance metrics (Precision, Recall, F1-Score)
- Computes independent recall rates for each clone type
- Generates publication-quality visualizations (300 DPI)

**Robustness Testing Mode**:
- Compares performance on original vs. noisy data
- Analyzes embedding vector stability
- Calculates robustness scores using cosine similarity
- Generates detailed robustness analysis reports

**Output Files**:
- `performance.txt`: Detailed performance metrics report
- `confusion_matrix_publication.png`: Confusion matrix visualization (300 DPI)
- `confusion_matrix.txt`: Confusion matrix numerical values
- `roc_curve.png`: ROC curve with AUC calculation
- `distance_distribution.svg`: Distance distribution plots
- `distance_by_type_distribution.svg`: Distance distribution by clone type
- `per_type_metrics.png`: Performance metrics per clone type
- `dataset_distribution.png`: Dataset type distribution pie chart
- `loss_curve_publication.png`: Training loss curve (if log file exists)
- `robustness_detailed_results.csv`: Detailed robustness test results (robustness mode)
- `robustness_score_distribution.png`: Robustness score distribution (robustness mode)
- `robustness_distance_shift.png`: Distance shift scatter plot (robustness mode)

**Key Implementation Details**:
- Automatic CFG file checking in `GNNdata/proccessed_cfg`
- Publication-quality figure generation with Times New Roman font
- Support for both standard and robustness evaluation modes
- Automatic threshold optimization using ROC analysis

## Supporting Modules

### Data Preprocessing Module

#### `preprocess_graphs.py` - Graph Data Preprocessing
**Function**: Converts DOT format control flow graphs and JSON function mappings to PyTorch Geometric Data objects
**Purpose**: Prepares standardized graph data format for GNN models
**Input**: DOT files (control flow graphs) + JSON files (function mappings)
**Output**: .pt files (PyTorch tensor format graph data)

#### `tool.py` - CFG Generation Tool
**Function**: Generates control flow graphs (CFG) from smart contract bytecode
**Purpose**: Converts raw bytecode to structured graph representation
**Core Class**: `ContractCFGGenerator`
**Output**: DOT files (control flow graphs) + JSON files (function-basic block mappings)

### Model Architecture Module

#### `model.py` - Model Definitions
**Function**: Defines hierarchical and flat GNN model architectures
**Core Classes**:
- `HierarchicalGNN`: Hierarchical graph neural network with gated fusion of hierarchical and flat features
- `FlatGNN`: Flat graph neural network for ablation experiments
**Features**: Uses GAT (Graph Attention Network) for node feature learning, supports function-level pooling

#### `dataset.py` - Dataset Processing
**Function**: Defines paired graph dataset and batch processing functions
**Core Classes**:
- `PairedGraphDataset`: Paired graph dataset class
- `collate_fn_hierarchical`: Hierarchical batch processing function
**Purpose**: Provides paired graph data for Siamese networks

### Encoder Module

#### `encoder/train.py` - Encoder Training
**Function**: Trains sequence-to-sequence autoencoder for node feature encoding
**Purpose**: Encodes opcode sequences into fixed-dimensional vector representations
**Features**: Bidirectional LSTM encoder + unidirectional LSTM decoder, supports teacher forcing

#### `encoder/prepare_data.py` - Encoder Data Preparation
**Function**: Prepares training data for encoder
**Core Classes**:
- `Vocabulary`: Vocabulary management class
- `OpcodeDataset`: Opcode dataset class
- `PadCollate`: Sequence padding batch processing class
**Purpose**: Handles variable-length sequences, builds vocabulary

#### `encoder/model.py` - Encoder Model Definitions
**Function**: Defines sequence-to-sequence autoencoder models
**Core Classes**:
- `Encoder`: Bidirectional LSTM encoder
- `Decoder`: Unidirectional LSTM decoder
- `Seq2SeqAutoencoder`: Complete autoencoder model

#### `encoder/evaluate.py` - Encoder Evaluation
**Function**: Evaluates encoder performance and conducts ablation experiments
**Purpose**: Compares semantic abstraction encoder vs. raw token encoder performance

### Auxiliary Tools Module

#### `run_robotness_test.py` - Robustness Testing
**Function**: Tests model robustness against structural noise
**Purpose**: Adds orphan block noise to graphs and analyzes model stability
**Features**: Supports multiple noise types, generates detailed robustness analysis reports

#### `tools_label/tool.py` - Baseline Tool Evaluation
**Function**: Evaluates performance of traditional clone detection tools
**Supported Tools**: Deckard, EClone, Nicad, SmartEmbed, SourcererCC
**Purpose**: Performance comparison with baseline methods

#### `tools_label/combine_tool.py` - Tool Combination Evaluation
**Function**: Evaluates performance of multiple tool combinations
**Purpose**: Analyzes effectiveness of tool combination strategies

#### `encoderTrainer.py` - Encoder Trainer
**Function**: Simplified encoder training script
**Purpose**: Quick training of encoder models

#### `check.py` - Data Validation Tool
**Function**: Checks basic statistical information of datasets
**Purpose**: Validates dataset completeness and distribution

#### `encoder/check.py` - Encoder Validation Tool
**Function**: Validates encoder-related data and models
**Purpose**: Verifies encoder model files, vocabulary files, and encoder functionality

## Complete Usage Workflow

### 1. Data Preparation Phase
```bash
# 1. Generate CFG from bytecode
python tool.py

# 2. Preprocess graph data
python preprocess_graphs.py --dot_dir GNNdata/cfg --json_dir GNNdata/function --output_dir GNNdata/ptdata

# 3. Train encoder
cd encoder
python train.py
cd ..
```

### 2. Model Training Phase
```bash
# Train hierarchical GNN model
python train.py --model_type hierarchical --epochs 20 --batch_size 32 --lr 0.001 --model_path models/hierarchical

# Train flat GNN model (ablation study)
python train.py --model_type base --epochs 20 --batch_size 32 --lr 0.001 --model_path models/base
```

### 3. Model Evaluation Phase
```bash
# Standard evaluation
python evaluation.py --model_type hierarchical --model_path models/hierarchical/best_model.pth --test_csv test.csv --output_dir results/hierarchical

# Robustness testing
python evaluation.py --model_type hierarchical --model_path models/hierarchical/best_model.pth --test_csv test.csv --robust True --output_dir results/robustness
```

### 4. Baseline Comparison Phase
```bash
# Evaluate traditional tools
python tools_label/tool.py

# Evaluate tool combinations
python tools_label/combine_tool.py
```

## Important Notes

1. **Path Consistency**: Ensure consistent data paths and model types between training and evaluation
2. **Dependencies**: Requires PyTorch, PyTorch Geometric, NetworkX, and other libraries
3. **Data Format**: CSV files must contain contract_id, clone_contract_id, groundtruth, type columns
4. **GPU Memory**: Monitor GPU memory usage when training large models
5. **Random Seeds**: Fixed random seed (42) ensures reproducible results

## File Dependencies

```
train.py -> model.py, dataset.py
evaluation.py -> model.py, dataset.py, train.py
preprocess_graphs.py -> encoder/evaluate.py
encoder/train.py -> encoder/model.py, encoder/prepare_data.py
run_robustness_test.py -> model.py, dataset.py, preprocess_graphs.py
```

## Output File Types

- `*.pth`: Trained model files
- `*.pt`: Preprocessed graph data files
- `*.csv`: Dataset files
- `*.png/svg`: Generated visualization files (300 DPI publication quality)
- `*.txt`: Evaluation report files
- `*.json`: Configuration and data mapping files

## train.py vs evaluation.py Detailed Comparison

### Parameter Comparison Table

| Parameter | train.py | evaluation.py | Notes |
|-----------|----------|---------------|-------|
| `--model_type` | default='base' | default='hierarchical' | Different default model types |
| `--processed_dir` | default='GNNdata/noAbstract_ptdata' | default='GNNdata/noABstract_ptdata' | Note spelling differences |
| `--train_csv` | default='train.csv' | default='train.csv' | Same |
| `--val_csv` | default='val.csv' | default='val.csv' | Same |
| `--test_csv` | Not available | default='test.csv' | Training doesn't need test set |
| `--batch_size` | default=32 | default=128 | Larger batch size for evaluation |
| `--lr` | default=0.001 | Not available | Evaluation doesn't need learning rate |
| `--margin` | default=1.0 | Not available | Evaluation doesn't need loss function parameters |
| `--model_path` | default='noAbstractbasemodel' | default='noAbstractmodel/best_model.pth' | Training save path vs evaluation load path |
| `--epochs` | default=20 | Not available | Evaluation doesn't need training epochs |
| `--log_path` | Not available | default='training_log.json' | Evaluation needs to read training logs |
| `--robust` | Not available | default=False | Only evaluation script supports robustness testing |
| `--output_dir` | Not available | default='noAbstracthierReport' | Only evaluation script needs output directory |

### Key Differences

1. **Default Model Types**:
   - `train.py` defaults to 'base' model
   - `evaluation.py` defaults to 'hierarchical' model

2. **Data Directory Spelling**:
   - `train.py`: 'noAbstract_ptdata'
   - `evaluation.py`: 'noABstract_ptdata' (note case differences)

3. **CFG Check Directories**:
   - `train.py`: checks 'GNNdata/noAbstact_proccessed_cfg'
   - `evaluation.py`: checks 'GNNdata/proccessed_cfg'

4. **Functional Focus**:
   - `train.py`: Focuses on model training, saves models and logs
   - `evaluation.py`: Focuses on model evaluation, generates reports and visualizations

### Usage Recommendations

1. **Training**: Use `train.py` for model training
2. **Evaluation**: Use `evaluation.py` for model evaluation, ensure parameter path consistency
3. **Path Attention**: Default paths differ slightly between scripts, maintain consistency during use
4. **Model Matching**: Ensure model type used in evaluation matches the training type


Related data in https://drive.google.com/drive/folders/1IkyD3pVcEPuPZVmpvyOL8MKvsTYve6cg
