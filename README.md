# FP-GINE-MT: Multitask Molecular Property Prediction Model Based on GINE and Hybrid Fingerprints

## Project Overview

FP-GINE-MT is a multitask learning model based on Graph Neural Networks (GINE) and hybrid molecular fingerprints for predicting multiple thermodynamic properties of molecules. This model combines the structural perception capabilities of graph neural networks with the feature representation advantages of traditional molecular fingerprints to achieve efficient and accurate multi-task prediction.

### Key Features

- Multi-task learning framework for simultaneous prediction of multiple molecular properties
- Integration of GINE (Graph Isomorphism Network with Edge features) and hybrid molecular fingerprints
- Support for cross-validation and hyperparameter optimization
- Pre-trained models and data preprocessing tools

## Environment Requirements

- Python 3.6+
- PyTorch
- PyTorch Geometric
- RDKit
- scikit-learn
- pandas
- numpy
- tqdm
- optuna
- joblib

## Installation Guide

1. Clone the project locally
2. Install required dependencies:
```bash
pip install torch torch-geometric rdkit scikit-learn pandas numpy tqdm optuna joblib
```

## Data Preparation

The dataset should include the following columns:
- smiles: SMILES representation of molecules
- val: Target property values
- prop: Property type identifier

Data file path is configured in the `CSV_PATH` variable in the code.

## Model Architecture

### 1. Molecular Representation
- GINE Layer: Captures molecular graph structure information
- Hybrid Fingerprints: Combines MACCS and PubChem fingerprints
- Cross-modal Interaction: Implements complementary enhancement of graph representation and fingerprint features

### 2. Multi-task Prediction
- Task Selector: Dynamically selects current prediction task
- Fusion Layer: Integrates multi-modal features
- Prediction Head: Outputs target property values

## Usage Instructions

### Model Training

1. Prepare the dataset
2. Configure hyperparameters (in the constants section at the beginning of the code)
3. Run the training script:
```python
python FP-GINE-MT.ipynb
```

### Main Parameters

- TEST_SPLIT_RATIO: Test set ratio
- N_FOLDS: Number of cross-validation folds
- N_TRIALS_OPTUNA: Number of Optuna optimization trials
- N_EPOCHS_FOLD_TRAINING: Number of epochs per fold training
- PATIENCE_FOLD_TRAINING: Early stopping patience value

## Model Output

The training process will generate:
- Pre-trained model files (.pth)
- Data preprocessors (scalers)
- Pre-generated molecular representation data
- Test results file

All output files are saved in the directory specified by `BEST_MODEL_SAVE_DIR`.

## Performance Evaluation

The model evaluates performance using the following metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coefficient of Determination)

## Notes

- Ensure correct input data format
- Monitor GPU memory usage during training
- Model architecture and hyperparameters can be adjusted as needed
