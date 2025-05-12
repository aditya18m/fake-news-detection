# Fake News Detection on Social Graphs

This project compares classical machine learning models and Graph Convolutional Networks (GCNs) for fake news detection using the **GossipCop** subset of the UPFD dataset. Classical models use handcrafted graph-theoretic features; the GCN learns directly from graph structure.

## Project Structure
```text
project/
├── notebooks/ # Jupyter notebooks
│   ├── EDA_FeatureExtraction.ipynb
│   ├── Classical.ipynb
│   └── GNN.ipynb
├── models/ # Saved trained models
│   ├── random_forest_final_model.joblib
│   ├── logistic_regression_final_model.joblib
│   └── gcn_best_model.pt
├── results/ # Saved feature matrix and labels for ML
│   ├── gossipcop_features_extended_18.npy
│   └── gossipcop_labels.npy
├── data/ # UPFD dataset (not included due to space constraints)
│   └── gossipcop/
```


## Notebooks

- `EDA_FeatureExtraction.ipynb`: Extracts 18 graph features using NetworkX and saves to `results/`
- `Classical.ipynb`: Trains & evaluates Random Forest and Logistic Regression on extracted features
- `GNN.ipynb`: Trains & evaluates a GCN using PyTorch Geometric

## Saved Models

Trained models are saved under `models/`:
- `random_forest_final_model.joblib`
- `logistic_regression_final_model.joblib`
- `gcn_best_model.pt`

## Dataset & Features

- Raw data: Place the UPFD `gossipcop/` subset in `data/`
- Preprocessed features and labels are stored in `results/`
