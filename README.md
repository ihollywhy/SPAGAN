## SPAGAN in PyTorch

This is a PyTorch implementation of the paper "SPAGAN: Shortest Path Graph Attention Network"

### Prerequisites

We prefer to create a new conda environment to run the code.

#### PyTorch
Version >= 1.0.0

#### PyTorch-geometric
We use [torch_geometric](https://github.com/rusty1s/pytorch_geometric), [torch_scatter] and [torch_sparse] as backbone to implement the path attention mechanism. Please follow the [official website](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html) to install them.

#### networkx 
We use [networkx](https://networkx.github.io/) to load the graph dataset.

#### graph-tool
We use graph-tool for fast APSP calculation. Please follow the [official website](https://graph-tool.skewed.de/) to install.

### Run the code:
Activating the corresponding conda env if you use a new conda environment.

```
python train_spgat.py
```