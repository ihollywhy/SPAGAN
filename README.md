## SPAGAN in PyTorch

This is a PyTorch implementation of the paper "SPAGAN: Shortest Path Graph Attention Network"

### Prerequisites

We prefer to create a new conda enviroment to run the code.

#### PyTorch

    version > 1.0.0

#### PyTorch-geometric

    We use [torch_geometric](https://github.com/rusty1s/pytorch_geometric), [torch_scatter] and [torch_sparse] as backbone to implement the path attention mechanism. Please follow the [official website](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html) to install them.

#### networkx 

    We use [networkx](https://networkx.github.io/) to load the graph dataset.

#### graph-tool

    We use graph-tool for fast apsp calculation. Please follow the [official website](https://graph-tool.skewed.de/) to install.

### Run the code:

    activate the corresponding conda env if you use a new conda enviroment

    python train_spgat.py