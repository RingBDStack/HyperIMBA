# HyperIMBA
Code for "Hyperbolic Geometric Graph Representation Learning for Hierarchy-imbalance Node Classification".

## Overview
- main.py: the core of our model, including the structure and the process of training.
- calculator.py: the code about calculating Poincare embedding and class-aware Ricci curvature weights
- dataloader.py: providing data loading and processing. 
- models/: including the basic layers we used in the main model.

# Environment
Our experimental environments are listed in `environments.yaml`, you can create a virtual environment with conda and run the following order.
```
conda env create -f environments.yaml
```

# Install
Enter the virtual environment and run the `requirements.txt`.
```
pip install -r requirements.txt
```

# Datasets
All the datasets are provided by [pytorch_geometric](https://github.com/ZZy979/pytorch-tutorial).

# Usage
Run the following order to train our model.
```
python main.py
```

## Reference
````
@inproceedings{fu2023hyperimba,
  title={Hyperbolic Geometric Graph Representation Learning for Hierarchy-imbalance Node Classification},
  author={Fu Xingcheng, Wei Yuecen, Sun Qingyun, Yuan Haonan, Wu Jia, Peng Hao and Li Jianxin},
  booktitle={Proceedings of the 2022 World Wide Web Conference},
  year={2022}
}
````
