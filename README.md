
# Non-small Cell Lung Cancer - Parenchymal Attention
Welcome to [ParenchymalAttention](/ParenchymalAttention/) 
## Table of Contents
- [Abstract:](#abstract)
- [Requirements:](#requirements)
- [Getting Started:](#getting-started)
- [Components:](#components)
- [Citation:](#citation)

---
## Abstract:
---
## Requirements: 
Review [requirements.txt](/ParenchymalAttention/requirements.txt) for necessary libraries

---
## Getting Started:
### Training a Model from Scratch:
```
./ParenchymalAttention/ $ python main.py --config /config/training.yaml
```
### Training a Model from Existing network:
```
Not available at this time (check back later :p)
```
### Evaluating a Model on New data:
```
./ParenchymalAttention/ $ python main.py --config /config/training.yaml
```
---
## Components
### [Config:](/ParenchymalAttention/configs/)
Folder containing master yaml files for training and inference protocols for ML scientist to run experiment and benchmarking
### [Networks:](/ParenchymalAttention/ParenchymalAttention/networks/)

### [data:](/ParenchymalAttention/ParenchymalAttention/data/)
### [results:](/ParenchymalAttention/results)
### [Utilities:](/ParenchymalAttention/ParenchymalAttention/utils/)
Folder containing all utility function for visualization of results, saving networks, loading networks
### [bin:](/ParenchymalAttention/ParenchymalAttention/bin/)
Folder designed to contain **.sh** filetypes in order to allow rapid deployment of experiments to benchmark networks, alongside code for statistical analysis.

----
## Citation:
```

```